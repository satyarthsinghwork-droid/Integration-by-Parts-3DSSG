from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .config import ProjectPaths
from .protocol_3dssg import write_official_objects_manifest


def load_intrinsics(info_path: str) -> np.ndarray:
    with open(info_path, "r") as f:
        for line in f:
            if line.startswith("m_calibrationColorIntrinsic"):
                parts = line.strip().split(" = ")[1].split()
                matrix = np.array([float(x) for x in parts]).reshape(4, 4)
                return matrix[:3, :3]
    return np.eye(3)


def project_points(points_3d: np.ndarray, pose_4x4: np.ndarray, intrinsics_3x3: np.ndarray, width: int, height: int):
    # points_3d: (N, 3)
    # pose_4x4: camera to world matrix. We need world to camera.
    world_to_cam = np.linalg.inv(pose_4x4)
    
    # Homogeneous coordinates
    points_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # Transform to camera space
    points_cam = (world_to_cam @ points_h.T).T  # (N, 4)
    points_cam = points_cam[:, :3]
    
    # Keep only points in front of the camera (Z > 0)
    # Note: 3RScan uses OpenGL coordinate system where Z is negative in front of camera?
    # Actually, standard 3RScan uses Z negative for depth. Let's filter Z < 0.
    valid_mask = points_cam[:, 2] < 0
    if not np.any(valid_mask):
        return None
        
    points_cam = points_cam[valid_mask]
    
    # Project to 2D
    # Flip Z to positive for standard pinhole projection
    points_cam[:, 2] = -points_cam[:, 2]
    
    points_2d = (intrinsics_3x3 @ points_cam.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    
    # Check if points are within image boundaries
    in_img_mask = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
    )
    
    # If less than 10% of object points are visible, skip it
    if np.sum(in_img_mask) < (points_3d.shape[0] * 0.1) or np.sum(in_img_mask) < 20:
        return None
        
    valid_points_2d = points_2d[in_img_mask]
    
    x1, y1 = np.min(valid_points_2d, axis=0)
    x2, y2 = np.max(valid_points_2d, axis=0)
    
    # Expand slightly
    padding = 5
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(width - 1, int(x2) + padding)
    y2 = min(height - 1, int(y2) + padding)
    
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None
        
    return [x1, y1, x2, y2]


def prepare_3rscan_rgb_tokens(
    scan_dir: Path, 
    objects_json: Path,
    output_dir: Path, 
    model_name: str, 
    device: str | None
) -> None:
    from transformers import CLIPProcessor, CLIPVisionModel
    
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = CLIPProcessor.from_pretrained(model_name)
    vision_encoder = CLIPVisionModel.from_pretrained(model_name).to(device_obj).eval()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(objects_json, "r") as f:
        data = json.load(f)
    scans = data.get("scans", [])

    @torch.no_grad()
    def extract_patch_tokens(crop_rgb):
        pil_image = Image.fromarray(np.ascontiguousarray(crop_rgb))
        inputs = processor(images=pil_image, return_tensors="pt").to(device_obj)
        hidden = vision_encoder(**inputs).last_hidden_state
        return hidden[:, 1:, :].squeeze(0).cpu()

    import tempfile
    
    for scan in tqdm(scans, desc="Extracting continuous RGB frames for 3RScan"):
        scan_id = scan["scan"]
        scan_path = scan_dir / scan_id
        seq_zip_path = scan_path / "sequence.zip"
        ply_path = scan_path / "labels.instances.annotated.v2.ply"
        
        if not seq_zip_path.exists() or not ply_path.exists():
            continue
            
        save_path = output_dir / f"{scan_id}.pt"
        if save_path.exists():
            continue
            
        # 1. Parse PLY for object 3D points
        with open(ply_path, "r") as f:
            lines = f.readlines()
        header_end = 0
        num_vertices = 0
        for i, line in enumerate(lines):
            if line.startswith("element vertex"):
                num_vertices = int(line.strip().split()[-1])
            elif line.startswith("end_header"):
                header_end = i + 1
                break
        
        try:
            vertex_data = np.loadtxt(lines[header_end : header_end + num_vertices], dtype=np.float32, usecols=(0, 1, 2, 6))
        except Exception:
            continue
            
        points_3d = vertex_data[:, :3]
        object_ids = vertex_data[:, 3].astype(np.int32)
        
        object_points_map = {}
        for obj in scan.get("objects", []):
            obj_id = int(obj["id"])
            mask = (object_ids == obj_id)
            if np.sum(mask) > 10:
                object_points_map[obj_id] = points_3d[mask]

        # 2. Extract video frames from zip
        temp_extract_dir = output_dir / "temp_extract"
        temp_extract_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_extract_dir) as tmp_dir:
            with zipfile.ZipFile(seq_zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
                
            info_file = os.path.join(tmp_dir, "_info.txt")
            if not os.path.exists(info_file):
                continue
            intrinsics = load_intrinsics(info_file)
            
            # Find all color frames
            color_files = sorted([f for f in os.listdir(tmp_dir) if f.endswith(".color.jpg")])
            
            # Sample 20 continuous frames across the video to simulate temporal tracking
            if len(color_files) > 20:
                indices = np.linspace(0, len(color_files)-1, 20, dtype=int)
                sampled_files = [color_files[i] for i in indices]
            else:
                sampled_files = color_files
                
            frames_data = []
            
            for f_name in sampled_files:
                frame_idx = f_name.split(".")[0].split("-")[1]
                pose_file = os.path.join(tmp_dir, f"frame-{frame_idx}.pose.txt")
                
                if not os.path.exists(pose_file):
                    continue
                    
                pose_4x4 = np.loadtxt(pose_file)
                if np.all(pose_4x4 == 0) or np.isinf(pose_4x4).any():
                    continue # Invalid pose
                    
                img_path = os.path.join(tmp_dir, f_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                
                frame_objects = []
                for obj_id, pts in object_points_map.items():
                    bbox = project_points(pts, pose_4x4, intrinsics, width, height)
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        crop = image[y1:y2, x1:x2]
                        patch_tokens = extract_patch_tokens(crop)
                        
                        frame_objects.append({
                            "instance_token": str(obj_id),
                            "bbox": bbox,
                            "patch_tokens": patch_tokens
                        })
                        
                if frame_objects:
                    frames_data.append({
                        "timestamp": int(frame_idx),
                        "sample_token": f"{scan_id}_{frame_idx}",
                        "objects": frame_objects
                    })
            
            if frames_data:
                torch.save({
                    "dataset_version": "3rscan",
                    "scene_token": scan_id,
                    "frames": frames_data
                }, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract sampled RGB object views for static 3DSSG.")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--objects-json", type=Path, default=None)
    parser.add_argument("--official-subset-dir", type=Path, default=Path("official_splits"))
    parser.add_argument("--output-dir", type=Path, default=ProjectPaths().reference_root / "3rscan_rgb_temporal")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    objects_json = args.objects_json
    if objects_json is None:
        objects_json = write_official_objects_manifest(
            args.official_subset_dir,
            args.output_dir.parent / "official_objects_manifest.json",
        )
    prepare_3rscan_rgb_tokens(args.scan_dir, objects_json, args.output_dir, args.model_name, args.device)


if __name__ == "__main__":
    main()
