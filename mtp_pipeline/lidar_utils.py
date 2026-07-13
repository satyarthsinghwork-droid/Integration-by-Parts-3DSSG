from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import ProjectPaths


class PointMAEObjectEncoder(nn.Module):
    """Local geometry encoder used to produce 64 LiDAR tokens of dimension 384."""

    def __init__(self, encoder_channel: int = 384):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        b, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(b * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        global_feature = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([global_feature.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        return torch.max(feature, dim=2)[0].reshape(b, g, self.encoder_channel)


def preprocess_object_points(points: np.ndarray, num_points: int = 1024) -> np.ndarray:
    points = points.T.astype(np.float32)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    radius = np.max(np.linalg.norm(points, axis=1))
    if radius > 0:
        points = points / radius
    n = len(points)
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.random.choice(n, num_points - n, replace=True)
        points = np.concatenate([points, points[idx]], axis=0)
        return points
    return points[idx]


def farthest_point_sampling(points: np.ndarray, num_centers: int = 64):
    n = points.shape[0]
    centers = np.zeros(num_centers, dtype=np.int64)
    distances = np.ones(n) * 1e10
    farthest = np.random.randint(0, n)
    for i in range(num_centers):
        centers[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))
    return points[centers], centers


def group_points(points: np.ndarray, centers: np.ndarray, group_size: int = 32) -> np.ndarray:
    groups = []
    for center in centers:
        dist = np.linalg.norm(points - center, axis=1)
        idx = np.argsort(dist)[:group_size]
        groups.append(points[idx])
    return np.stack(groups, axis=0).astype(np.float32)


def prepare_lidar_tokens(reference_root: Path, output_dir: Path, version: str, checkpoint: Path | None, device: str | None) -> None:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import points_in_box

    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataroot = reference_root / version
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)
    encoder = PointMAEObjectEncoder().to(device_obj).eval()
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        state_dict = state.get("base_model", state.get("model", state))
        encoder.load_state_dict(state_dict, strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(nusc.sample, desc="Preparing LiDAR tokens"):
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_path = nusc.get_sample_data_path(lidar_token)
        pc = LidarPointCloud.from_file(lidar_path)
        _, boxes, _ = nusc.get_sample_data(lidar_token)
        for box in boxes:
            ann = nusc.get("sample_annotation", box.token)
            save_path = output_dir / f"{ann['token']}.pt"
            if save_path.exists():
                continue
            mask = points_in_box(box, pc.points[:3, :])
            object_points = pc.points[:3, mask]
            if object_points.shape[1] < 5:
                continue
            processed = preprocess_object_points(object_points, num_points=1024)
            centers, _ = farthest_point_sampling(processed, num_centers=64)
            groups = group_points(processed, centers, group_size=32)
            groups_tensor = torch.tensor(groups, dtype=torch.float32, device=device_obj).unsqueeze(0)
            with torch.no_grad():
                lidar_tokens = encoder(groups_tensor).squeeze(0).cpu()
            torch.save(
                {
                    "dataset_version": version,
                    "scene_token": sample["scene_token"],
                    "sample_token": sample["token"],
                    "timestamp": sample["timestamp"],
                    "prev_sample_token": sample["prev"],
                    "next_sample_token": sample["next"],
                    "annotation_token": ann["token"],
                    "instance_token": ann["instance_token"],
                    "category_name": box.name,
                    "sample_data_token": lidar_token,
                    "lidar_path": lidar_path,
                    "num_points": object_points.shape[1],
                    "fps_centers": torch.tensor(centers, dtype=torch.float32),
                    "lidar_tokens": lidar_tokens,
                },
                save_path,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Point-MAE-style LiDAR object tokens for nuScenes.")
    parser.add_argument("--reference-root", type=Path, default=ProjectPaths.reference_root)
    parser.add_argument("--output-dir", type=Path, default=ProjectPaths.reference_root / "lidar_patch_tokens")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_lidar_tokens(args.reference_root, args.output_dir, args.version, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
