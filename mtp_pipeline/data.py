from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import PipelineConfig, ProjectPaths


class TemporalSceneDataset(Dataset):
    def __init__(self, scene_database_dir: str | Path, max_frames: int | None = None):
        self.scene_database_dir = Path(scene_database_dir)
        self.scene_tokens = sorted([p.stem for p in self.scene_database_dir.glob("*.pt")])
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.scene_tokens)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        token = self.scene_tokens[idx]
        scene_path = self.scene_database_dir / f"{token}.pt"
        scene = torch.load(scene_path, map_location="cpu", weights_only=False)
        if self.max_frames is None:
            return scene
        return {**scene, "frames": scene["frames"][: self.max_frames]}


def scene_database_summary(scene_database_dir: str | Path) -> dict[str, Any]:
    scene_database_dir = Path(scene_database_dir)
    category_counts: Counter[str] = Counter()
    camera_counts: Counter[str] = Counter()
    frame_counts = []
    object_counts = []
    track_lengths = []

    for scene_path in scene_database_dir.glob("*.pt"):
        scene = torch.load(scene_path, map_location="cpu", weights_only=False)
        frames = scene.get("frames", [])

        frame_counts.append(len(frames))

        tracklets = Counter()
        for frame in frames:
            timestamp = frame.get("timestamp", "")
            camera_counts[timestamp] += 1
            objects = frame.get("objects", [])
            object_counts.append(len(objects))
            for obj in objects:
                cat = obj.get("category_name", "unknown")
                instance = obj.get("instance_token", "unknown")
                category_counts[cat] += 1
                tracklets[instance] += 1
        
        for count in tracklets.values():
            track_lengths.append(count)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "num_scenes": len(list(scene_database_dir.glob("*.pt"))),
        "total_frames": sum(frame_counts),
        "total_objects": sum(object_counts),
        "avg_frames_per_scene": avg(frame_counts),
        "avg_objects_per_frame": avg(object_counts),
        "avg_track_length": avg(track_lengths),
        "unique_categories": len(category_counts),
        "top_categories": category_counts.most_common(10),
    }


def build_label_maps(scene_database_dir: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    scene_database_dir = Path(scene_database_dir)
    categories = set()
    for scene_path in scene_database_dir.glob("*.pt"):
        scene = torch.load(scene_path, map_location="cpu", weights_only=False)
        for frame in scene.get("frames", []):
            for obj in frame.get("objects", []):
                categories.add(obj.get("category_name", "unknown"))

    # Also make sure we include everything from RELATIONS
    sorted_cats = sorted(categories)
    if "unknown" not in sorted_cats:
        sorted_cats.insert(0, "unknown")

    label_to_id = {label: i for i, label in enumerate(sorted_cats)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_to_id, id_to_label


def compute_predicate_weights(scene_database_dir: str | Path, relation_labels: list[str]) -> torch.Tensor:
    """Computes alpha weights (inverse sqrt frequency) for each relation class to balance long-tail classes."""
    scene_database_dir = Path(scene_database_dir)
    freq = {label: 0 for label in relation_labels}
    
    for scene_path in scene_database_dir.glob("*.pt"):
        scene = torch.load(scene_path, map_location="cpu", weights_only=False)
        for rel in scene.get("relationships", []):
            pred = rel.get("predicate")
            if pred in freq:
                freq[pred] += 1
                
    counts = torch.tensor([freq[label] for label in relation_labels], dtype=torch.float32)
    # 1.0 / sqrt(frequency) smoothing. clamp_min prevents div by zero.
    alpha = 1.0 / torch.sqrt(counts.clamp_min(1.0))
    # Normalize so mean weight is 1.0
    alpha = alpha / alpha.mean()
    return alpha


def stack_frame_tensors(frame: dict[str, Any], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    rgb = torch.stack([obj["patch_tokens"].float().to(device) for obj in frame["objects"]])
    lidar = torch.stack([obj["lidar_tokens"].float().to(device) for obj in frame["objects"]])
    text = torch.stack([obj["text_features"].float().to(device) for obj in frame["objects"]])
    return rgb, lidar, text

