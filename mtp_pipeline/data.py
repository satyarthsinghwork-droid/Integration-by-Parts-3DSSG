from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


LABEL_SPACE_FILE = "label_space.json"


class TemporalSceneDataset(Dataset):
    def __init__(self, scene_database_dir: str | Path, max_frames: int | None = None):
        self.scene_database_dir = Path(scene_database_dir)
        self.scene_tokens = sorted([path.stem for path in self.scene_database_dir.glob("*.pt")])
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.scene_tokens)

    def __getitem__(self, index: int) -> dict[str, Any]:
        token = self.scene_tokens[index]
        scene = torch.load(self.scene_database_dir / f"{token}.pt", map_location="cpu", weights_only=False)
        if self.max_frames is None:
            return scene
        return {**scene, "frames": scene["frames"][: self.max_frames]}


def load_label_space(scene_database_dir: str | Path) -> tuple[list[str], list[str], str] | None:
    """Load the immutable labels written by the official database builder."""
    manifest_path = Path(scene_database_dir) / LABEL_SPACE_FILE
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    objects = list(manifest.get("object_labels", []))
    relations = list(manifest.get("relation_labels", []))
    if not objects or not relations:
        raise ValueError(f"Invalid label-space manifest: {manifest_path}")
    return objects, relations, str(manifest.get("protocol", "unspecified"))


def scene_database_summary(scene_database_dir: str | Path) -> dict[str, Any]:
    scene_database_dir = Path(scene_database_dir)
    category_counts: Counter[str] = Counter()
    frame_counts = []
    object_counts = []
    track_lengths = []

    for scene_path in scene_database_dir.glob("*.pt"):
        scene = torch.load(scene_path, map_location="cpu", weights_only=False)
        frames = scene.get("frames", [])
        frame_counts.append(len(frames))

        tracklets = Counter()
        for frame in frames:
            objects = frame.get("objects", [])
            object_counts.append(len(objects))
            for obj in objects:
                category_counts[obj.get("category_name", "unknown")] += 1
                tracklets[obj.get("instance_token", "unknown")] += 1
        track_lengths.extend(tracklets.values())

    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    label_space = load_label_space(scene_database_dir)
    return {
        "num_scenes": len(list(scene_database_dir.glob("*.pt"))),
        "total_frames": sum(frame_counts),
        "total_objects": sum(object_counts),
        "avg_frames_per_scene": average(frame_counts),
        "avg_objects_per_frame": average(object_counts),
        "avg_track_length": average(track_lengths),
        "unique_categories": len(category_counts),
        "top_categories": category_counts.most_common(10),
        "label_protocol": label_space[2] if label_space else "legacy inferred labels",
        "configured_object_classes": len(label_space[0]) if label_space else len(category_counts),
        "configured_relation_classes": len(label_space[1]) if label_space else None,
    }


def build_label_maps(scene_database_dir: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Return canonical labels when an official manifest is available."""
    scene_database_dir = Path(scene_database_dir)
    label_space = load_label_space(scene_database_dir)
    if label_space is not None:
        categories = label_space[0]
    else:
        categories_set = set()
        for scene_path in scene_database_dir.glob("*.pt"):
            scene = torch.load(scene_path, map_location="cpu", weights_only=False)
            for frame in scene.get("frames", []):
                for obj in frame.get("objects", []):
                    categories_set.add(obj.get("category_name", "unknown"))
        categories = sorted(categories_set)
        if "unknown" not in categories:
            categories.insert(0, "unknown")

    label_to_id = {label: index for index, label in enumerate(categories)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def compute_predicate_weights(
    scene_database_dir: str | Path,
    relation_labels: list[str],
    scene_tokens: list[str] | None = None,
) -> torch.Tensor:
    """Compute positive-label weights for the official multi-label relation loss."""
    scene_database_dir = Path(scene_database_dir)
    frequency = {label: 0 for label in relation_labels}
    paths = (
        [scene_database_dir / f"{token}.pt" for token in scene_tokens]
        if scene_tokens is not None
        else list(scene_database_dir.glob("*.pt"))
    )

    for scene_path in paths:
        if not scene_path.exists():
            continue
        scene = torch.load(scene_path, map_location="cpu", weights_only=False)
        for relation in scene.get("relationships", []):
            predicate = relation.get("predicate")
            if predicate in frequency:
                frequency[predicate] += 1

    counts = torch.tensor([frequency[label] for label in relation_labels], dtype=torch.float32)
    weights = 1.0 / torch.sqrt(counts.clamp_min(1.0))
    return weights / weights.mean()


def stack_frame_tensors(
    frame: dict[str, Any],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    rgb = torch.stack([obj["patch_tokens"].float().to(device) for obj in frame["objects"]])
    lidar = torch.stack([obj["lidar_tokens"].float().to(device) for obj in frame["objects"]])
    text = torch.stack([obj["text_features"].float().to(device) for obj in frame["objects"]])
    return rgb, lidar, text
