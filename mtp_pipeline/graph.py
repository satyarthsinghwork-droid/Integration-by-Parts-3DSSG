from __future__ import annotations

import hashlib
import random
from typing import Any

import torch


# Exact positive predicate ordering from the official OCRL 3DSSG subset.
RELATION_LABELS = [
    "attached to",
    "behind",
    "belonging to",
    "bigger than",
    "build in",
    "close by",
    "connected to",
    "cover",
    "front",
    "hanging in",
    "hanging on",
    "higher than",
    "inside",
    "leaning against",
    "left",
    "lower than",
    "lying in",
    "lying on",
    "part of",
    "right",
    "same as",
    "same symmetry as",
    "smaller than",
    "standing in",
    "standing on",
    "supported by",
]
RELATION_TO_ID = {label: index for index, label in enumerate(RELATION_LABELS)}
ID_TO_RELATION = {index: label for label, index in RELATION_TO_ID.items()}


def build_official_3dssg_edges(
    scene_objects: list[dict[str, Any]],
    ground_truth_relationships: list[dict[str, Any]],
    relation_labels: list[str] = RELATION_LABELS,
    max_negative_ratio: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Build all directed pairs and official 26-way multi-hot relation targets.

    OCRL's 3DSSG protocol has no learned none class. A pair with no annotation
    has an all-zero target, while a pair may have several positive predicates.
    """
    num_relations = len(relation_labels)
    if len(scene_objects) < 2:
        return (
            torch.empty(0, 2, dtype=torch.long),
            torch.empty(0, num_relations, dtype=torch.float32),
            [],
        )

    relation_to_id = {label: index for index, label in enumerate(relation_labels)}
    instance_to_idx = {str(obj["instance_token"]): index for index, obj in enumerate(scene_objects)}
    labels_by_pair: dict[tuple[int, int], set[int]] = {}
    for relation in ground_truth_relationships:
        source = str(relation.get("source_instance", ""))
        target = str(relation.get("target_instance", ""))
        predicate = relation.get("predicate")
        if source == target or source not in instance_to_idx or target not in instance_to_idx:
            continue
        if predicate not in relation_to_id:
            continue
        pair = (instance_to_idx[source], instance_to_idx[target])
        labels_by_pair.setdefault(pair, set()).add(relation_to_id[predicate])

    positives: list[tuple[list[int], torch.Tensor, dict[str, Any]]] = []
    negatives: list[tuple[list[int], torch.Tensor, dict[str, Any]]] = []
    for source_index, source_obj in enumerate(scene_objects):
        for target_index, target_obj in enumerate(scene_objects):
            if source_index == target_index:
                continue
            pair = (source_index, target_index)
            positive_ids = sorted(labels_by_pair.get(pair, set()))
            target = torch.zeros(num_relations, dtype=torch.float32)
            if positive_ids:
                target[positive_ids] = 1.0
            item = (
                [source_index, target_index],
                target,
                {
                    "source_instance": str(source_obj["instance_token"]),
                    "target_instance": str(target_obj["instance_token"]),
                    "relations": [relation_labels[index] for index in positive_ids],
                    "is_positive": bool(positive_ids),
                },
            )
            if positive_ids:
                positives.append(item)
            else:
                negatives.append(item)

    if max_negative_ratio is not None and positives:
        # Deterministic per-scene negative sampling prevents all-zero labels
        # from dominating the relation objective while keeping runs reproducible.
        seed_material = "|".join(str(obj["instance_token"]) for obj in scene_objects)
        seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        random.Random(seed).shuffle(negatives)
        negatives = negatives[: max_negative_ratio * len(positives)]
    selected = positives + negatives
    if not selected:
        return (
            torch.empty(0, 2, dtype=torch.long),
            torch.empty(0, num_relations, dtype=torch.float32),
            [],
        )

    edge_list, targets, metadata = zip(*selected)
    return torch.tensor(edge_list, dtype=torch.long), torch.stack(list(targets)), list(metadata)


# Retained as a compatibility alias for local notebooks. It now follows the
# official multi-label protocol rather than the earlier 41-class softmax path.
def build_candidate_3dssg_edges(
    scene_objects: list[dict[str, Any]],
    ground_truth_relationships: list[dict[str, Any]],
    max_negative_ratio: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    return build_official_3dssg_edges(
        scene_objects,
        ground_truth_relationships,
        relation_labels=RELATION_LABELS,
        max_negative_ratio=max_negative_ratio,
    )


def build_3dssg_edges(
    scene_objects: list[dict[str, Any]],
    ground_truth_relationships: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    return build_official_3dssg_edges(
        scene_objects,
        ground_truth_relationships,
        relation_labels=RELATION_LABELS,
        max_negative_ratio=0,
    )


def _object_spatial_vector(obj: dict[str, Any], device: torch.device | str) -> torch.Tensor:
    """Return world-space centre, spread, box size, volume and max side."""
    stats = obj.get("point_stats") or obj.get("geometry")
    if not stats:
        raise ValueError(
            "3DSSG graph training requires world-space point_stats; rebuild the corrected static database."
        )
    required = ("center", "std", "bbox_size", "volume", "max_side")
    missing = [name for name in required if name not in stats]
    if missing:
        raise ValueError(f"Incomplete world-space geometry: {missing}")
    center = torch.tensor(stats["center"], dtype=torch.float32, device=device)
    spread = torch.tensor(stats["std"], dtype=torch.float32, device=device)
    size = torch.tensor(stats["bbox_size"], dtype=torch.float32, device=device)
    volume = torch.tensor([float(stats["volume"])], dtype=torch.float32, device=device)
    max_side = torch.tensor([float(stats["max_side"])], dtype=torch.float32, device=device)
    return torch.cat([center, spread, size, volume, max_side])

def edge_geometric_features(
    objects: list[dict[str, Any]],
    edge_index: torch.Tensor,
    device: torch.device | str,
    extended_geometry: bool = False,
) -> torch.Tensor:
    """Build directed spatial features; v3 appends distance, direction and AABB IoU."""
    geom_dim = 16 if extended_geometry else 11
    if edge_index.numel() == 0:
        return torch.empty(0, geom_dim, dtype=torch.float32, device=device)

    spatial = torch.stack([_object_spatial_vector(obj, device) for obj in objects], dim=0)
    source = spatial[edge_index[:, 0].to(spatial.device)]
    target = spatial[edge_index[:, 1].to(spatial.device)]
    center_delta = source[:, 0:3] - target[:, 0:3]
    spread_delta = source[:, 3:6] - target[:, 3:6]
    size_delta = source[:, 6:9] - target[:, 6:9]
    volume_ratio = torch.log(source[:, 9:10].clamp_min(1e-6) / target[:, 9:10].clamp_min(1e-6))
    side_ratio = torch.log(source[:, 10:11].clamp_min(1e-6) / target[:, 10:11].clamp_min(1e-6))
    features = [center_delta, spread_delta, size_delta, volume_ratio, side_ratio]
    if extended_geometry:
        distance = torch.linalg.vector_norm(center_delta, dim=-1, keepdim=True)
        direction = center_delta / distance.clamp_min(1e-6)
        source_min, source_max = source[:, 0:3] - source[:, 6:9] / 2, source[:, 0:3] + source[:, 6:9] / 2
        target_min, target_max = target[:, 0:3] - target[:, 6:9] / 2, target[:, 0:3] + target[:, 6:9] / 2
        intersection = (torch.minimum(source_max, target_max) - torch.maximum(source_min, target_min)).clamp_min(0)
        intersection_volume = intersection.prod(dim=-1, keepdim=True)
        source_volume = source[:, 6:9].clamp_min(1e-6).prod(dim=-1, keepdim=True)
        target_volume = target[:, 6:9].clamp_min(1e-6).prod(dim=-1, keepdim=True)
        iou = intersection_volume / (source_volume + target_volume - intersection_volume).clamp_min(1e-6)
        features.extend([distance, direction, iou])
    return torch.nan_to_num(torch.cat(features, dim=-1), nan=0.0, posinf=10.0, neginf=-10.0)

def node_labels_for_objects(
    objects: list[dict[str, Any]],
    label_to_id: dict[str, int],
    device: torch.device | str,
) -> torch.Tensor:
    missing = sorted({obj.get("category_name") for obj in objects if obj.get("category_name") not in label_to_id})
    if missing:
        raise ValueError(f"Objects outside the saved label space: {missing[:5]}")
    return torch.tensor(
        [label_to_id[obj["category_name"]] for obj in objects],
        dtype=torch.long,
        device=device,
    )


def edge_representation(node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return node_features.new_zeros((0, node_features.size(-1) * 3))
    source = node_features[edge_index[:, 0]]
    target = node_features[edge_index[:, 1]]
    return torch.cat([source, target, source * target], dim=-1)
