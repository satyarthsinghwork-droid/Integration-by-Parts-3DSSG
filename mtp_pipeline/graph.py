from __future__ import annotations

from typing import Any

import torch

# The 41 relationship predicates from 3DSSG
RELATION_LABELS = [
    "none", "supported by", "left", "right", "front", "behind", "close by", "inside",
    "bigger than", "smaller than", "higher than", "lower than", "same symmetry as",
    "same as", "attached to", "standing on", "lying on", "hanging on", "connected to",
    "leaning against", "part of", "belonging to", "build in", "standing in", "cover",
    "lying in", "hanging in", "same color", "same material", "same texture", "same shape",
    "same state", "same object type", "messier than", "cleaner than", "fuller than",
    "more closed", "more open", "brighter than", "darker than", "more comfortable than"
]

RELATION_TO_ID = {label: idx for idx, label in enumerate(RELATION_LABELS)}
ID_TO_RELATION = {idx: label for label, idx in RELATION_TO_ID.items()}


def build_3dssg_edges(
    scene_objects: list[dict[str, Any]], 
    ground_truth_relationships: list[dict[str, Any]]
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """
    Build edges directly from the 3DSSG ground truth relationship annotations.
    """
    if not ground_truth_relationships:
        return torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.long), []

    # Create mapping from instance_token to index in the unique objects list
    instance_to_idx = {obj["instance_token"]: idx for idx, obj in enumerate(scene_objects)}
    
    edge_list = []
    label_list = []
    metadata = []
    
    for rel in ground_truth_relationships:
        src = rel["source_instance"]
        tgt = rel["target_instance"]
        predicate = rel["predicate"]
        
        # Only include edges where both nodes actually exist in our extracted objects
        if src in instance_to_idx and tgt in instance_to_idx:
            if predicate not in RELATION_TO_ID:
                predicate = "none"
                
            edge_list.append([instance_to_idx[src], instance_to_idx[tgt]])
            label_list.append(RELATION_TO_ID[predicate])
            metadata.append({
                "source_instance": src,
                "target_instance": tgt,
                "relation": predicate,
            })

    if not edge_list:
        return torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.long), []

    edge_index = torch.tensor(edge_list, dtype=torch.long)
    edge_labels = torch.tensor(label_list, dtype=torch.long)
    return edge_index, edge_labels, metadata




def _object_spatial_vector(obj: dict[str, Any], device: torch.device | str) -> torch.Tensor:
    """Return center, spread, box size, volume, and max side using 3D stats when available.

    Existing databases may only contain 2D RGB boxes, so the function falls back gracefully.
    The resulting vector has 11 values: center(3), spread(3), size(3), volume(1), max_side(1).
    """
    stats = obj.get("point_stats") or obj.get("geometry") or {}
    if stats:
        center = torch.tensor(stats.get("center", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
        spread = torch.tensor(stats.get("std", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
        size = torch.tensor(stats.get("bbox_size", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
        volume = torch.tensor([float(stats.get("volume", torch.prod(size).item()))], dtype=torch.float32, device=device)
        max_side = torch.tensor([float(stats.get("max_side", size.max().item() if size.numel() else 0.0))], dtype=torch.float32, device=device)
        return torch.cat([center, spread, size, volume, max_side])

    bbox = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
    if len(bbox) < 4:
        bbox = [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    center = torch.tensor([(x1 + x2) * 0.5, (y1 + y2) * 0.5, 0.0], dtype=torch.float32, device=device)
    spread = torch.tensor([width * 0.5, height * 0.5, 0.0], dtype=torch.float32, device=device)
    size = torch.tensor([width, height, 1.0], dtype=torch.float32, device=device)
    volume = torch.tensor([width * height], dtype=torch.float32, device=device)
    max_side = torch.tensor([max(width, height)], dtype=torch.float32, device=device)
    return torch.cat([center, spread, size, volume, max_side])


def edge_geometric_features(
    objects: list[dict[str, Any]],
    edge_index: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the 11-D relationship descriptor used by the edge predictor.

    It mirrors the base paper's descriptor: center difference, spread difference, box-size
    difference, log volume ratio, and log max-side ratio.
    """
    if edge_index.numel() == 0:
        return torch.empty(0, 11, dtype=torch.float32, device=device)

    spatial = torch.stack([_object_spatial_vector(obj, device) for obj in objects], dim=0)
    src = spatial[edge_index[:, 0].to(spatial.device)]
    tgt = spatial[edge_index[:, 1].to(spatial.device)]

    center_delta = src[:, 0:3] - tgt[:, 0:3]
    spread_delta = src[:, 3:6] - tgt[:, 3:6]
    size_delta = src[:, 6:9] - tgt[:, 6:9]
    volume_ratio = torch.log((src[:, 9:10].clamp_min(1e-6)) / (tgt[:, 9:10].clamp_min(1e-6)))
    side_ratio = torch.log((src[:, 10:11].clamp_min(1e-6)) / (tgt[:, 10:11].clamp_min(1e-6)))
    geom = torch.cat([center_delta, spread_delta, size_delta, volume_ratio, side_ratio], dim=-1)
    return torch.nan_to_num(geom, nan=0.0, posinf=10.0, neginf=-10.0)


def node_labels_for_objects(objects: list[dict[str, Any]], label_to_id: dict[str, int], device: torch.device | str) -> torch.Tensor:
    return torch.tensor([label_to_id[obj["category_name"]] for obj in objects], dtype=torch.long, device=device)


def edge_representation(node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return node_features.new_zeros((0, node_features.size(-1) * 3))
    source = node_features[edge_index[:, 0]]
    target = node_features[edge_index[:, 1]]
    return torch.cat([source, target, source * target], dim=-1)
