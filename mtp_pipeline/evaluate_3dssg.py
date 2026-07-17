from __future__ import annotations

import argparse
import heapq
import json
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import PipelineConfig, ProjectPaths
from .data import TemporalSceneDataset, build_label_maps, load_label_space
from .graph import build_official_3dssg_edges, edge_geometric_features, node_labels_for_objects
from .models import DynamicSceneGraphModel
from .splits import get_3rscan_splits
from .temporal import process_scene, process_static_scene


TABLE_KS = (20, 50, 100)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _recall_from_ranks(ranks: list[int], k: int) -> float:
    return sum(rank <= k for rank in ranks) / len(ranks) if ranks else 0.0


def _class_mean_recall(ranks_by_class: dict[int, list[int]], k: int) -> float:
    recalls = [_recall_from_ranks(ranks, k) for ranks in ranks_by_class.values() if ranks]
    return _average(recalls)


def _official_triplet_mean_recall(ranks_by_class: dict[int, list[int]], k: int) -> float:
    """Reproduce OCRL's released Table-2 triplet mR implementation.

    The reference helper iterates ``range(cls_matrix.max())`` and consequently
    omits the numerically largest observed predicate ID. Keeping this behavior
    is necessary when comparing against numbers produced by the released code.
    """
    observed = [label for label, ranks in ranks_by_class.items() if ranks]
    if not observed:
        return 0.0
    maximum = max(observed)
    recalls = [_recall_from_ranks(ranks_by_class[label], k) for label in range(maximum) if ranks_by_class.get(label)]
    return _average(recalls)


def _percent_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _percent_tree(item) for key, item in value.items()}
    if isinstance(value, float):
        return round(value * 100.0, 2)
    return value


def _top_product_indices(
    source_probs: torch.Tensor,
    target_probs: torch.Tensor,
    relation_probs: torch.Tensor,
    limit: int,
) -> list[tuple[int, int, int, float]]:
    """Exact top products without materializing 160 x 160 x 26 values."""
    if limit <= 0:
        return []
    source_order = torch.argsort(source_probs, descending=True).tolist()
    target_order = torch.argsort(target_probs, descending=True).tolist()
    relation_order = torch.argsort(relation_probs, descending=True).tolist()
    if not source_order or not target_order or not relation_order:
        return []

    def score(i: int, j: int, k: int) -> float:
        return float(
            source_probs[source_order[i]]
            * target_probs[target_order[j]]
            * relation_probs[relation_order[k]]
        )

    queue = [(-score(0, 0, 0), 0, 0, 0)]
    visited = {(0, 0, 0)}
    result: list[tuple[int, int, int, float]] = []
    while queue and len(result) < limit:
        negative_score, i, j, k = heapq.heappop(queue)
        result.append((source_order[i], target_order[j], relation_order[k], -negative_score))
        for next_i, next_j, next_k in ((i + 1, j, k), (i, j + 1, k), (i, j, k + 1)):
            candidate = (next_i, next_j, next_k)
            if (
                next_i < len(source_order)
                and next_j < len(target_order)
                and next_k < len(relation_order)
                and candidate not in visited
            ):
                visited.add(candidate)
                heapq.heappush(queue, (-score(next_i, next_j, next_k), next_i, next_j, next_k))
    return result


def _top_relation_indices(relation_probs: torch.Tensor, limit: int) -> list[tuple[int, float]]:
    order = torch.argsort(relation_probs, descending=True)[:limit]
    return [(int(index), float(relation_probs[index])) for index in order]


def _global_ranked_candidates(
    node_probs: torch.Tensor,
    relation_probs: torch.Tensor,
    edge_index: torch.Tensor,
    per_edge_limit: int,
    use_nodes: bool,
    global_limit: int = 100,
) -> list[tuple[float, int, int, int, int]]:
    """Mirror the official graph-constraint ranking with a bounded global heap."""
    heap: list[tuple[float, int, int, int, int, int]] = []
    serial = 0
    for edge_id, pair in enumerate(edge_index.tolist()):
        source_index, target_index = pair
        if use_nodes:
            local = _top_product_indices(
                node_probs[source_index],
                node_probs[target_index],
                relation_probs[edge_id],
                per_edge_limit,
            )
            candidates = [
                (score, edge_id, source_label, target_label, relation_label)
                for source_label, target_label, relation_label, score in local
            ]
        else:
            local = _top_relation_indices(relation_probs[edge_id], per_edge_limit)
            candidates = [(score, edge_id, -1, -1, relation_label) for relation_label, score in local]

        for score, current_edge, source_label, target_label, relation_label in candidates:
            item = (score, serial, current_edge, source_label, target_label, relation_label)
            serial += 1
            if len(heap) < global_limit:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    return [
        (score, edge_id, source_label, target_label, relation_label)
        for score, _serial, edge_id, source_label, target_label, relation_label in sorted(heap, reverse=True)
    ]


def _scene_recall(
    node_probs: torch.Tensor,
    relation_probs: torch.Tensor,
    node_targets: torch.Tensor,
    relation_targets: torch.Tensor,
    edge_index: torch.Tensor,
    graph_constraints: bool,
    use_nodes: bool,
) -> tuple[dict[int, float], dict[int, dict[int, float | None]]]:
    """Official Table 3 and Table 10 semantics for one scene."""
    positive_relations = [
        torch.nonzero(target > 0.5, as_tuple=False).flatten().tolist()
        for target in relation_targets
    ]
    positive_edges = sum(bool(labels) for labels in positive_relations)
    empty_by_k = {
        k: {index: None for index in range(relation_targets.size(1))}
        for k in TABLE_KS
    }
    if positive_edges == 0:
        return {k: 0.0 for k in TABLE_KS}, empty_by_k

    per_edge_limit = 1 if graph_constraints else 1000
    candidates = _global_ranked_candidates(
        node_probs,
        relation_probs,
        edge_index,
        per_edge_limit=per_edge_limit,
        use_nodes=use_nodes,
        global_limit=max(TABLE_KS),
    )

    class_totals = defaultdict(int)
    for labels in positive_relations:
        for relation_label in labels:
            class_totals[relation_label] += 1

    recalls: dict[int, float] = {}
    class_recalls_by_k: dict[int, dict[int, float | None]] = {}
    for k in TABLE_KS:
        correct_edges: set[int] = set()
        class_hits = defaultdict(int)
        for _score, edge_id, source_label, target_label, relation_label in candidates[:k]:
            if edge_id in correct_edges or relation_label not in positive_relations[edge_id]:
                continue
            if use_nodes:
                source_gt = int(node_targets[edge_index[edge_id, 0]])
                target_gt = int(node_targets[edge_index[edge_id, 1]])
                if source_label != source_gt or target_label != target_gt:
                    continue
            correct_edges.add(edge_id)
            # Preserve the official mean-recall convention for multi-label pairs.
            for gt_relation in positive_relations[edge_id]:
                class_hits[gt_relation] += 1

        recalls[k] = len(correct_edges) / positive_edges
        class_recalls_by_k[k] = {
            relation_label: (
                class_hits[relation_label] / class_totals[relation_label]
                if class_totals[relation_label]
                else None
            )
            for relation_label in range(relation_targets.size(1))
        }

    return recalls, class_recalls_by_k


def _triplet_ranks_for_scene(
    node_probs: torch.Tensor,
    relation_probs: torch.Tensor,
    node_targets: torch.Tensor,
    relation_targets: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[list[int], dict[int, list[int]]]:
    """Official Table 2 triplet ranks, capped at 101 as in the reference code."""
    ranks: list[int] = []
    ranks_by_relation: dict[int, list[int]] = defaultdict(list)
    for edge_id, pair in enumerate(edge_index.tolist()):
        source_index, target_index = pair
        candidates = _top_product_indices(
            node_probs[source_index],
            node_probs[target_index],
            relation_probs[edge_id],
            limit=101,
        )
        candidate_rank = {
            (source_label, target_label, relation_label): rank
            for rank, (source_label, target_label, relation_label, _score) in enumerate(candidates, start=1)
        }
        gt_relations = torch.nonzero(relation_targets[edge_id] > 0.5, as_tuple=False).flatten().tolist()
        if not gt_relations:
            none_rank = next((rank for rank, item in enumerate(candidates, start=1) if item[3] < 0.5), 102)
            ranks.append(none_rank)
            continue

        source_gt = int(node_targets[source_index])
        target_gt = int(node_targets[target_index])
        relation_ranks = [
            candidate_rank.get((source_gt, target_gt, relation_label), 102)
            for relation_label in gt_relations
        ]
        adjusted_ranks = [rank - counter for counter, rank in enumerate(sorted(relation_ranks))]
        # This pairing intentionally follows the released OCRL code: adjusted
        # sorted ranks are zipped with predicates in annotation order.
        for relation_label, rank in zip(gt_relations, adjusted_ranks):
            ranks_by_relation[relation_label].append(rank)
        ranks.extend(adjusted_ranks)

    return ranks, ranks_by_relation


def _predicate_ranks_for_scene(
    relation_probs: torch.Tensor,
    relation_targets: torch.Tensor,
) -> tuple[list[int], dict[int, list[int]]]:
    ranks: list[int] = []
    ranks_by_relation: dict[int, list[int]] = defaultdict(list)
    for edge_id, scores in enumerate(relation_probs):
        gt_relations = torch.nonzero(relation_targets[edge_id] > 0.5, as_tuple=False).flatten().tolist()
        order = torch.argsort(scores, descending=True)
        if not gt_relations:
            none_rank = next((rank for rank, index in enumerate(order, start=1) if scores[index] < 0.5), 7)
            ranks.append(none_rank)
            continue

        relation_ranks = []
        for relation_label in gt_relations:
            rank = min(int((scores > scores[relation_label]).sum().item()) + 1, 7)
            relation_ranks.append(rank)
        adjusted_ranks = [rank - counter for counter, rank in enumerate(sorted(relation_ranks))]
        for relation_label, rank in zip(gt_relations, adjusted_ranks):
            ranks_by_relation[relation_label].append(rank)
        ranks.extend(adjusted_ranks)
    return ranks, ranks_by_relation


def _object_ranks_for_scene(
    node_probs: torch.Tensor,
    node_targets: torch.Tensor,
) -> tuple[list[int], dict[int, list[int]]]:
    ranks = []
    ranks_by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(node_targets.tolist()):
        rank = min(int((node_probs[index] > node_probs[index, target]).sum().item()) + 1, 12)
        ranks.append(rank)
        ranks_by_class[target].append(rank)
    return ranks, ranks_by_class


def _pool_node_features(results: dict[str, Any]) -> torch.Tensor:
    embeddings = results["temporal_embeddings"]
    mask = results["mask"]
    pooled = []
    for index in range(len(results["instance_tokens"])):
        valid = mask[index]
        pooled.append(embeddings[index, valid].mean(dim=0) if valid.any() else embeddings[index, 0])
    return torch.stack(pooled)


def _unique_objects(scene: dict[str, Any], instance_tokens: list[str]) -> list[dict[str, Any]]:
    by_instance = {}
    for frame in scene.get("frames", []):
        for obj in frame.get("objects", []):
            by_instance.setdefault(str(obj["instance_token"]), obj)
    return [dict(by_instance[token]) for token in instance_tokens if token in by_instance]


@torch.no_grad()
def evaluate_model(args: argparse.Namespace) -> None:
    paths = ProjectPaths(reference_root=args.reference_root, output_root=args.output_root)
    database_dir = args.database if args.database is not None else paths.output_root / "3rscan_official_static_database"

    label_space = load_label_space(database_dir)
    if label_space is None:
        raise ValueError(
            "This database is missing label_space.json. The legacy database cannot be evaluated "
            "as an official base-paper comparison."
        )
    object_labels, relation_labels, protocol = label_space
    if len(object_labels) != 160 or len(relation_labels) != 26:
        raise ValueError(
            f"Expected the official 160/26 OCRL label space, found "
            f"{len(object_labels)} objects and {len(relation_labels)} relations."
        )
    label_to_id, _ = build_label_maps(database_dir)

    dataset = TemporalSceneDataset(database_dir)
    _, validation_tokens = get_3rscan_splits(
        dataset.scene_tokens,
        train_scans=args.train_scans,
        val_scans=args.val_scans,
        seed=args.split_seed,
    )
    if not validation_tokens:
        validation_tokens = dataset.scene_tokens
    dataset.scene_tokens = validation_tokens
    if getattr(args, "max_scenes", None) is not None:
        dataset.scene_tokens = dataset.scene_tokens[: args.max_scenes]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_labels = checkpoint.get("label_to_id")
    checkpoint_relations = checkpoint.get("relation_labels")
    if checkpoint_labels != label_to_id or checkpoint_relations != relation_labels:
        raise ValueError(
            "The checkpoint label space does not match this official database. "
            "Train a fresh model; the previous 41-relation checkpoint is not compatible."
        )

    config_values = checkpoint.get("config", {})
    allowed_config = {field.name for field in fields(PipelineConfig)}
    config = PipelineConfig(**{key: value for key, value in config_values.items() if key in allowed_config})
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DynamicSceneGraphModel(
        num_node_classes=len(object_labels),
        num_edge_classes=len(relation_labels),
        dim=config.dim,
        num_parts=config.num_parts,
        num_heads=config.num_heads,
        fusion_layers=config.fusion_layers,
        temporal_layers=config.temporal_layers,
        dropout=config.dropout,
        graph_use_text=config.graph_use_text,
        graph_input_mode=config.graph_input_mode,
        use_rgb_token_mask=config.use_rgb_token_mask,
        extended_geometry=config.extended_geometry,
        graph_context=config.graph_context,
        graph_knn_neighbors=config.graph_knn_neighbors,
        conditioned_part_queries=config.conditioned_part_queries,
        hybrid_spatial_init=config.hybrid_spatial_init,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    print(f"Protocol: {protocol}")
    print(f"Evaluating {len(dataset.scene_tokens)} official validation scenes on {device}.")

    object_ranks: list[int] = []
    object_ranks_by_class: dict[int, list[int]] = defaultdict(list)
    predicate_ranks: list[int] = []
    predicate_ranks_by_class: dict[int, list[int]] = defaultdict(list)
    triplet_ranks: list[int] = []
    triplet_ranks_by_class: dict[int, list[int]] = defaultdict(list)
    recall_values = {
        task: {constraint: {k: [] for k in TABLE_KS} for constraint in ("with_graph_constraints", "without_graph_constraints")}
        for task in ("SGCls", "PredCls")
    }
    mean_recall_scene_values = {
        task: {constraint: {relation: {k: [] for k in TABLE_KS} for relation in range(len(relation_labels))}
               for constraint in ("with_graph_constraints", "without_graph_constraints")}
        for task in ("SGCls", "PredCls")
    }

    evaluated_scenes = 0
    for scene in tqdm(dataset, desc="Evaluating official OCRL 3DSSG protocol"):
        if args.mode == "static":
            results = process_static_scene(scene=scene, scene_model=model.object_encoder, device=device)
        else:
            results = process_scene(
                scene=scene,
                scene_model=model.object_encoder,
                association_model=model.association,
                aggregation_model=model.aggregation,
                temporal_model=model.temporal,
                temporal_loss=lambda _embeddings, _mask: torch.zeros((), device=device),
                device=device,
            )
        if results is None:
            continue

        instance_tokens = results["instance_tokens"]
        objects = _unique_objects(scene, instance_tokens)
        if len(objects) != len(instance_tokens) or len(objects) < 2:
            continue

        edge_index, relation_targets, _metadata = build_official_3dssg_edges(
            objects,
            scene.get("relationships", []),
            relation_labels=relation_labels,
            max_negative_ratio=None,
        )
        if edge_index.numel() == 0:
            continue

        pooled_nodes = _pool_node_features(results)
        edge_index = edge_index.to(device)
        relation_targets = relation_targets.to(device)
        node_targets = node_labels_for_objects(objects, label_to_id, device)
        geometry = edge_geometric_features(objects, edge_index, device, extended_geometry=config.extended_geometry)

        node_logits, edge_logits, _geometry_reconstruction = model.predict_graph(
            pooled_nodes,
            edge_index,
            geometry,
        )
        node_probs = torch.softmax(node_logits, dim=-1).cpu()
        relation_probs = torch.sigmoid(edge_logits).cpu()
        # Ranking runs over many small candidate sets. Moving probabilities once
        # avoids thousands of GPU scalar synchronizations inside Python heaps.
        node_targets = node_targets.cpu()
        relation_targets = relation_targets.cpu()
        edge_index = edge_index.cpu()

        scene_object_ranks, scene_object_by_class = _object_ranks_for_scene(node_probs, node_targets)
        scene_predicate_ranks, scene_predicate_by_class = _predicate_ranks_for_scene(relation_probs, relation_targets)
        scene_triplet_ranks, scene_triplet_by_class = _triplet_ranks_for_scene(
            node_probs,
            relation_probs,
            node_targets,
            relation_targets,
            edge_index,
        )
        object_ranks.extend(scene_object_ranks)
        predicate_ranks.extend(scene_predicate_ranks)
        triplet_ranks.extend(scene_triplet_ranks)
        for key, values in scene_object_by_class.items():
            object_ranks_by_class[key].extend(values)
        for key, values in scene_predicate_by_class.items():
            predicate_ranks_by_class[key].extend(values)
        for key, values in scene_triplet_by_class.items():
            triplet_ranks_by_class[key].extend(values)

        for task, use_nodes in (("SGCls", True), ("PredCls", False)):
            for constraint, graph_constraints in (
                ("with_graph_constraints", True),
                ("without_graph_constraints", False),
            ):
                scene_recalls, scene_class_recalls = _scene_recall(
                    node_probs,
                    relation_probs,
                    node_targets,
                    relation_targets,
                    edge_index,
                    graph_constraints=graph_constraints,
                    use_nodes=use_nodes,
                )
                for k in TABLE_KS:
                    recall_values[task][constraint][k].append(scene_recalls[k])
                    for relation, value in scene_class_recalls[k].items():
                        if value is not None:
                            mean_recall_scene_values[task][constraint][relation][k].append(value)

        evaluated_scenes += 1

    table_2 = {
        "Object": {
            "R@1": _recall_from_ranks(object_ranks, 1),
            "R@5": _recall_from_ranks(object_ranks, 5),
            "mR@1": _class_mean_recall(object_ranks_by_class, 1),
            "mR@5": _class_mean_recall(object_ranks_by_class, 5),
        },
        "Predicate": {
            "R@1": _recall_from_ranks(predicate_ranks, 1),
            "R@3": _recall_from_ranks(predicate_ranks, 3),
            "mR@1": _class_mean_recall(predicate_ranks_by_class, 1),
            "mR@3": _class_mean_recall(predicate_ranks_by_class, 3),
        },
        "Triplet": {
            "R@50": _recall_from_ranks(triplet_ranks, 50),
            "R@100": _recall_from_ranks(triplet_ranks, 100),
            "mR@50": _official_triplet_mean_recall(triplet_ranks_by_class, 50),
            "mR@100": _official_triplet_mean_recall(triplet_ranks_by_class, 100),
        },
    }
    table_3 = {
        task: {
            constraint: {f"R@{k}": _average(values) for k, values in metrics.items()}
            for constraint, metrics in constraints.items()
        }
        for task, constraints in recall_values.items()
    }
    table_10 = {
        task: {
            constraint: {
                f"mR@{k}": _average(
                    [
                        _average(scene_values)
                        for relation, values_by_k in relations.items()
                        if (scene_values := values_by_k[k])
                    ]
                )
                for k in TABLE_KS
            }
            for constraint, relations in constraints.items()
        }
        for task, constraints in mean_recall_scene_values.items()
    }
    summary = {
        "protocol": protocol,
        "object_classes": len(object_labels),
        "positive_relation_classes": len(relation_labels),
        "evaluated_scenes": evaluated_scenes,
        "base_paper_table_2": table_2,
        "base_paper_table_3": table_3,
        "base_paper_table_10_mean_recall": table_10,
    }

    print("\n--- Official OCRL/3DSSG Protocol Results (percent) ---")
    print(json.dumps(_percent_tree(summary), indent=4))
    output_path = getattr(args, "result_output", None)
    if output_path is None:
        output_path = args.checkpoint.parent / f"evaluation_official_{args.checkpoint.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on the official OCRL/3DSSG 160/26 multi-label protocol.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=ProjectPaths().reference_root)
    parser.add_argument("--output-root", type=Path, default=ProjectPaths().output_root)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--mode", choices=["static", "temporal"], default="static")
    parser.add_argument("--train-scans", type=Path, default=Path("official_splits/train_scans.txt"))
    parser.add_argument("--val-scans", type=Path, default=Path("official_splits/validation_scans.txt"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional small evaluation slice for a smoke test.")
    parser.add_argument("--result-output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_model(parse_args())
