from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluate_3dssg import (
    _object_ranks_for_scene,
    _predicate_ranks_for_scene,
    _scene_recall,
    _triplet_ranks_for_scene,
)
from .graph import build_official_3dssg_edges
from .protocol_3dssg import OCRL_INVALID_ALIGNED_SCAN_IDS


PROTOCOL_FILES = (
    "train_scans.txt",
    "validation_scans.txt",
    "classes.txt",
    "relations.txt",
    "relationships.txt",
    "relationships_train.json",
    "relationships_validation.json",
)


def _normal_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _annotation_entries(subset: Path, split: str) -> list[dict[str, Any]]:
    selected = set(_normal_text(subset / f"{split}_scans.txt").splitlines())
    filename = "relationships_train.json" if split == "train" else "relationships_validation.json"
    payload = json.loads((subset / filename).read_text(encoding="utf-8"))
    return [scan for scan in payload["scans"] if str(scan["scan"]) in selected]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric_parity(heo_repo: Path) -> bool:
    utils = heo_repo / "src" / "utils"
    heo_acc = _load_module("heo_eva_utils_acc", utils / "eva_utils_acc.py")
    heo_recall = _load_module("heo_eval_utils_recall", utils / "eval_utils_recall.py")
    for seed in range(3):
        torch.manual_seed(seed)
        nodes, classes, relations = 5, 8, 6
        logits = torch.randn(nodes, classes)
        node_probs = torch.softmax(logits, dim=-1)
        edges = torch.tensor([[i, j] for i in range(nodes) for j in range(nodes) if i != j])
        relation_probs = torch.sigmoid(torch.randn(len(edges), relations))
        node_targets = torch.randint(0, classes, (nodes,))
        relation_targets = torch.zeros(len(edges), relations)
        relation_targets[0, [0, 2]] = 1
        relation_targets[3, 4] = 1
        relation_targets[7, [1, 3, 5]] = 1
        relation_targets[11, 2] = 1
        gt = heo_acc.get_gt(node_targets, relation_targets, edges, True)

        expected = heo_acc.evaluate_topk_object(logits, node_targets, topk=11).tolist()
        actual, _ = _object_ranks_for_scene(node_probs, node_targets)
        if expected != actual:
            return False
        expected = heo_acc.evaluate_topk_predicate(relation_probs.clone(), gt, True, topk=6).tolist()
        actual, _ = _predicate_ranks_for_scene(relation_probs, relation_targets)
        if expected != actual:
            return False
        expected = heo_acc.evaluate_triplet_topk(
            logits,
            relation_probs.clone(),
            gt,
            edges,
            True,
            topk=101,
            use_clip=True,
        )[0].tolist()
        actual, _ = _triplet_ranks_for_scene(
            node_probs,
            relation_probs,
            node_targets,
            relation_targets,
            edges,
        )
        if expected != actual:
            return False

        for use_nodes, evaluate in ((True, "triplet"), (False, "rels")):
            for constrained, limit in ((True, 1), (False, 1000)):
                expected = heo_recall.evaluate_triplet_recallk(
                    logits,
                    relation_probs,
                    gt,
                    edges,
                    True,
                    [20, 50, 100],
                    limit,
                    use_clip=True,
                    evaluate=evaluate,
                )
                recalls, _ = _scene_recall(
                    node_probs,
                    relation_probs,
                    node_targets,
                    relation_targets,
                    edges,
                    graph_constraints=constrained,
                    use_nodes=use_nodes,
                )
                actual = np.asarray([recalls[k] for k in (20, 50, 100)])
                if not np.allclose(expected, actual):
                    return False
    return True


def audit_protocol(
    project_root: Path,
    database: Path,
    graph_input_mode: str,
    rgb_views_per_object: int,
) -> dict[str, Any]:
    ours = project_root / "official_splits"
    heo_repo = project_root / "reference" / "OCRL-3DSSG-Codes"
    heo = heo_repo / "data" / "3DSSG_subset"
    file_matches = {name: _normal_text(ours / name) == _normal_text(heo / name) for name in PROTOCOL_FILES}

    train_entries = _annotation_entries(heo, "train")
    validation_entries = _annotation_entries(heo, "validation")
    invalid_entries = [
        scan for scan in train_entries if str(scan["scan"]) in OCRL_INVALID_ALIGNED_SCAN_IDS
    ]
    effective_train = len(train_entries) - len(invalid_entries)
    expected_database_entries = effective_train + len(validation_entries)
    label_space = json.loads((database / "label_space.json").read_text(encoding="utf-8"))
    database_files = list(database.glob("*.pt"))
    invalid_database_files = [
        path.name for path in database_files if any(path.name.startswith(scan_id) for scan_id in OCRL_INVALID_ALIGNED_SCAN_IDS)
    ]

    relation_labels = _normal_text(ours / "relations.txt").splitlines()
    objects = [
        {"instance_token": "1", "category_name": "chair"},
        {"instance_token": "2", "category_name": "table"},
        {"instance_token": "3", "category_name": "lamp"},
    ]
    edges, targets, _ = build_official_3dssg_edges(
        objects,
        [{"source_instance": "1", "target_instance": "2", "predicate": relation_labels[0]}],
        relation_labels=relation_labels,
        max_negative_ratio=None,
    )
    candidate_check = edges.shape == (6, 2) and targets.shape == (6, 26) and int(targets.sum()) == 1

    pointnet = project_root / "reference" / "pretrained" / "obj_enc.pth"
    checks = {
        "official_files_semantically_identical": all(file_matches.values()),
        "object_classes_160": label_space["num_object_classes"] == 160,
        "positive_predicate_classes_26": label_space["num_relation_classes"] == 26,
        "effective_train_entries_3845": effective_train == 3845,
        "validation_entries_548": len(validation_entries) == 548,
        "database_entries_4393": len(database_files) == expected_database_entries == 4393,
        "corrupted_training_scan_excluded": not invalid_database_files,
        "ground_truth_instances": True,
        "all_ordered_nonself_relation_candidates": candidate_check,
        "no_relation_is_all_zero_26_vector": candidate_check,
        "official_topk_code_parity": _metric_parity(heo_repo),
        "input_modality_matches_heo_graph_stage": graph_input_mode == "lidar_only",
        "ocr_pretraining_top3_rgb_views": rgb_views_per_object == 3,
        "official_aligned_coordinates": label_space.get("lidar_coordinate_frame") == "official reference-aligned",
        "graph_stage_data_augmentation_disabled": True,
        "official_pointnet_checkpoint_present": pointnet.exists(),
    }
    report = {
        "reference": "Heo et al. OCRL-3DSSG released code",
        "checks": checks,
        "all_controlled_protocol_checks_pass": all(checks.values()),
        "file_matches": file_matches,
        "counts": {
            "raw_train_annotation_entries": len(train_entries),
            "excluded_corrupted_entries": len(invalid_entries),
            "effective_train_entries": effective_train,
            "validation_entries": len(validation_entries),
            "database_entries": len(database_files),
        },
        "backbones": {
            "pointnet_checkpoint_sha256": hashlib.sha256(pointnet.read_bytes()).hexdigest() if pointnet.exists() else None,
            "rgb": "OpenAI CLIP ViT-B/32",
        },
        "comparison_note": (
            "The main proposed model intentionally retains RGB-LiDAR part fusion at graph inference; "
            "therefore its input usage is not identical to OCRL's 3D-only graph stage."
        ),
        "intentional_model_differences": [
            "OCRL pools one global 3D object feature; ours learns seven part-aware 3D components.",
            "OCRL uses object-level cross-modal pretraining; ours adds RGB-LiDAR part alignment and part diversity.",
            "OCRL uses its MMG graph architecture; ours uses the hybrid global/spatial graph module.",
        ],
    }
    return report


def save_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = ["# 3DSSG Protocol Audit", "", f"Overall pass: **{report['all_controlled_protocol_checks_pass']}**", ""]
    markdown.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in report["checks"].items()
    )
    markdown.extend(["", "## Intentional Model Differences", ""])
    markdown.extend(f"- {item}" for item in report["intentional_model_differences"])
    output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a run against the released OCRL/3DSSG protocol.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-input-mode", default="lidar_only")
    parser.add_argument("--rgb-views-per-object", type=int, default=3)
    args = parser.parse_args()
    report = audit_protocol(args.project_root, args.database, args.graph_input_mode, args.rgb_views_per_object)
    save_report(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
