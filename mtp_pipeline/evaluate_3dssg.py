from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PipelineConfig, ProjectPaths
from .data import TemporalSceneDataset, build_label_maps
from .graph import RELATION_LABELS, build_3dssg_edges, edge_geometric_features, node_labels_for_objects
from .models import DynamicSceneGraphModel
from .temporal import process_scene


def calculate_recall(ground_truth_triplets: set, predicted_triplets: list, k: int) -> float:
    """Calculate Recall@K for a single scene."""
    if not ground_truth_triplets:
        return 1.0
    
    top_k_preds = set(predicted_triplets[:k])
    hits = len(ground_truth_triplets.intersection(top_k_preds))
    return hits / len(ground_truth_triplets)


@torch.no_grad()
def evaluate_model(args: argparse.Namespace) -> None:
    paths = ProjectPaths(reference_root=args.reference_root, output_root=args.output_root)
    database_dir = args.database if args.database is not None else (paths.output_root / "3rscan_database")
    
    print(f"Loading database from {database_dir}...")
    label_to_id, _ = build_label_maps(database_dir)

    dataset = TemporalSceneDataset(database_dir)
    
    # 80/20 Train/Val Split (Use only the 20% validation set for evaluation)
    from .train import get_3rscan_splits
    _, val_tokens = get_3rscan_splits(dataset.scene_tokens)
    if not val_tokens:
        val_tokens = dataset.scene_tokens
    dataset.scene_tokens = val_tokens
    print(f"Evaluating on {len(dataset.scene_tokens)} validation scenes...")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    config = PipelineConfig()
    model = DynamicSceneGraphModel(
        num_node_classes=len(label_to_id),
        num_edge_classes=len(RELATION_LABELS),
        dim=config.dim,
        num_parts=config.num_parts,
        num_heads=config.num_heads,
        fusion_layers=config.fusion_layers,
        temporal_layers=config.temporal_layers,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        print(f"Checkpoint/model mismatch: {len(missing)} missing tensors, {len(unexpected)} unexpected tensors")
        print("Use a checkpoint trained after the geometry-aware edge upgrade for publishable metrics.")
    model.eval()

    # Track metrics
    sgcls_recalls = {"20": [], "50": [], "100": []}
    predcls_recalls = {"20": [], "50": [], "100": []}
    
    # Track per-predicate recall for mR@K
    predicate_hits_sgcls = {k: defaultdict(int) for k in [20, 50, 100]}
    predicate_totals = defaultdict(int)
    predicate_hits_predcls = {k: defaultdict(int) for k in [20, 50, 100]}

    for scene in tqdm(loader, desc="Evaluating 3DSSG metrics"):
        results = process_scene(
            scene=scene,
            scene_model=model.object_encoder,
            association_model=model.association,
            aggregation_model=model.aggregation,
            temporal_model=model.temporal,
            temporal_loss=lambda x, y: torch.tensor(0.0, device=device),
            device=device,
        )
        
        if results is None or results["temporal_embeddings"] is None:
            continue
            
        instance_tokens = results["instance_tokens"]
        temporal_embeddings = results["temporal_embeddings"]
        mask = results["mask"]

        # Pool temporal embeddings
        pooled_nodes = []
        for i in range(len(instance_tokens)):
            valid_frames = mask[i]
            if not valid_frames.any():
                pooled = temporal_embeddings[i, 0]
            else:
                pooled = temporal_embeddings[i, valid_frames].mean(dim=0)
            pooled_nodes.append(pooled)
        pooled_node_features = torch.stack(pooled_nodes)

        unique_objects = []
        for instance_id in instance_tokens:
            category = "unknown"
            for frame in scene["frames"]:
                for obj in frame["objects"]:
                    if obj["instance_token"] == instance_id:
                        category = obj["category_name"]
                        break
                if category != "unknown":
                    break
            unique_objects.append({"instance_token": instance_id, "category_name": category})

        # GT Edges
        edge_index, edge_labels, metadata = build_3dssg_edges(unique_objects, scene.get("relationships", []))
        if edge_index.numel() == 0:
            continue
            
        edge_index = edge_index.to(device)
        edge_geom = edge_geometric_features(unique_objects, edge_index, device)
        gt_node_labels = node_labels_for_objects(unique_objects, label_to_id, device)
        
        # Predictions
        node_logits = model.predict_nodes(pooled_node_features)
        edge_logits = model.predict_edges(pooled_node_features, edge_index, edge_geom)
        
        node_probs = F.softmax(node_logits, dim=-1)
        edge_probs = F.softmax(edge_logits, dim=-1)
        
        # --- SGCls (Predict Object Labels AND Predicate Labels) ---
        gt_triplets = set()
        for idx in range(len(metadata)):
            src_idx = edge_index[idx, 0].item()
            tgt_idx = edge_index[idx, 1].item()
            src_label = gt_node_labels[src_idx].item()
            tgt_label = gt_node_labels[tgt_idx].item()
            rel_label = edge_labels[idx].item()
            gt_triplets.add((src_idx, tgt_idx, src_label, tgt_label, rel_label))
            predicate_totals[rel_label] += 1
            
        # Score = P(src_node) * P(tgt_node) * P(edge)
        sgcls_candidates = []
        for idx in range(len(metadata)):
            src_idx = edge_index[idx, 0].item()
            tgt_idx = edge_index[idx, 1].item()
            
            # For each edge, consider all possible predicate labels (ignoring 'none' which is usually index 0)
            for rel_cls in range(1, len(RELATION_LABELS)):
                pred_src_label = node_probs[src_idx].argmax().item()
                pred_tgt_label = node_probs[tgt_idx].argmax().item()
                
                score = (node_probs[src_idx, pred_src_label] * 
                         node_probs[tgt_idx, pred_tgt_label] * 
                         edge_probs[idx, rel_cls]).item()
                         
                sgcls_candidates.append({
                    "triplet": (src_idx, tgt_idx, pred_src_label, pred_tgt_label, rel_cls),
                    "score": score
                })
                
        sgcls_candidates.sort(key=lambda x: x["score"], reverse=True)
        predicted_sgcls_triplets = [x["triplet"] for x in sgcls_candidates]
        
        for k in [20, 50, 100]:
            sgcls_recalls[str(k)].append(calculate_recall(gt_triplets, predicted_sgcls_triplets, k))
            top_k = set(predicted_sgcls_triplets[:k])
            for gt in gt_triplets:
                if gt in top_k:
                    predicate_hits_sgcls[k][gt[4]] += 1
                    
        # --- PredCls (Given Object Labels, Predict Predicate Labels) ---
        predcls_candidates = []
        for idx in range(len(metadata)):
            src_idx = edge_index[idx, 0].item()
            tgt_idx = edge_index[idx, 1].item()
            
            # We are GIVEN the ground truth node labels
            src_label = gt_node_labels[src_idx].item()
            tgt_label = gt_node_labels[tgt_idx].item()
            
            for rel_cls in range(1, len(RELATION_LABELS)):
                score = edge_probs[idx, rel_cls].item()
                predcls_candidates.append({
                    "triplet": (src_idx, tgt_idx, src_label, tgt_label, rel_cls),
                    "score": score
                })
                
        predcls_candidates.sort(key=lambda x: x["score"], reverse=True)
        predicted_predcls_triplets = [x["triplet"] for x in predcls_candidates]
        
        for k in [20, 50, 100]:
            predcls_recalls[str(k)].append(calculate_recall(gt_triplets, predicted_predcls_triplets, k))
            top_k = set(predicted_predcls_triplets[:k])
            for gt in gt_triplets:
                if gt in top_k:
                    predicate_hits_predcls[k][gt[4]] += 1

    # Aggregate Results
    results_summary = {
        "SGCls": {
            "R@20": sum(sgcls_recalls["20"]) / max(len(sgcls_recalls["20"]), 1),
            "R@50": sum(sgcls_recalls["50"]) / max(len(sgcls_recalls["50"]), 1),
            "R@100": sum(sgcls_recalls["100"]) / max(len(sgcls_recalls["100"]), 1),
        },
        "PredCls": {
            "R@20": sum(predcls_recalls["20"]) / max(len(predcls_recalls["20"]), 1),
            "R@50": sum(predcls_recalls["50"]) / max(len(predcls_recalls["50"]), 1),
            "R@100": sum(predcls_recalls["100"]) / max(len(predcls_recalls["100"]), 1),
        }
    }
    
    # Calculate mR@K
    for task_name, hits_dict in [("SGCls", predicate_hits_sgcls), ("PredCls", predicate_hits_predcls)]:
        for k in [20, 50, 100]:
            per_class_recalls = []
            for pred_id, total in predicate_totals.items():
                if total > 0:
                    per_class_recalls.append(hits_dict[k][pred_id] / total)
            
            mR_k = sum(per_class_recalls) / len(per_class_recalls) if per_class_recalls else 0.0
            results_summary[task_name][f"mR@{k}"] = mR_k

    print("\n--- 3DSSG Evaluation Results ---")
    print(json.dumps(results_summary, indent=4))
    
    out_path = args.checkpoint.parent / f"evaluation_{args.checkpoint.stem}.json"
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"Saved evaluation results to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Integration by Parts on 3DSSG Metrics (SGCls, PredCls).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--reference-root", type=Path, default=ProjectPaths().reference_root)
    parser.add_argument("--output-root", type=Path, default=ProjectPaths().output_root)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_model(parse_args())

