from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PipelineConfig, ProjectPaths, ensure_output_dirs
from .data import TemporalSceneDataset, build_label_maps, compute_predicate_weights, load_label_space
from .graph import build_official_3dssg_edges, edge_geometric_features, node_labels_for_objects
from .losses import (
    DynamicEdgeConsistencyLoss,
    GraphPredictionLoss,
    RepresentationLoss,
    TemporalConsistencyLoss,
    TemporalPartAlignmentLoss,
)
from .models import DynamicSceneGraphModel
from .splits import get_3rscan_splits
from .temporal import process_scene, process_static_scene

def graph_losses_for_3dssg_scene(
    scene: dict[str, Any],
    results: dict[str, Any],
    model: DynamicSceneGraphModel,
    label_to_id: dict[str, int],
    graph_loss_fn: GraphPredictionLoss,
    dynamic_loss_fn: DynamicEdgeConsistencyLoss,
    device: torch.device,
    relation_labels: list[str],
    max_negative_ratio: int | None = None,
) -> dict[str, torch.Tensor]:
    
    zero = results["temporal_loss"].new_zeros(())
    
    tracklets = results["tracklets"]
    instance_tokens = results["instance_tokens"]
    temporal_embeddings = results["temporal_embeddings"]
    mask = results["mask"]

    if temporal_embeddings is None or not instance_tokens:
        return {"node_loss": zero, "edge_loss": zero, "lse_loss": zero, "dynamic_loss": zero}

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
        selected = {"instance_token": instance_id, "category_name": "unknown"}
        for frame in scene["frames"]:
            for obj in frame["objects"]:
                if obj["instance_token"] == instance_id:
                    selected = dict(obj)
                    break
            if selected["category_name"] != "unknown":
                break
        unique_objects.append(selected)

    edge_index, edge_labels, _ = build_official_3dssg_edges(
        unique_objects,
        scene.get("relationships", []),
        relation_labels=relation_labels,
        max_negative_ratio=max_negative_ratio,
    )
    edge_index = edge_index.to(device)
    edge_labels = edge_labels.to(device)
    node_labels = node_labels_for_objects(unique_objects, label_to_id, device)
    edge_geom = edge_geometric_features(unique_objects, edge_index, device)

    node_logits, edge_logits, geom_reconstruction = model.predict_graph(
        pooled_node_features,
        edge_index,
        edge_geom,
    )
    
    losses = graph_loss_fn(
        node_logits,
        node_labels,
        edge_logits,
        edge_labels,
        geom_reconstruction=geom_reconstruction,
        geom_targets=edge_geom,
    )

    edge_probs_by_frame = []
    for t, frame_encoded in enumerate(results["encoded_scene"]):
        frame_nodes = frame_encoded["embeddings"]
        frame_objects = frame_encoded["objects"]
        
        frame_instance_to_idx = {obj["instance_token"]: idx for idx, obj in enumerate(frame_objects)}
        
        frame_edge_list = []
        frame_edge_keys = []
        for src, tgt in edge_index.cpu().numpy():
            src_token = instance_tokens[src]
            tgt_token = instance_tokens[tgt]
            if src_token in frame_instance_to_idx and tgt_token in frame_instance_to_idx:
                frame_edge_list.append([frame_instance_to_idx[src_token], frame_instance_to_idx[tgt_token]])
                frame_edge_keys.append((src_token, tgt_token))
                
        if not frame_edge_list:
            continue
            
        frame_edge_idx = torch.tensor(frame_edge_list, dtype=torch.long, device=device)
        frame_geom = edge_geometric_features(frame_objects, frame_edge_idx, device)
        frame_edge_logits = model.predict_edges(frame_nodes, frame_edge_idx, frame_geom)
        
        if frame_edge_logits.numel():
            frame_probs = torch.sigmoid(frame_edge_logits)
            frame_dict = {key: frame_probs[idx] for idx, key in enumerate(frame_edge_keys)}
            edge_probs_by_frame.append(frame_dict)
            
    dynamic_loss = dynamic_loss_fn(edge_probs_by_frame) if edge_probs_by_frame else zero
    
    return {
        "node_loss": losses["node_loss"],
        "edge_loss": losses["edge_loss"],
        "lse_loss": losses["lse_loss"],
        "dynamic_loss": dynamic_loss,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args: argparse.Namespace) -> Path:
    _set_seed(getattr(args, "seed", 42))
    paths = ProjectPaths(reference_root=args.reference_root, output_root=args.output_root)
    ensure_output_dirs(paths)

    num_parts = args.num_parts
    lambda_object = 1.0
    lambda_temporal = args.lambda_temporal
    lambda_temporal_part = args.lambda_temporal_part
    if args.mode == "static":
        lambda_temporal = 0.0
        lambda_temporal_part = 0.0

    if args.ablation == "holistic":
        num_parts = 1
    elif args.ablation == "no-object-align":
        lambda_object = 0.0
    elif args.ablation == "no-temporal":
        lambda_temporal = 0.0
        lambda_temporal_part = 0.0

    config = PipelineConfig(
        learning_rate=args.learning_rate,
        lambda_temporal=lambda_temporal,
        lambda_temporal_part=lambda_temporal_part,
        lambda_node=args.lambda_node,
        lambda_edge=args.lambda_edge,
        lambda_dynamic=args.lambda_dynamic,
        lambda_object=lambda_object,
        num_parts=num_parts,
    )

    database_dir = args.database if args.database is not None else (paths.output_root / "3rscan_database")
    label_to_id, id_to_label = build_label_maps(database_dir)
    label_space = load_label_space(database_dir)
    if label_space is None:
        raise ValueError(
            "This database has no official label-space manifest. Rebuild it with "
            "build_3rscan_static_database.py before starting a comparable run."
        )
    object_labels, relation_labels, label_protocol = label_space
    if len(object_labels) != 160 or len(relation_labels) != 26:
        raise ValueError(
            f"Expected the official 160/26 OCRL label space, found "
            f"{len(object_labels)} objects and {len(relation_labels)} relations."
        )
    print(f"Label protocol: {label_protocol}")

    dataset = TemporalSceneDataset(database_dir, max_frames=args.max_frames)
    
    train_tokens, val_tokens = get_3rscan_splits(
        dataset.scene_tokens,
        train_scans=getattr(args, "train_scans", None),
        val_scans=getattr(args, "val_scans", None),
        seed=getattr(args, "split_seed", 42),
    )
    dataset.scene_tokens = train_tokens
    
    if args.max_scenes is not None:
        dataset.scene_tokens = dataset.scene_tokens[: args.max_scenes]
    
    split_name = "official 3DSSG" if getattr(args, "train_scans", None) and getattr(args, "val_scans", None) else "deterministic 80/20"
    print(f"Using {split_name} split.")
    print(f"Training on {len(dataset.scene_tokens)} scenes (Validation reserved: {len(val_tokens)})")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=args.shuffle, collate_fn=lambda batch: batch[0])

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DynamicSceneGraphModel(
        num_node_classes=len(label_to_id),
        num_edge_classes=len(relation_labels),
        dim=config.dim,
        num_parts=config.num_parts,
        num_heads=config.num_heads,
        fusion_layers=config.fusion_layers,
        temporal_layers=config.temporal_layers,
        dropout=config.dropout,
        graph_use_text=config.graph_use_text,
    ).to(device)

    representation_loss = RepresentationLoss(
        lambda_part=config.lambda_part,
        lambda_object=config.lambda_object,
        temperature=config.temperature,
    ).to(device)
    temporal_loss = TemporalConsistencyLoss(temperature=config.temperature).to(device)
    temporal_part_loss = TemporalPartAlignmentLoss(temperature=config.temperature).to(device)
    
    print("Computing Alpha class weights to balance rare 3DSSG relationships...")
    alpha_weights = compute_predicate_weights(database_dir, relation_labels, scene_tokens=dataset.scene_tokens).to(device)
    graph_loss = GraphPredictionLoss(alpha=alpha_weights).to(device)
    
    dynamic_loss = DynamicEdgeConsistencyLoss().to(device)

    history = []
    resume_optimizer_state = None
    start_epoch = 0
    if getattr(args, "resume_checkpoint", None) is not None:
        print(f"Warm-starting from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        history = checkpoint.get("history", [])
        resume_optimizer_state = checkpoint.get("optimizer")
        start_epoch = int(history[-1].get("epoch", len(history))) if history else 0
        if start_epoch >= args.epochs:
            print(f"Checkpoint already contains {start_epoch} epochs; target is {args.epochs}. Nothing to resume.")
            return args.resume_checkpoint
        print(f"Resuming at epoch {start_epoch + 1} of {args.epochs}.")
        if missing:
            print(f"Initialized new parameters: {len(missing)} tensors")
        if unexpected:
            print(f"Ignored checkpoint parameters: {len(unexpected)} tensors")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        print("Restored optimizer state from checkpoint.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = {
            "total": 0.0,
            "representation": 0.0,
            "temporal": 0.0,
            "temporal_part": 0.0,
            "node": 0.0,
            "edge": 0.0,
            "lse": 0.0,
            "dynamic": 0.0,
        }
        used = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for scene in progress:
            optimizer.zero_grad(set_to_none=True)
            if args.mode == "static":
                results = process_static_scene(
                    scene=scene,
                    scene_model=model.object_encoder,
                    representation_loss=representation_loss,
                    device=device,
                )
            else:
                results = process_scene(
                    scene=scene,
                    scene_model=model.object_encoder,
                    association_model=model.association,
                    aggregation_model=model.aggregation,
                    temporal_model=model.temporal,
                    temporal_loss=temporal_loss,
                    representation_loss=representation_loss,
                    device=device,
                )
            if results is None:
                continue

            temp_part = temporal_part_loss(results["encoded_scene"])
            
            g = graph_losses_for_3dssg_scene(
                scene=scene,
                results=results,
                model=model,
                label_to_id=label_to_id,
                graph_loss_fn=graph_loss,
                dynamic_loss_fn=dynamic_loss,
                device=device,
                relation_labels=relation_labels,
                max_negative_ratio=args.negative_ratio,
            )
            
            total_loss = (
                results["representation_loss"]
                + config.lambda_temporal * results["temporal_loss"]
                + config.lambda_temporal_part * temp_part
                + config.lambda_node * g["node_loss"]
                + config.lambda_edge * (g["edge_loss"] + 0.1 * g["lse_loss"])
                + config.lambda_dynamic * g["dynamic_loss"]
            )
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            used += 1
            totals["total"] += float(total_loss.detach().cpu())
            totals["representation"] += float(results["representation_loss"].detach().cpu())
            totals["temporal"] += float(results["temporal_loss"].detach().cpu())
            totals["temporal_part"] += float(temp_part.detach().cpu())
            totals["node"] += float(g["node_loss"].detach().cpu())
            totals["edge"] += float(g["edge_loss"].detach().cpu())
            totals["lse"] += float(g["lse_loss"].detach().cpu())
            totals["dynamic"] += float(g["dynamic_loss"].detach().cpu())
            progress.set_postfix({key: round(value / used, 4) for key, value in totals.items()})

        epoch_metrics = {key: value / max(used, 1) for key, value in totals.items()}
        epoch_metrics["epoch"] = epoch + 1
        history.append(epoch_metrics)

        checkpoint_path = paths.checkpoints / f"integration_by_parts_3rscan_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.__dict__,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "relation_labels": relation_labels,
                "history": history,
                "seed": getattr(args, "seed", 42),
                "mode": args.mode,
                "database": str(database_dir),
            },
            checkpoint_path,
        )

    metrics_path = paths.output_root / "training_history_3rscan.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {metrics_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Integration by Parts pipeline on 3DSSG.")
    parser.add_argument("--reference-root", type=Path, default=ProjectPaths().reference_root)
    parser.add_argument("--output-root", type=Path, default=ProjectPaths().output_root)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--mode", choices=["static", "temporal"], default="temporal")
    parser.add_argument("--train-scans", type=Path, default=None, help="Official 3DSSG train_scans.txt.")
    parser.add_argument("--val-scans", type=Path, default=None, help="Official 3DSSG validation_scans.txt.")
    parser.add_argument("--split-seed", type=int, default=42, help="Fallback seed when official split files are not provided.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed for training.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--negative-ratio", type=int, default=None, help="Optional negative-pair cap; omit for the official all-pairs protocol.")
    parser.add_argument("--lambda-temporal", type=float, default=1.0)
    parser.add_argument("--lambda-temporal-part", type=float, default=0.5)
    parser.add_argument("--lambda-node", type=float, default=1.0)
    parser.add_argument("--lambda-edge", type=float, default=1.0)
    parser.add_argument("--lambda-dynamic", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--num-parts", type=int, default=7)
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "holistic", "no-object-align", "no-temporal"],
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

