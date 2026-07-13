from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PipelineConfig, ProjectPaths, ensure_output_dirs
from .data import TemporalSceneDataset, build_label_maps, compute_predicate_weights
from .graph import RELATION_LABELS, build_3dssg_edges, edge_geometric_features, node_labels_for_objects
from .losses import (
    DynamicEdgeConsistencyLoss,
    GraphPredictionLoss,
    RepresentationLoss,
    TemporalConsistencyLoss,
    TemporalPartAlignmentLoss,
)
from .models import DynamicSceneGraphModel
from .temporal import process_scene

def get_3rscan_splits(scene_tokens: list[str], seed: int = 42) -> tuple[list[str], list[str]]:
    """Deterministically split scenes into 80% train, 20% validation."""
    tokens = sorted(scene_tokens)
    if len(tokens) < 2:
        return tokens, []
    rng = random.Random(seed)
    rng.shuffle(tokens)
    split_idx = max(1, int(len(tokens) * 0.8))
    return tokens[:split_idx], tokens[split_idx:]

def graph_losses_for_3dssg_scene(
    scene: dict[str, Any],
    results: dict[str, Any],
    model: DynamicSceneGraphModel,
    label_to_id: dict[str, int],
    graph_loss_fn: GraphPredictionLoss,
    dynamic_loss_fn: DynamicEdgeConsistencyLoss,
    device: torch.device,
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

    edge_index, edge_labels, _ = build_3dssg_edges(unique_objects, scene.get("relationships", []))
    edge_index = edge_index.to(device)
    edge_labels = edge_labels.to(device)
    node_labels = node_labels_for_objects(unique_objects, label_to_id, device)
    edge_geom = edge_geometric_features(unique_objects, edge_index, device)

    node_logits = model.predict_nodes(pooled_node_features)
    edge_logits, geom_reconstruction = model.predict_edges(pooled_node_features, edge_index, edge_geom, return_aux=True)
    
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
            frame_probs = F.softmax(frame_edge_logits, dim=-1)
            frame_dict = {key: frame_probs[idx] for idx, key in enumerate(frame_edge_keys)}
            edge_probs_by_frame.append(frame_dict)
            
    dynamic_loss = dynamic_loss_fn(edge_probs_by_frame) if edge_probs_by_frame else zero
    
    return {
        "node_loss": losses["node_loss"],
        "edge_loss": losses["edge_loss"],
        "lse_loss": losses["lse_loss"],
        "dynamic_loss": dynamic_loss,
    }


def train(args: argparse.Namespace) -> Path:
    paths = ProjectPaths(reference_root=args.reference_root, output_root=args.output_root)
    ensure_output_dirs(paths)

    num_parts = args.num_parts
    lambda_object = 1.0
    lambda_temporal = args.lambda_temporal
    lambda_temporal_part = args.lambda_temporal_part

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

    dataset = TemporalSceneDataset(database_dir, max_frames=args.max_frames)
    
    # 80/20 Train/Val Split
    train_tokens, val_tokens = get_3rscan_splits(dataset.scene_tokens)
    dataset.scene_tokens = train_tokens
    
    if args.max_scenes is not None:
        dataset.scene_tokens = dataset.scene_tokens[: args.max_scenes]
    
    print(f"Training on {len(dataset.scene_tokens)} scenes (Validation reserved: {len(val_tokens)})")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=args.shuffle, collate_fn=lambda batch: batch[0])

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DynamicSceneGraphModel(
        num_node_classes=len(label_to_id),
        num_edge_classes=len(RELATION_LABELS),
        dim=config.dim,
        num_parts=config.num_parts,
        num_heads=config.num_heads,
        fusion_layers=config.fusion_layers,
        temporal_layers=config.temporal_layers,
        dropout=config.dropout,
    ).to(device)

    representation_loss = RepresentationLoss(
        lambda_part=config.lambda_part,
        lambda_object=config.lambda_object,
        temperature=config.temperature,
    ).to(device)
    temporal_loss = TemporalConsistencyLoss(temperature=config.temperature).to(device)
    temporal_part_loss = TemporalPartAlignmentLoss(temperature=config.temperature).to(device)
    
    print("Computing Alpha class weights to balance rare 3DSSG relationships...")
    alpha_weights = compute_predicate_weights(database_dir, RELATION_LABELS).to(device)
    graph_loss = GraphPredictionLoss(alpha=alpha_weights).to(device)
    
    dynamic_loss = DynamicEdgeConsistencyLoss().to(device)

    history = []
    if getattr(args, "resume_checkpoint", None) is not None:
        print(f"Warm-starting from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        history = checkpoint.get("history", [])
        if missing:
            print(f"Initialized new parameters: {len(missing)} tensors")
        if unexpected:
            print(f"Ignored checkpoint parameters: {len(unexpected)} tensors")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    for epoch in range(args.epochs):
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
                "config": config.__dict__,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "relation_labels": RELATION_LABELS,
                "history": history,
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
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

