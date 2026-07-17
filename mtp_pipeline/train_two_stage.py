from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PipelineConfig, ProjectPaths, ensure_output_dirs
from .data import (
    TemporalSceneDataset,
    build_label_maps,
    compute_object_weights,
    compute_predicate_weights,
    load_label_space,
)
from .evaluate_3dssg import (
    _class_mean_recall,
    _object_ranks_for_scene,
    _pool_node_features,
    _predicate_ranks_for_scene,
    _recall_from_ranks,
    _unique_objects,
)
from .graph import build_official_3dssg_edges, edge_geometric_features, node_labels_for_objects
from .losses import DynamicEdgeConsistencyLoss, GraphPredictionLoss, RepresentationLoss
from .models import DynamicSceneGraphModel
from .splits import get_3rscan_splits
from .temporal import process_static_scene
from .train import _set_seed, graph_losses_for_3dssg_scene


def _setup(args: Any):
    paths = ProjectPaths(reference_root=args.reference_root, output_root=args.output_root)
    ensure_output_dirs(paths)
    database_dir = Path(args.database)
    label_space = load_label_space(database_dir)
    if label_space is None:
        raise ValueError("The two-stage run requires an official database with label_space.json.")
    object_labels, relation_labels, protocol = label_space
    if len(object_labels) != 160 or len(relation_labels) != 26:
        raise ValueError(f"Expected the official 160/26 label space, found {len(object_labels)}/{len(relation_labels)}.")
    label_to_id, id_to_label = build_label_maps(database_dir)

    all_data = TemporalSceneDataset(database_dir, max_frames=getattr(args, "max_frames", None))
    train_tokens, validation_tokens = get_3rscan_splits(
        all_data.scene_tokens,
        train_scans=args.train_scans,
        val_scans=args.val_scans,
        seed=getattr(args, "split_seed", 42),
        exclude_ocr_invalid_scan=getattr(args, "exclude_ocr_invalid_scan", False),
    )
    if getattr(args, "max_scenes", None) is not None:
        train_tokens = train_tokens[: args.max_scenes]
    if getattr(args, "max_validation_scenes", None) is not None:
        validation_tokens = validation_tokens[: args.max_validation_scenes]

    train_data = TemporalSceneDataset(database_dir, max_frames=getattr(args, "max_frames", None))
    validation_data = TemporalSceneDataset(database_dir, max_frames=getattr(args, "max_frames", None))
    train_data.scene_tokens = train_tokens
    validation_data.scene_tokens = validation_tokens
    print(f"Protocol: {protocol}")
    print(f"Training entries: {len(train_tokens)} | Validation entries: {len(validation_tokens)}")
    return (
        paths,
        database_dir,
        object_labels,
        relation_labels,
        label_to_id,
        id_to_label,
        train_data,
        validation_data,
    )


def _config(args: Any) -> PipelineConfig:
    return PipelineConfig(
        dim=getattr(args, "embedding_dim", 384),
        num_parts=getattr(args, "num_parts", 7),
        graph_input_mode=getattr(args, "graph_input_mode", "multimodal"),
        use_rgb_token_mask=getattr(args, "use_rgb_token_mask", True),
        extended_geometry=getattr(args, "extended_geometry", False),
        graph_context=getattr(args, "graph_context", "hybrid"),
        graph_knn_neighbors=getattr(args, "graph_knn_neighbors", 8),
        conditioned_part_queries=getattr(args, "conditioned_part_queries", True),
        hybrid_spatial_init=getattr(args, "hybrid_spatial_init", 0.15),
        learning_rate=args.learning_rate,
        lambda_diversity=getattr(args, "lambda_diversity", 0.05),
        lambda_part_alignment=getattr(args, "lambda_part_alignment", 0.2),
        lambda_object=getattr(args, "lambda_object", 0.5),
        lambda_temporal=0.0,
        lambda_temporal_part=0.0,
        lambda_node=getattr(args, "lambda_node", 1.0),
        lambda_edge=getattr(args, "lambda_edge", 1.0),
        lambda_dynamic=0.0,
    )


def _build_model(
    config: PipelineConfig,
    num_object_classes: int,
    num_relation_classes: int,
) -> DynamicSceneGraphModel:
    return DynamicSceneGraphModel(
        num_node_classes=num_object_classes,
        num_edge_classes=num_relation_classes,
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
    )


def _representation_loss(config: PipelineConfig) -> RepresentationLoss:
    return RepresentationLoss(
        lambda_part=config.lambda_part,
        lambda_object=config.lambda_object,
        lambda_diversity=config.lambda_diversity,
        lambda_part_alignment=config.lambda_part_alignment,
        temperature=config.temperature,
    )


def _load_or_compute_object_weights(
    args: Any,
    database_dir: Path,
    object_labels: list[str],
    train_tokens: list[str],
) -> torch.Tensor:
    cache = Path(getattr(args, "class_weight_cache", Path(args.output_root) / "train_object_weights.pt"))
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("labels") == object_labels and payload.get("train_tokens") == train_tokens:
            return payload["weights"].float()
    weights = compute_object_weights(database_dir, object_labels, train_tokens, exponent=0.25)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"labels": object_labels, "train_tokens": train_tokens, "weights": weights}, cache)
    return weights


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    model: DynamicSceneGraphModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: PipelineConfig,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    relation_labels: list[str],
    history: list[dict[str, float]],
    stage: str,
    database_dir: Path,
    best_score: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config.__dict__,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "relation_labels": relation_labels,
        "history": history,
        "stage": stage,
        "database": str(database_dir),
        "best_selection_score": float(best_score),
    }
    if extra:
        payload.update(extra)
    return payload


@torch.inference_mode()
def _validate(
    model: DynamicSceneGraphModel,
    dataset: TemporalSceneDataset,
    label_to_id: dict[str, int],
    relation_labels: list[str],
    config: PipelineConfig,
    device: torch.device,
    graph_stage: bool,
) -> dict[str, float]:
    model.eval()
    object_ranks: list[int] = []
    object_by_class: dict[int, list[int]] = defaultdict(list)
    predicate_ranks: list[int] = []
    predicate_by_class: dict[int, list[int]] = defaultdict(list)
    true_positive = false_positive = false_negative = true_negative = 0
    evaluated = 0

    for scene in tqdm(dataset, desc="Official validation proxy", leave=False):
        results = process_static_scene(scene, model.object_encoder, representation_loss=None, device=device)
        if results is None:
            continue
        objects = _unique_objects(scene, results["instance_tokens"])
        if len(objects) != len(results["instance_tokens"]):
            continue
        pooled_nodes = _pool_node_features(results)
        node_targets = node_labels_for_objects(objects, label_to_id, device)

        if graph_stage:
            if len(objects) < 2:
                continue
            edge_index, relation_targets, _ = build_official_3dssg_edges(
                objects,
                scene.get("relationships", []),
                relation_labels=relation_labels,
                max_negative_ratio=None,
            )
            if edge_index.numel() == 0:
                continue
            edge_index = edge_index.to(device)
            relation_targets = relation_targets.to(device)
            geometry = edge_geometric_features(
                objects,
                edge_index,
                device,
                extended_geometry=config.extended_geometry,
            )
            node_logits, edge_logits, _ = model.predict_graph(pooled_nodes, edge_index, geometry)
            relation_probs = torch.sigmoid(edge_logits).cpu()
            relation_targets_cpu = relation_targets.cpu()
            scene_predicate_ranks, scene_predicate_by_class = _predicate_ranks_for_scene(
                relation_probs,
                relation_targets_cpu,
            )
            predicate_ranks.extend(scene_predicate_ranks)
            for key, values in scene_predicate_by_class.items():
                predicate_by_class[key].extend(values)
            binary_prediction = relation_probs >= 0.5
            binary_target = relation_targets_cpu > 0.5
            true_positive += int((binary_prediction & binary_target).sum())
            false_positive += int((binary_prediction & ~binary_target).sum())
            false_negative += int((~binary_prediction & binary_target).sum())
            true_negative += int((~binary_prediction & ~binary_target).sum())
        else:
            node_logits = model.node_head(pooled_nodes)

        node_probs = torch.softmax(node_logits, dim=-1).cpu()
        scene_object_ranks, scene_object_by_class = _object_ranks_for_scene(node_probs, node_targets.cpu())
        object_ranks.extend(scene_object_ranks)
        for key, values in scene_object_by_class.items():
            object_by_class[key].extend(values)
        evaluated += 1

    metrics = {
        "evaluated_entries": float(evaluated),
        "object_r1": _recall_from_ranks(object_ranks, 1),
        "object_r5": _recall_from_ranks(object_ranks, 5),
        "object_mr1": _class_mean_recall(object_by_class, 1),
        "object_mr5": _class_mean_recall(object_by_class, 5),
    }
    if graph_stage:
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        accuracy = (true_positive + true_negative) / max(
            true_positive + true_negative + false_positive + false_negative,
            1,
        )
        metrics.update(
            {
                "predicate_r1": _recall_from_ranks(predicate_ranks, 1),
                "predicate_r3": _recall_from_ranks(predicate_ranks, 3),
                "predicate_mr1": _class_mean_recall(predicate_by_class, 1),
                "predicate_mr3": _class_mean_recall(predicate_by_class, 3),
                "predicate_precision_05": precision,
                "predicate_recall_05": recall,
                "predicate_f1_05": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "predicate_binary_accuracy_05": accuracy,
            }
        )
        metrics["selection_score"] = (
            0.30 * metrics["object_r1"]
            + 0.15 * metrics["object_mr1"]
            + 0.30 * metrics["predicate_r1"]
            + 0.15 * metrics["predicate_mr1"]
            + 0.10 * metrics["predicate_r3"]
        )
    else:
        metrics["selection_score"] = 0.8 * metrics["object_r1"] + 0.2 * metrics["object_mr1"]
    return metrics


def _print_validation(metrics: dict[str, float]) -> None:
    visible = {key: round(value * 100.0, 2) for key, value in metrics.items() if key != "evaluated_entries"}
    visible["evaluated_entries"] = int(metrics["evaluated_entries"])
    print("Validation (%):")
    print(json.dumps(visible, indent=2))


def _cosine_scheduler(optimizer: torch.optim.Optimizer, epochs: int, minimum_factor: float = 0.02):
    def factor(epoch: int) -> float:
        progress = min(max(epoch / max(epochs, 1), 0.0), 1.0)
        return minimum_factor + 0.5 * (1.0 - minimum_factor) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def pretrain_object_encoder(args: Any) -> Path:
    """Stage 1: learn conditioned parts and context-free object classification."""
    _set_seed(getattr(args, "seed", 42))
    (
        paths,
        database_dir,
        object_labels,
        relation_labels,
        label_to_id,
        id_to_label,
        train_data,
        validation_data,
    ) = _setup(args)
    config = _config(args)
    device = torch.device(args.device if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(config, len(object_labels), len(relation_labels)).to(device)
    representation_loss = _representation_loss(config).to(device)
    object_weights = _load_or_compute_object_weights(
        args,
        database_dir,
        object_labels,
        train_data.scene_tokens,
    ).to(device)
    loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
    optimizer = torch.optim.AdamW(
        list(model.object_encoder.parameters()) + list(model.node_head.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _cosine_scheduler(optimizer, args.epochs)
    history: list[dict[str, float]] = []
    start_epoch = 0
    best_score = float("-inf")

    if getattr(args, "resume_checkpoint", None):
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("stage") != "object_pretraining":
            raise ValueError("The pretraining resume checkpoint belongs to a different stage.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint.get("history", [])
        start_epoch = int(history[-1]["epoch"]) if history else 0
        best_score = float(checkpoint.get("best_selection_score", float("-inf")))
        print(f"Resuming object pretraining at epoch {start_epoch + 1}/{args.epochs}.")

    latest_path = paths.checkpoints / "object_pretrain_latest.pt"
    best_path = paths.checkpoints / "object_pretrain_best.pt"
    validation_interval = max(int(getattr(args, "validation_interval", 3)), 1)
    grad_clip = float(getattr(args, "grad_clip", 1.0))
    node_scale = float(getattr(args, "lambda_pretrain_node", 1.0))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = {key: 0.0 for key in ("total", "node", "representation", "diversity", "part_alignment", "object")}
        used = 0
        progress = tqdm(loader, desc=f"Object pretrain {epoch + 1}/{args.epochs}")
        for scene in progress:
            optimizer.zero_grad(set_to_none=True)
            results = process_static_scene(scene, model.object_encoder, representation_loss=None, device=device)
            if results is None:
                continue
            objects = _unique_objects(scene, results["instance_tokens"])
            if len(objects) != len(results["instance_tokens"]):
                continue
            outputs = results["encoded_scene"][0]["outputs"]
            categories = [str(obj.get("category_name", "unknown")) for obj in objects]
            representation = representation_loss(outputs, categories)
            node_features = _pool_node_features(results)
            node_targets = node_labels_for_objects(objects, label_to_id, device)
            node_logits = model.node_head(node_features)
            node_loss = F.cross_entropy(
                node_logits,
                node_targets,
                weight=object_weights,
                label_smoothing=float(getattr(args, "node_label_smoothing", 0.05)),
            )
            total_loss = representation["representation_loss"] + node_scale * node_loss
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.object_encoder.parameters()) + list(model.node_head.parameters()),
                    grad_clip,
                )
            optimizer.step()

            used += 1
            values = {
                "total": total_loss,
                "node": node_loss,
                "representation": representation["representation_loss"],
                "diversity": representation["diversity_loss"],
                "part_alignment": representation["part_alignment_loss"],
                "object": representation["object_loss"],
            }
            for key, value in values.items():
                totals[key] += float(value.detach().cpu())
            progress.set_postfix(total=round(totals["total"] / used, 4), node=round(totals["node"] / used, 4))

        epoch_metrics = {key: value / max(used, 1) for key, value in totals.items()}
        epoch_metrics.update({"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"]})
        should_validate = (epoch + 1) % validation_interval == 0 or epoch + 1 == args.epochs
        if should_validate:
            validation = _validate(model, validation_data, label_to_id, relation_labels, config, device, graph_stage=False)
            epoch_metrics.update({f"val_{key}": value for key, value in validation.items()})
            _print_validation(validation)
        history.append(epoch_metrics)
        scheduler.step()

        current_score = float(epoch_metrics.get("val_selection_score", float("-inf")))
        best_score = max(best_score, current_score)
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            config,
            label_to_id,
            id_to_label,
            relation_labels,
            history,
            "object_pretraining",
            database_dir,
            best_score,
            extra={"seed": getattr(args, "seed", 42)},
        )
        _atomic_save(payload, latest_path)
        if should_validate and current_score >= best_score:
            _atomic_save(payload, best_path)
            print(f"Saved best object-pretraining checkpoint: {best_path}")
        (paths.output_root / "object_pretraining_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if not best_path.exists():
        _atomic_save(payload, best_path)
    print(f"Object pretraining complete. Best checkpoint: {best_path}")
    return best_path


def train_graph_from_pretrained(args: Any) -> Path:
    """Stage 2: initialize from stage 1 and train the hybrid scene graph."""
    _set_seed(getattr(args, "seed", 42))
    (
        paths,
        database_dir,
        object_labels,
        relation_labels,
        label_to_id,
        id_to_label,
        train_data,
        validation_data,
    ) = _setup(args)
    config = _config(args)
    device = torch.device(args.device if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(config, len(object_labels), len(relation_labels)).to(device)
    representation_loss = _representation_loss(config).to(device)
    object_weights = _load_or_compute_object_weights(
        args,
        database_dir,
        object_labels,
        train_data.scene_tokens,
    ).to(device)
    predicate_weights = compute_predicate_weights(
        database_dir,
        relation_labels,
        scene_tokens=train_data.scene_tokens,
    ).to(device)
    graph_loss = GraphPredictionLoss(
        alpha=predicate_weights,
        negative_weight=float(getattr(args, "edge_negative_weight", 1.0)),
        edge_loss_mode=getattr(args, "edge_loss_mode", "hybrid"),
        edge_balance_mix=float(getattr(args, "edge_balance_mix", 0.15)),
        node_weights=object_weights,
        node_balance_mix=float(getattr(args, "node_balance_mix", 0.15)),
        node_label_smoothing=float(getattr(args, "node_label_smoothing", 0.05)),
    ).to(device)
    dynamic_loss = DynamicEdgeConsistencyLoss().to(device)

    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("stage") != "graph_finetuning":
            raise ValueError("The graph resume checkpoint belongs to a different stage.")
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        pretrain_checkpoint = Path(args.pretrain_checkpoint)
        # The stage-1 optimizer state is not needed here; keep it off the GPU.
        checkpoint = torch.load(pretrain_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "object_pretraining":
            raise ValueError("Stage 2 requires an object-pretraining checkpoint.")
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing or unexpected:
            raise ValueError(f"Pretraining architecture mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"Initialized graph training from: {pretrain_checkpoint}")
        del checkpoint

    encoder_parameters = list(model.object_encoder.parameters())
    graph_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("object_encoder.")
        and not name.startswith("association.")
        and not name.startswith("temporal.")
    ]
    encoder_lr_scale = float(getattr(args, "encoder_lr_scale", 0.15))
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": config.learning_rate * encoder_lr_scale, "name": "object_encoder"},
            {"params": graph_parameters, "lr": config.learning_rate, "name": "graph"},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = _cosine_scheduler(optimizer, args.epochs)
    history: list[dict[str, float]] = []
    start_epoch = 0
    best_score = float("-inf")
    if resume_checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint.get("history", [])
        start_epoch = int(history[-1]["epoch"]) if history else 0
        best_score = float(checkpoint.get("best_selection_score", float("-inf")))
        print(f"Resuming graph training at epoch {start_epoch + 1}/{args.epochs}.")

    loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
    latest_path = paths.checkpoints / "graph_latest.pt"
    best_path = paths.checkpoints / "graph_best_official_proxy.pt"
    validation_interval = max(int(getattr(args, "validation_interval", 5)), 1)
    freeze_epochs = max(int(getattr(args, "encoder_freeze_epochs", 3)), 0)
    grad_clip = float(getattr(args, "grad_clip", 1.0))
    edge_warmup_epochs = max(int(getattr(args, "edge_warmup_epochs", 5)), 0)
    edge_warmup_start = float(getattr(args, "edge_warmup_start", 0.5))

    def edge_lambda(epoch_index: int) -> float:
        if edge_warmup_epochs <= 1 or epoch_index >= edge_warmup_epochs - 1:
            return config.lambda_edge
        progress = epoch_index / float(edge_warmup_epochs - 1)
        return config.lambda_edge * (edge_warmup_start + (1.0 - edge_warmup_start) * progress)

    for epoch in range(start_epoch, args.epochs):
        encoder_trainable = epoch >= freeze_epochs
        for parameter in encoder_parameters:
            parameter.requires_grad_(encoder_trainable)
        model.train()
        if not encoder_trainable:
            model.object_encoder.eval()
        current_edge_lambda = edge_lambda(epoch)
        totals = {
            key: 0.0
            for key in ("total", "representation", "diversity", "part_alignment", "object", "node", "edge", "lse")
        }
        used = 0
        progress = tqdm(loader, desc=f"Graph finetune {epoch + 1}/{args.epochs}")
        for scene in progress:
            optimizer.zero_grad(set_to_none=True)
            results = process_static_scene(scene, model.object_encoder, representation_loss=None, device=device)
            if results is None:
                continue
            objects = _unique_objects(scene, results["instance_tokens"])
            if len(objects) != len(results["instance_tokens"]):
                continue
            if encoder_trainable:
                outputs = results["encoded_scene"][0]["outputs"]
                categories = [str(obj.get("category_name", "unknown")) for obj in objects]
                representation = representation_loss(outputs, categories)
            else:
                zero = results["temporal_loss"]
                representation = {
                    "representation_loss": zero,
                    "diversity_loss": zero,
                    "part_alignment_loss": zero,
                    "object_loss": zero,
                }
            results["representation_loss"] = representation["representation_loss"]
            graph = graph_losses_for_3dssg_scene(
                scene=scene,
                results=results,
                model=model,
                label_to_id=label_to_id,
                graph_loss_fn=graph_loss,
                dynamic_loss_fn=dynamic_loss,
                device=device,
                relation_labels=relation_labels,
                max_negative_ratio=None,
                extended_geometry=config.extended_geometry,
                compute_dynamic=False,
            )
            total_loss = (
                representation["representation_loss"]
                + config.lambda_node * graph["node_loss"]
                + current_edge_lambda * (graph["edge_loss"] + 0.1 * graph["lse_loss"])
            )
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    grad_clip,
                )
            optimizer.step()

            used += 1
            values = {
                "total": total_loss,
                "representation": representation["representation_loss"],
                "diversity": representation["diversity_loss"],
                "part_alignment": representation["part_alignment_loss"],
                "object": representation["object_loss"],
                "node": graph["node_loss"],
                "edge": graph["edge_loss"],
                "lse": graph["lse_loss"],
            }
            for key, value in values.items():
                totals[key] += float(value.detach().cpu())
            progress.set_postfix(
                total=round(totals["total"] / used, 4),
                node=round(totals["node"] / used, 4),
                edge=round(totals["edge"] / used, 4),
            )

        epoch_metrics = {key: value / max(used, 1) for key, value in totals.items()}
        epoch_metrics.update(
            {
                "epoch": epoch + 1,
                "lambda_edge": current_edge_lambda,
                "encoder_trainable": float(encoder_trainable),
                "encoder_learning_rate": optimizer.param_groups[0]["lr"],
                "graph_learning_rate": optimizer.param_groups[1]["lr"],
            }
        )
        should_validate = (epoch + 1) % validation_interval == 0 or epoch + 1 == args.epochs
        if should_validate:
            validation = _validate(model, validation_data, label_to_id, relation_labels, config, device, graph_stage=True)
            epoch_metrics.update({f"val_{key}": value for key, value in validation.items()})
            _print_validation(validation)
        history.append(epoch_metrics)
        scheduler.step()

        current_score = float(epoch_metrics.get("val_selection_score", float("-inf")))
        is_best = should_validate and current_score >= best_score
        best_score = max(best_score, current_score)
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            config,
            label_to_id,
            id_to_label,
            relation_labels,
            history,
            "graph_finetuning",
            database_dir,
            best_score,
            extra={
                "seed": getattr(args, "seed", 42),
                "pretrain_checkpoint": str(args.pretrain_checkpoint),
                "edge_loss_mode": getattr(args, "edge_loss_mode", "hybrid"),
                "edge_balance_mix": float(getattr(args, "edge_balance_mix", 0.15)),
                "node_balance_mix": float(getattr(args, "node_balance_mix", 0.15)),
            },
        )
        _atomic_save(payload, latest_path)
        if is_best:
            _atomic_save(payload, best_path)
            print(f"Saved new best official-proxy checkpoint: {best_path}")
        (paths.output_root / "graph_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if not best_path.exists():
        _atomic_save(payload, best_path)
    print(f"Graph training complete. Best checkpoint: {best_path}")
    print(f"Latest checkpoint: {latest_path}")
    return best_path
