from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .build_3rscan_static_database import build_3rscan_static_scene_database
from .evaluate_3dssg import evaluate_model
from .prepare_3rscan_lidar import prepare_3rscan_lidar_tokens
from .prepare_3rscan_rgb import prepare_3rscan_rgb_tokens
from .prepare_3rscan_text import prepare_3rscan_text_embeddings
from .protocol_3dssg import load_official_objects, write_official_objects_manifest
from .train_two_stage import pretrain_object_encoder, train_graph_from_pretrained


POINTNET_SHA256 = "7d14a73e5c26e83f03c3933b73c2b70f2e022225d0286a8b18007c23a29aaf24"
SEED = 42


@dataclass(frozen=True)
class RunLayout:
    """All paths needed for one isolated, resumable two-stage experiment."""

    project_root: Path
    scan_root: Path
    work_root: Path

    @property
    def official_splits(self) -> Path:
        return self.project_root / "official_splits"

    @property
    def alignment_metadata(self) -> Path:
        return self.project_root / "reference" / "3RScan.json"

    @property
    def pointnet_checkpoint(self) -> Path:
        return self.project_root / "reference" / "pretrained" / "obj_enc.pth"

    @property
    def data_root(self) -> Path:
        return self.work_root / "pipeline_data"

    @property
    def output_root(self) -> Path:
        return self.work_root / "pipeline_outputs"

    @property
    def object_manifest(self) -> Path:
        return self.data_root / "official_objects_manifest.json"

    @property
    def text_tokens(self) -> Path:
        return self.data_root / "3rscan_text_embeddings_v6"

    @property
    def rgb_tokens(self) -> Path:
        return self.data_root / "3rscan_rgb_views_v6"

    @property
    def lidar_tokens(self) -> Path:
        return self.data_root / "3rscan_pointnet_aligned_tokens_v6"

    @property
    def database(self) -> Path:
        return self.output_root / "3rscan_official_protocol_matched_database_v6"

    @property
    def experiment(self) -> Path:
        return self.output_root / "official_multimodal_parts_384_v6"

    @property
    def stage1_output(self) -> Path:
        return self.experiment / "stage1_part_pretrain"

    @property
    def stage2_output(self) -> Path:
        return self.experiment / "stage2_graph"

    @property
    def train_scans(self) -> Path:
        return self.official_splits / "train_scans.txt"

    @property
    def validation_scans(self) -> Path:
        return self.official_splits / "validation_scans.txt"


def default_layout(scan_root: Path, work_root: Path | None = None) -> RunLayout:
    project_root = Path(__file__).resolve().parents[1]
    return RunLayout(
        project_root=project_root,
        scan_root=scan_root.expanduser().resolve(),
        work_root=(work_root or project_root).expanduser().resolve(),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_static_dependencies(layout: RunLayout) -> None:
    required = [
        layout.official_splits / "classes.txt",
        layout.official_splits / "relations.txt",
        layout.official_splits / "relationships_train.json",
        layout.official_splits / "relationships_validation.json",
        layout.official_splits / "train_scans.txt",
        layout.official_splits / "validation_scans.txt",
        layout.alignment_metadata,
        layout.pointnet_checkpoint,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing committed protocol files:\n" + "\n".join(missing))
    pointnet_hash = _sha256(layout.pointnet_checkpoint)
    if pointnet_hash != POINTNET_SHA256:
        raise ValueError(
            "The PointNet checkpoint does not match the verified OCRL object encoder. "
            f"Expected {POINTNET_SHA256}, found {pointnet_hash}."
        )


def write_manifest(layout: RunLayout) -> Path:
    _validate_static_dependencies(layout)
    manifest = write_official_objects_manifest(layout.official_splits, layout.object_manifest)
    scan_count = len(load_official_objects(layout.official_splits))
    if scan_count != 1335:
        raise RuntimeError(f"Expected 1335 official scan groups, found {scan_count}.")
    print(f"Generated {manifest} for {scan_count} official scan groups.")
    return manifest


def prepare_text(layout: RunLayout, device: str) -> None:
    manifest = write_manifest(layout)
    prepare_3rscan_text_embeddings(
        objects_json=manifest,
        output_dir=layout.text_tokens,
        model_name="openai/clip-vit-base-patch32",
        device=device,
    )


def prepare_rgb(layout: RunLayout, device: str) -> None:
    if not layout.scan_root.exists():
        raise FileNotFoundError(f"3RScan root not found: {layout.scan_root}")
    manifest = write_manifest(layout)
    prepare_3rscan_rgb_tokens(
        scan_dir=layout.scan_root,
        objects_json=manifest,
        output_dir=layout.rgb_tokens,
        model_name="openai/clip-vit-base-patch32",
        device=device,
    )


def prepare_lidar(layout: RunLayout, device: str) -> None:
    if not layout.scan_root.exists():
        raise FileNotFoundError(f"3RScan root not found: {layout.scan_root}")
    _validate_static_dependencies(layout)
    prepare_3rscan_lidar_tokens(
        scan_dir=layout.scan_root,
        objects_json=None,
        output_dir=layout.lidar_tokens,
        checkpoint=layout.pointnet_checkpoint,
        device=device,
        official_subset_dir=layout.official_splits,
        alignment_metadata=layout.alignment_metadata,
        points_per_object=128,
        tokens_per_object=64,
    )


def build_database(layout: RunLayout) -> Path:
    saved = build_3rscan_static_scene_database(
        objects_json=None,
        relationships_json=None,
        official_subset_dir=layout.official_splits,
        lidar_dir=layout.lidar_tokens,
        text_dir=layout.text_tokens,
        rgb_dir=layout.rgb_tokens,
        output_dir=layout.database,
        rgb_views_per_object=3,
        exclude_ocr_invalid_scan=True,
        require_aligned_lidar=True,
        refresh_if_inputs_newer=True,
    )
    if saved != 4393:
        raise RuntimeError(f"Expected 4393 OCRL-compatible graph entries, found {saved}.")
    label_space = json.loads((layout.database / "label_space.json").read_text(encoding="utf-8"))
    if label_space["num_object_classes"] != 160 or label_space["num_relation_classes"] != 26:
        raise RuntimeError("The database is not using the official 160-object/26-predicate space.")
    print("Protocol database ready: 3845 training + 548 validation entries.")
    return layout.database


def prepare_protocol_data(layout: RunLayout, device: str = "cuda") -> Path:
    """Run all resumable preparation stages from raw 3RScan data."""
    prepare_text(layout, device)
    prepare_rgb(layout, device)
    prepare_lidar(layout, device)
    return build_database(layout)


def _common_args(layout: RunLayout, output_root: Path, device: str) -> dict:
    return {
        "reference_root": layout.data_root,
        "output_root": output_root,
        "database": layout.database,
        "train_scans": layout.train_scans,
        "val_scans": layout.validation_scans,
        "split_seed": SEED,
        "seed": SEED,
        "max_scenes": None,
        "max_validation_scenes": None,
        "max_frames": None,
        "embedding_dim": 384,
        "num_parts": 7,
        "graph_input_mode": "multimodal",
        "use_rgb_token_mask": True,
        "extended_geometry": False,
        "graph_context": "hybrid",
        "graph_knn_neighbors": 8,
        "conditioned_part_queries": True,
        "hybrid_spatial_init": 0.15,
        "exclude_ocr_invalid_scan": True,
        "device": device,
    }


def run_stage1(layout: RunLayout, epochs: int = 20, device: str = "cuda", resume: bool = True) -> Path:
    args = _common_args(layout, layout.stage1_output, device)
    latest = layout.stage1_output / "checkpoints" / "object_pretrain_latest.pt"
    args.update(
        {
            "epochs": epochs,
            "learning_rate": 2e-4,
            "lambda_diversity": 0.05,
            "lambda_part_alignment": 0.30,
            "lambda_object": 1.00,
            "lambda_pretrain_node": 1.00,
            "node_label_smoothing": 0.02,
            "validation_interval": 2,
            "class_weight_cache": layout.experiment / "train_object_weights.pt",
            "grad_clip": 1.0,
            "resume_checkpoint": latest if resume and latest.exists() else None,
        }
    )
    return pretrain_object_encoder(argparse.Namespace(**args))


def run_stage2(
    layout: RunLayout,
    pretrain_checkpoint: Path | None = None,
    epochs: int = 80,
    device: str = "cuda",
    resume: bool = True,
) -> Path:
    pretrain_checkpoint = pretrain_checkpoint or (
        layout.stage1_output / "checkpoints" / "object_pretrain_best.pt"
    )
    if not pretrain_checkpoint.exists():
        raise FileNotFoundError(f"Run Stage 1 first; missing {pretrain_checkpoint}")
    args = _common_args(layout, layout.stage2_output, device)
    latest = layout.stage2_output / "checkpoints" / "graph_latest.pt"
    args.update(
        {
            "epochs": epochs,
            "learning_rate": 1e-4,
            "lambda_diversity": 0.02,
            "lambda_part_alignment": 0.10,
            "lambda_object": 0.25,
            "lambda_node": 1.0,
            "lambda_edge": 1.0,
            "edge_loss_mode": "hybrid",
            "edge_balance_mix": 0.05,
            "edge_negative_weight": 1.0,
            "node_balance_mix": 0.05,
            "node_label_smoothing": 0.02,
            "edge_warmup_epochs": 3,
            "edge_warmup_start": 0.70,
            "encoder_freeze_epochs": 2,
            "encoder_lr_scale": 0.10,
            "validation_interval": 5,
            "class_weight_cache": layout.experiment / "train_object_weights.pt",
            "grad_clip": 1.0,
            "pretrain_checkpoint": pretrain_checkpoint,
            "resume_checkpoint": latest if resume and latest.exists() else None,
        }
    )
    return train_graph_from_pretrained(argparse.Namespace(**args))


def run_official_evaluation(
    layout: RunLayout,
    checkpoint: Path,
    device: str = "cuda",
    max_scenes: int | None = None,
) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    result_output = layout.stage2_output / "evaluation" / f"evaluation_official_{checkpoint.stem}.json"
    result_output.parent.mkdir(parents=True, exist_ok=True)
    evaluate_model(
        argparse.Namespace(
            checkpoint=checkpoint,
            reference_root=layout.data_root,
            output_root=layout.stage2_output,
            database=layout.database,
            mode="static",
            train_scans=layout.train_scans,
            val_scans=layout.validation_scans,
            split_seed=SEED,
            device=device,
            max_scenes=max_scenes,
            result_output=result_output,
        )
    )


def verify_prepared_data(layout: RunLayout) -> dict[str, object]:
    _validate_static_dependencies(layout)
    report: dict[str, object] = {
        "official_scan_groups": len(load_official_objects(layout.official_splits)),
        "pointnet_sha256": _sha256(layout.pointnet_checkpoint),
        "text_object_files": len(list(layout.text_tokens.glob("*.pt"))),
        "rgb_scene_files": len(list(layout.rgb_tokens.glob("*.pt"))),
        "lidar_object_files": len(list(layout.lidar_tokens.glob("*.pt"))),
        "database_entries": len(list(layout.database.glob("*.pt"))),
    }
    report["database_ready"] = report["database_entries"] == 4393
    print(json.dumps(report, indent=2))
    return report


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the two-stage Integration-by-Parts result from raw 3RScan data."
    )
    parser.add_argument(
        "action",
        choices=("manifest", "prepare-text", "prepare-rgb", "prepare-lidar", "build-database", "prepare", "stage1", "stage2", "train", "evaluate", "verify", "all"),
    )
    parser.add_argument("--scan-root", type=Path, default=Path("3RScan"))
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="Do not resume from latest training checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = default_layout(args.scan_root, args.work_root)
    device = _device(args.device)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = layout.project_root / "artifacts" / "checkpoints" / "graph_best_official_proxy.pt"

    if args.action == "manifest":
        write_manifest(layout)
    elif args.action == "prepare-text":
        prepare_text(layout, device)
    elif args.action == "prepare-rgb":
        prepare_rgb(layout, device)
    elif args.action == "prepare-lidar":
        prepare_lidar(layout, device)
    elif args.action == "build-database":
        build_database(layout)
    elif args.action == "prepare":
        prepare_protocol_data(layout, device)
    elif args.action == "stage1":
        run_stage1(layout, args.stage1_epochs, device, resume=not args.fresh)
    elif args.action == "stage2":
        run_stage2(layout, epochs=args.stage2_epochs, device=device, resume=not args.fresh)
    elif args.action == "train":
        stage1 = run_stage1(layout, args.stage1_epochs, device, resume=not args.fresh)
        run_stage2(layout, stage1, args.stage2_epochs, device, resume=not args.fresh)
    elif args.action == "evaluate":
        run_official_evaluation(layout, checkpoint.resolve(), device, args.max_scenes)
    elif args.action == "verify":
        verify_prepared_data(layout)
    elif args.action == "all":
        prepare_protocol_data(layout, device)
        stage1 = run_stage1(layout, args.stage1_epochs, device, resume=not args.fresh)
        best = run_stage2(layout, stage1, args.stage2_epochs, device, resume=not args.fresh)
        run_official_evaluation(layout, best, device)


if __name__ == "__main__":
    main()
