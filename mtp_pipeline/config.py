from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by the clean pipeline."""

    reference_root: Path = Path("pipeline_data")
    output_root: Path = Path("pipeline_outputs")

    @property
    def nuscenes_root(self) -> Path:
        return self.reference_root / "v1.0-mini"

    @property
    def rgb_tokens(self) -> Path:
        return self.reference_root / "rgb_patch_tokens"

    @property
    def lidar_tokens(self) -> Path:
        return self.reference_root / "lidar_patch_tokens"

    @property
    def text_tokens(self) -> Path:
        return self.reference_root / "text_embeddings"

    @property
    def scene_database(self) -> Path:
        return self.reference_root / "scene_database.pt"

    @property
    def checkpoints(self) -> Path:
        return self.output_root / "checkpoints"

    @property
    def graphs(self) -> Path:
        return self.output_root / "graphs"


@dataclass(frozen=True)
class PipelineConfig:
    """Hyperparameters aligned with the paper and slides."""

    dim: int = 384
    num_parts: int = 7
    # Category-name prompts remain representation supervision only in SGCls.
    graph_use_text: bool = False
    # ``lidar_only`` matches OCRL graph-stage inputs while retaining RGB/text
    # as cross-modal supervision for object pretraining.
    graph_input_mode: str = "multimodal"
    # Versioned v3 experiment: respect padded multi-view RGB tokens.
    use_rgb_token_mask: bool = False
    # Versioned v3 experiment: extend relation geometry from 11 to 16 dimensions.
    extended_geometry: bool = False
    # Optional v4 graph-only ablation; legacy Transformer context remains the default.
    graph_context: str = "transformer"
    graph_knn_neighbors: int = 0
    # Final two-stage experiment: condition shared part identities on each object's LiDAR context.
    conditioned_part_queries: bool = False
    # Initial contribution of the spatial residual in hybrid graph context.
    hybrid_spatial_init: float = 0.15
    num_heads: int = 8
    fusion_layers: int = 4
    temporal_layers: int = 4
    dropout: float = 0.1
    temperature: float = 0.07
    lambda_part: float = 1.0
    lambda_diversity: float | None = None
    lambda_part_alignment: float | None = None
    lambda_object: float = 1.0
    lambda_temporal: float = 1.0
    lambda_temporal_part: float = 0.5
    lambda_node: float = 1.0
    lambda_edge: float = 1.0
    lambda_dynamic: float = 0.5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    camera_priority: tuple[str, ...] = (
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
        "CAM_BACK_LEFT",
    )


def ensure_output_dirs(paths: ProjectPaths) -> None:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    paths.graphs.mkdir(parents=True, exist_ok=True)

