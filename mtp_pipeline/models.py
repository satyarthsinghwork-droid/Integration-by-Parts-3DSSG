from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputProjection(nn.Module):
    """Project pretrained RGB and text features into the common d-dimensional space."""

    def __init__(self, dim: int = 384, rgb_dim: int = 768, text_dim: int = 512):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)

    def forward(self, rgb_tokens: torch.Tensor, lidar_tokens: torch.Tensor, text_features: torch.Tensor):
        return self.rgb_proj(rgb_tokens), lidar_tokens, self.text_proj(text_features)


class ComponentQueries(nn.Module):
    """Shared learnable part queries Q_P from the paper."""

    def __init__(self, num_parts: int = 8, dim: int = 384):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_parts, dim) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.queries.unsqueeze(0).expand(batch_size, -1, -1)


class CrossAttention(nn.Module):
    """Component query attention over modality tokens, matching equations 1-6."""

    def __init__(self, dim: int = 384):
        super().__init__()
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor):
        q = self.wq(queries)
        k = self.wk(tokens)
        v = self.wv(tokens)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attention = torch.softmax(scores, dim=-1)
        components = torch.matmul(attention, v)
        return components, attention


class ObjectPooling(nn.Module):
    """Attention pooling from components to modality-specific object embeddings."""

    def __init__(self, dim: int = 384):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, components: torch.Tensor):
        alpha = torch.softmax(self.score(components), dim=1)
        return (alpha * components).sum(dim=1), alpha


class MultimodalComponentEncoder(nn.Module):
    """Paper stages: component discovery, RGB/LiDAR components, object pooling."""

    def __init__(self, dim: int = 384, num_parts: int = 8):
        super().__init__()
        self.projector = InputProjection(dim=dim)
        self.query_layer = ComponentQueries(num_parts=num_parts, dim=dim)
        self.rgb_attention = CrossAttention(dim=dim)
        self.lidar_attention = CrossAttention(dim=dim)
        self.pool = ObjectPooling(dim=dim)

    def forward(self, rgb_tokens: torch.Tensor, lidar_tokens: torch.Tensor, text_features: torch.Tensor):
        rgb_tokens, lidar_tokens, text_object = self.projector(rgb_tokens, lidar_tokens, text_features)
        queries = self.query_layer(rgb_tokens.size(0))

        rgb_components, rgb_attention = self.rgb_attention(queries, rgb_tokens)
        lidar_components, lidar_attention = self.lidar_attention(queries, lidar_tokens)
        rgb_object, rgb_alpha = self.pool(rgb_components)
        lidar_object, lidar_alpha = self.pool(lidar_components)

        return {
            "rgb_components": rgb_components,
            "lidar_components": lidar_components,
            "rgb_object": rgb_object,
            "lidar_object": lidar_object,
            "text_object": text_object,
            "rgb_attention": rgb_attention,
            "lidar_attention": lidar_attention,
            "rgb_alpha": rgb_alpha,
            "lidar_alpha": lidar_alpha,
        }


class FusionTransformer(nn.Module):
    """Transformer fusion token module for equation 12."""

    def __init__(self, dim: int = 384, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fusion_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, rgb_object: torch.Tensor, lidar_object: torch.Tensor, text_object: torch.Tensor) -> torch.Tensor:
        batch_size = rgb_object.size(0)
        fusion = self.fusion_token.expand(batch_size, -1, -1)
        sequence = torch.stack([rgb_object, lidar_object, text_object], dim=1)
        sequence = torch.cat([fusion, sequence], dim=1)
        return self.norm(self.transformer(sequence)[:, 0])


class SceneRepresentationModel(nn.Module):
    """Equations 1-12: component learning, alignment outputs, and object fusion."""

    def __init__(self, dim: int = 384, num_parts: int = 8, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.component_encoder = MultimodalComponentEncoder(dim=dim, num_parts=num_parts)
        self.fusion = FusionTransformer(dim=dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)

    def forward(self, rgb_tokens: torch.Tensor, lidar_tokens: torch.Tensor, text_features: torch.Tensor):
        outputs = self.component_encoder(rgb_tokens, lidar_tokens, text_features)
        outputs["fused_object"] = self.fusion(
            outputs["rgb_object"], outputs["lidar_object"], outputs["text_object"]
        )
        return outputs


class SoftObjectAssociation(nn.Module):
    """Soft correspondence matrix A_{t,t+1}, equation 13."""

    def __init__(self, dim: int = 384):
        super().__init__()
        self.scale = math.sqrt(dim)

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        similarity = torch.matmul(z_t, z_tp1.transpose(-2, -1)) / self.scale
        return torch.softmax(similarity, dim=-1)


class TemporalAggregation(nn.Module):
    """Soft aggregation of next-frame objects, equation 14."""

    def forward(self, association: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        return torch.matmul(association, z_tp1)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, dim: int = 384, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)


class TemporalTransformer(nn.Module):
    """Temporal contextualization of tracklets, equation 15."""

    def __init__(self, dim: int = 384, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.position = TemporalPositionalEncoding(dim=dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tracklets: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        output = self.transformer(self.position(tracklets), src_key_padding_mask=src_key_padding_mask)
        return self.norm(output)


class NodeClassifier(nn.Module):
    """Node prediction head for equations 17-18."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_classes),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.net(node_features)


class EdgeClassifier(nn.Module):
    """Geometry-aware directed edge head with an LSE reconstruction branch."""

    def __init__(self, dim: int, num_relations: int, geom_dim: int = 11):
        super().__init__()
        self.num_relations = num_relations
        self.geom_dim = geom_dim
        self.geom_proj = nn.Sequential(
            nn.LayerNorm(geom_dim),
            nn.Linear(geom_dim, dim),
            nn.GELU(),
        )
        self.hidden = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(dim, num_relations)
        self.geom_reconstruct = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, geom_dim),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if geom_features is None:
            geom_features = source.new_zeros((source.size(0), self.geom_dim))
        geom_features = geom_features.to(device=source.device, dtype=source.dtype)
        geom_embedding = self.geom_proj(geom_features)
        pair = torch.cat([source, target, source * target, geom_embedding], dim=-1)
        hidden = self.hidden(pair)
        return self.classifier(hidden), self.geom_reconstruct(hidden)


class DynamicSceneGraphModel(nn.Module):
    """Final integrated model: representation, temporal association, and graph heads."""

    def __init__(self, num_node_classes: int, num_edge_classes: int, dim: int = 384, num_parts: int = 8,
                 num_heads: int = 8, fusion_layers: int = 2, temporal_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.object_encoder = SceneRepresentationModel(
            dim=dim,
            num_parts=num_parts,
            num_heads=num_heads,
            num_layers=fusion_layers,
            dropout=dropout,
        )
        self.association = SoftObjectAssociation(dim=dim)
        self.aggregation = TemporalAggregation()
        self.temporal = TemporalTransformer(dim=dim, num_heads=num_heads, num_layers=temporal_layers, dropout=dropout)
        
        # GNN Context Layer (Message Passing between objects in the scene)
        self.context_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        
        self.node_head = NodeClassifier(dim=dim, num_classes=num_node_classes)
        self.edge_head = EdgeClassifier(dim=dim, num_relations=num_edge_classes)

    def encode_objects(self, rgb_tokens: torch.Tensor, lidar_tokens: torch.Tensor, text_features: torch.Tensor):
        return self.object_encoder(rgb_tokens, lidar_tokens, text_features)

    def _apply_context(self, node_features: torch.Tensor) -> torch.Tensor:
        if node_features.numel() == 0:
            return node_features
        # Add batch dimension for TransformerEncoderLayer (B, S, E) where B=1
        seq = node_features.unsqueeze(0)
        out = self.context_layer(seq)
        return out.squeeze(0)

    def predict_nodes(self, node_features: torch.Tensor) -> torch.Tensor:
        context_features = self._apply_context(node_features)
        return self.node_head(context_features)

    def predict_edges(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        geom_features: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if edge_index.numel() == 0:
            logits = node_features.new_zeros((0, self.edge_head.num_relations))
            geom_recon = node_features.new_zeros((0, self.edge_head.geom_dim))
            return (logits, geom_recon) if return_aux else logits
            
        context_features = self._apply_context(node_features)
        logits, geom_recon = self.edge_head(
            context_features[edge_index[:, 0]],
            context_features[edge_index[:, 1]],
            geom_features,
        )
        return (logits, geom_recon) if return_aux else logits

