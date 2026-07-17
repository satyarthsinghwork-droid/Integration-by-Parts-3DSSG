from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputProjection(nn.Module):
    """Project pretrained RGB and text features into the common d-dimensional space."""

    def __init__(self, dim: int = 384, rgb_dim: int = 768, lidar_dim: int = 512, text_dim: int = 512):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, dim)
        self.lidar_proj = nn.Linear(lidar_dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)

    def forward(self, rgb_tokens: torch.Tensor, lidar_tokens: torch.Tensor, text_features: torch.Tensor):
        return self.rgb_proj(rgb_tokens), self.lidar_proj(lidar_tokens), self.text_proj(text_features)


class ComponentQueries(nn.Module):
    """Shared part identities with optional object-conditioned offsets."""

    def __init__(self, num_parts: int = 8, dim: int = 384, conditioned: bool = False):
        super().__init__()
        self.num_parts = num_parts
        self.dim = dim
        self.conditioned = conditioned
        self.queries = nn.Parameter(torch.randn(num_parts, dim) * 0.02)
        if conditioned:
            self.conditioner = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, num_parts * dim),
            )
            # Start as the original shared-query model and learn specialization gradually.
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)
        else:
            self.conditioner = None

    def forward(self, batch_size: int, object_context: torch.Tensor | None = None) -> torch.Tensor:
        base = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        if self.conditioner is None:
            return base
        if object_context is None or object_context.shape != (batch_size, self.dim):
            raise ValueError(f"Expected object context {(batch_size, self.dim)}, got {getattr(object_context, 'shape', None)}.")
        offsets = self.conditioner(object_context).view(batch_size, self.num_parts, self.dim)
        return base + 0.25 * torch.tanh(offsets)


class CrossAttention(nn.Module):
    """Component query attention over modality tokens, matching equations 1-6."""

    def __init__(self, dim: int = 384):
        super().__init__()
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

    def forward(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ):
        q = self.wq(queries)
        k = self.wk(tokens)
        v = self.wv(tokens)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if token_mask is not None:
            token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
            if token_mask.shape != tokens.shape[:2]:
                raise ValueError(f"Expected token mask {tuple(tokens.shape[:2])}, got {tuple(token_mask.shape)}.")
            valid = token_mask.any(dim=1, keepdim=True).unsqueeze(1)
            scores = scores.masked_fill(~token_mask.unsqueeze(1), -1e4)
            attention = torch.softmax(scores, dim=-1)
            attention = torch.where(valid, attention, torch.zeros_like(attention))
        else:
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

    def __init__(self, dim: int = 384, num_parts: int = 8, conditioned_part_queries: bool = False):
        super().__init__()
        self.projector = InputProjection(dim=dim)
        self.query_layer = ComponentQueries(
            num_parts=num_parts,
            dim=dim,
            conditioned=conditioned_part_queries,
        )
        self.rgb_attention = CrossAttention(dim=dim)
        self.lidar_attention = CrossAttention(dim=dim)
        self.pool = ObjectPooling(dim=dim)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
        text_features: torch.Tensor,
        rgb_token_mask: torch.Tensor | None = None,
    ):
        rgb_tokens, lidar_tokens, text_object = self.projector(rgb_tokens, lidar_tokens, text_features)
        # LiDAR is available for every official object and provides one shared,
        # label-free context for both RGB and LiDAR part queries. Text remains global.
        object_context = lidar_tokens.mean(dim=1)
        queries = self.query_layer(rgb_tokens.size(0), object_context=object_context)

        rgb_components, rgb_attention = self.rgb_attention(queries, rgb_tokens, token_mask=rgb_token_mask)
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

    def forward(
        self,
        rgb_object: torch.Tensor,
        lidar_object: torch.Tensor,
        text_object: torch.Tensor,
        has_rgb_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = rgb_object.size(0)
        if has_rgb_mask is not None:
            has_rgb_mask = has_rgb_mask.to(device=rgb_object.device, dtype=torch.bool).unsqueeze(-1)
            rgb_object = torch.where(has_rgb_mask, rgb_object, torch.zeros_like(rgb_object))
        fusion = self.fusion_token.expand(batch_size, -1, -1)
        sequence = torch.stack([rgb_object, lidar_object, text_object], dim=1)
        sequence = torch.cat([fusion, sequence], dim=1)
        return self.norm(self.transformer(sequence)[:, 0])


class SceneRepresentationModel(nn.Module):
    """Equations 1-12: component learning, alignment outputs, and object fusion."""

    def __init__(
        self,
        dim: int = 384,
        num_parts: int = 8,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        graph_use_text: bool = False,
        graph_input_mode: str = "multimodal",
        use_rgb_token_mask: bool = False,
        conditioned_part_queries: bool = False,
    ):
        super().__init__()
        self.component_encoder = MultimodalComponentEncoder(
            dim=dim,
            num_parts=num_parts,
            conditioned_part_queries=conditioned_part_queries,
        )
        self.fusion = FusionTransformer(dim=dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        self.graph_use_text = graph_use_text
        if graph_input_mode not in {"multimodal", "lidar_only"}:
            raise ValueError(f"Unknown graph input mode: {graph_input_mode}")
        self.graph_input_mode = graph_input_mode
        self.use_rgb_token_mask = use_rgb_token_mask

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
        text_features: torch.Tensor,
        has_rgb_mask: torch.Tensor | None = None,
        rgb_token_mask: torch.Tensor | None = None,
    ):
        outputs = self.component_encoder(
            rgb_tokens,
            lidar_tokens,
            text_features,
            rgb_token_mask=rgb_token_mask if self.use_rgb_token_mask else None,
        )
        if has_rgb_mask is not None:
            outputs["has_rgb_mask"] = has_rgb_mask.to(device=outputs["rgb_object"].device, dtype=torch.bool)
        # The 3DSSG text vector is generated from the annotated class name.
        # It supervises cross-modal representation learning but must not reveal
        # the SGCls target to the graph classifier.
        graph_text = outputs["text_object"] if self.graph_use_text else torch.zeros_like(outputs["text_object"])
        if self.graph_input_mode == "lidar_only":
            outputs["fused_object"] = outputs["lidar_object"]
        else:
            outputs["fused_object"] = self.fusion(
                outputs["rgb_object"],
                outputs["lidar_object"],
                graph_text,
                has_rgb_mask=outputs.get("has_rgb_mask"),
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


class SpatialBidirectionalContext(nn.Module):
    """Geometry-aware KNN message passing with independently gated directions."""

    def __init__(self, dim: int, geom_dim: int, neighbors: int, dropout: float = 0.1):
        super().__init__()
        self.neighbors = neighbors
        self.geom = nn.Sequential(nn.LayerNorm(geom_dim), nn.Linear(geom_dim, dim), nn.GELU())
        self.message = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.update = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)

    def _knn_edges(self, edge_index: torch.Tensor, geometry: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if self.neighbors <= 0 or edge_index.numel() == 0:
            return torch.arange(edge_index.size(0), device=edge_index.device)
        distance = torch.linalg.vector_norm(geometry[:, :3], dim=-1)
        selected: list[torch.Tensor] = []
        for target in range(num_nodes):
            candidates = torch.nonzero(edge_index[:, 1] == target, as_tuple=False).flatten()
            if candidates.numel() <= self.neighbors:
                selected.append(candidates)
            elif candidates.numel():
                nearest = torch.topk(distance[candidates], self.neighbors, largest=False).indices
                selected.append(candidates[nearest])
        return torch.cat(selected) if selected else edge_index.new_empty((0,), dtype=torch.long)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, geometry: torch.Tensor | None) -> torch.Tensor:
        if node_features.numel() == 0 or edge_index.numel() == 0 or geometry is None:
            return node_features
        selected = self._knn_edges(edge_index, geometry, node_features.size(0))
        if selected.numel() == 0:
            return node_features
        edges = edge_index[selected]
        geom = self.geom(geometry[selected].to(dtype=node_features.dtype))
        source, target = edges[:, 0], edges[:, 1]
        source_features = node_features[source]
        target_features = node_features[target]
        gate = torch.sigmoid(self.gate(torch.cat([source_features, target_features, geom], dim=-1)))
        messages = gate * (self.message(source_features) + geom)
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, target, messages)
        counts = torch.zeros((node_features.size(0), 1), dtype=node_features.dtype, device=node_features.device)
        counts.index_add_(0, target, torch.ones((target.numel(), 1), dtype=node_features.dtype, device=node_features.device))
        return self.norm(node_features + self.update(aggregated / counts.clamp_min(1.0)))


class DynamicSceneGraphModel(nn.Module):
    """Final integrated model: representation, temporal association, and graph heads."""

    def __init__(self, num_node_classes: int, num_edge_classes: int, dim: int = 384, num_parts: int = 8,
                 num_heads: int = 8, fusion_layers: int = 2, temporal_layers: int = 2, dropout: float = 0.1,
                 graph_use_text: bool = False, graph_input_mode: str = "multimodal",
                 use_rgb_token_mask: bool = False,
                 extended_geometry: bool = False, graph_context: str = "transformer",
                 graph_knn_neighbors: int = 0, conditioned_part_queries: bool = False,
                 hybrid_spatial_init: float = 0.15):
        super().__init__()
        self.object_encoder = SceneRepresentationModel(
            dim=dim,
            num_parts=num_parts,
            num_heads=num_heads,
            num_layers=fusion_layers,
            dropout=dropout,
            graph_use_text=graph_use_text,
            graph_input_mode=graph_input_mode,
            use_rgb_token_mask=use_rgb_token_mask,
            conditioned_part_queries=conditioned_part_queries,
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
        
        self.graph_context = graph_context
        geom_dim = 16 if extended_geometry else 11
        self.spatial_context = (
            SpatialBidirectionalContext(dim=dim, geom_dim=geom_dim, neighbors=graph_knn_neighbors, dropout=dropout)
            if graph_context in {"spatial_gated", "hybrid"} else None
        )
        if graph_context == "hybrid":
            initial = min(max(float(hybrid_spatial_init), 1e-4), 1.0 - 1e-4)
            initial_bias = math.log(initial / (1.0 - initial))
            self.hybrid_gate = nn.Sequential(
                nn.LayerNorm(dim * 3),
                nn.Linear(dim * 3, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
            )
            nn.init.zeros_(self.hybrid_gate[-1].weight)
            nn.init.constant_(self.hybrid_gate[-1].bias, initial_bias)
            self.hybrid_norm = nn.LayerNorm(dim)
        else:
            self.hybrid_gate = None
            self.hybrid_norm = None
        self.node_head = NodeClassifier(dim=dim, num_classes=num_node_classes)
        self.edge_head = EdgeClassifier(
            dim=dim,
            num_relations=num_edge_classes,
            geom_dim=geom_dim,
        )

    def encode_objects(self, rgb_tokens: torch.Tensor, lidar_tokens: torch.Tensor, text_features: torch.Tensor):
        return self.object_encoder(rgb_tokens, lidar_tokens, text_features)

    def _apply_context(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        geom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_features.numel() == 0:
            return node_features
        if self.graph_context == "spatial_gated":
            return self.spatial_context(node_features, edge_index, geom_features) if self.spatial_context is not None and edge_index is not None else node_features
        # The hybrid starts close to the strong global baseline and learns how
        # much local geometry to add for each object.
        seq = node_features.unsqueeze(0)
        global_features = self.context_layer(seq).squeeze(0)
        if self.graph_context != "hybrid" or edge_index is None or self.spatial_context is None:
            return global_features
        spatial_features = self.spatial_context(node_features, edge_index, geom_features)
        gate_input = torch.cat([node_features, global_features, spatial_features], dim=-1)
        spatial_gate = torch.sigmoid(self.hybrid_gate(gate_input))
        return self.hybrid_norm(global_features + spatial_gate * (spatial_features - node_features))

    def predict_nodes(self, node_features: torch.Tensor) -> torch.Tensor:
        context_features = self._apply_context(node_features)
        return self.node_head(context_features)

    def predict_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict nodes and edges from one shared contextualized object set."""
        context_features = self._apply_context(node_features, edge_index, geom_features)
        node_logits = self.node_head(context_features)
        if edge_index.numel() == 0:
            edge_logits = context_features.new_zeros((0, self.edge_head.num_relations))
            geom_reconstruction = context_features.new_zeros((0, self.edge_head.geom_dim))
        else:
            edge_logits, geom_reconstruction = self.edge_head(
                context_features[edge_index[:, 0]],
                context_features[edge_index[:, 1]],
                geom_features,
            )
        return node_logits, edge_logits, geom_reconstruction

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
            
        context_features = self._apply_context(node_features, edge_index, geom_features)
        logits, geom_recon = self.edge_head(
            context_features[edge_index[:, 0]],
            context_features[edge_index[:, 1]],
            geom_features,
        )
        return (logits, geom_recon) if return_aux else logits

