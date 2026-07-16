from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentDiversityLoss(nn.Module):
    """Slide 36: keep learned parts distinct."""

    def forward(self, components: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            components = components[mask]
        if components.numel() == 0:
            return components.new_zeros(())
        components = F.normalize(components, dim=-1)
        sim = torch.matmul(components, components.transpose(-2, -1))
        k = sim.size(-1)
        eye = torch.eye(k, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, 0.0)
        return sim.pow(2).sum(dim=(-2, -1)).div(k * (k - 1)).mean()


class PartAlignmentLoss(nn.Module):
    """Slide 37: InfoNCE alignment of corresponding RGB/LiDAR parts."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        rgb_components: torch.Tensor,
        lidar_components: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            rgb_components = rgb_components[mask]
            lidar_components = lidar_components[mask]
        if rgb_components.numel() == 0:
            return lidar_components.new_zeros(())
        b, k, d = rgb_components.shape
        rgb = F.normalize(rgb_components.reshape(b * k, d), dim=-1)
        lidar = F.normalize(lidar_components.reshape(b * k, d), dim=-1)
        logits = torch.matmul(rgb, lidar.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class ObjectContrastiveLoss(nn.Module):
    """Slide 39 object alignment without treating same-class prompts as negatives."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def _multi_positive_loss(self, a: torch.Tensor, b: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        if a.numel() == 0:
            return b.new_zeros(())
        logits = torch.matmul(F.normalize(a, dim=-1), F.normalize(b, dim=-1).T) / self.temperature
        positive = positive.to(device=logits.device, dtype=torch.bool)
        if not positive.any(dim=1).all() or not positive.any(dim=0).all():
            raise ValueError("Every aligned object must have at least one positive target.")
        log_denom_a = torch.logsumexp(logits, dim=1)
        log_num_a = torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)
        log_denom_b = torch.logsumexp(logits, dim=0)
        log_num_b = torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=0)
        return -0.5 * ((log_num_a - log_denom_a).mean() + (log_num_b - log_denom_b).mean())

    def _same_category_mask(self, categories: list[str], device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [[left == right for right in categories] for left in categories],
            device=device,
            dtype=torch.bool,
        )

    def forward(
        self,
        rgb_object: torch.Tensor,
        lidar_object: torch.Tensor,
        text_object: torch.Tensor,
        category_ids: list[str],
        rgb_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        category_positive = self._same_category_mask(category_ids, lidar_object.device)
        losses = [self._multi_positive_loss(lidar_object, text_object, category_positive)]
        if rgb_mask is not None:
            rgb_mask = rgb_mask.to(device=lidar_object.device, dtype=torch.bool)
        if rgb_mask is None or rgb_mask.any():
            if rgb_mask is None:
                rgb_mask = torch.ones(lidar_object.size(0), device=lidar_object.device, dtype=torch.bool)
            rgb_categories = [category for category, available in zip(category_ids, rgb_mask.tolist()) if available]
            count = len(rgb_categories)
            instance_positive = torch.eye(count, device=lidar_object.device, dtype=torch.bool)
            text_positive = self._same_category_mask(rgb_categories, lidar_object.device)
            losses.append(self._multi_positive_loss(rgb_object[rgb_mask], lidar_object[rgb_mask], instance_positive))
            losses.append(self._multi_positive_loss(rgb_object[rgb_mask], text_object[rgb_mask], text_positive))
        return torch.stack(losses).mean()


class RepresentationLoss(nn.Module):
    """Slides 36-39: part diversity, cross-modal part alignment, object alignment."""

    def __init__(self, lambda_part: float = 1.0, lambda_object: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.lambda_part = lambda_part
        self.lambda_object = lambda_object
        self.diversity = ComponentDiversityLoss()
        self.part_alignment = PartAlignmentLoss(temperature=temperature)
        self.object_alignment = ObjectContrastiveLoss(temperature=temperature)

    def forward(self, outputs: dict[str, torch.Tensor], category_ids: list[str]) -> dict[str, torch.Tensor]:
        rgb_mask = outputs.get("has_rgb_mask")
        if rgb_mask is not None:
            rgb_mask = rgb_mask.to(device=outputs["rgb_components"].device, dtype=torch.bool)
        diversity = self.diversity(outputs["lidar_components"]) + self.diversity(outputs["rgb_components"], rgb_mask)
        part_align = self.part_alignment(outputs["rgb_components"], outputs["lidar_components"], rgb_mask)
        object_loss = self.object_alignment(
            outputs["rgb_object"],
            outputs["lidar_object"],
            outputs["text_object"],
            category_ids=category_ids,
            rgb_mask=rgb_mask,
        )
        part_loss = diversity + part_align
        total = self.lambda_part * part_loss + self.lambda_object * object_loss
        return {
            "representation_loss": total,
            "part_loss": part_loss,
            "diversity_loss": diversity,
            "part_alignment_loss": part_align,
            "object_loss": object_loss,
        }


class TemporalConsistencyLoss(nn.Module):
    """Slide 41: contrastive temporal consistency over object tracklets."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if embeddings is None or mask is None or mask.sum() < 2:
            device = embeddings.device if embeddings is not None else mask.device
            return torch.zeros((), device=device, requires_grad=True)

        z = F.normalize(embeddings, dim=-1)
        b, t, d = z.shape
        flat = z.reshape(b * t, d)
        valid = mask.reshape(b * t)
        track_ids = torch.arange(b, device=z.device).unsqueeze(1).expand(b, t).reshape(b * t)
        time_ids = torch.arange(t, device=z.device).unsqueeze(0).expand(b, t).reshape(b * t)

        valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() < 2:
            return torch.zeros((), device=z.device, requires_grad=True)

        flat = flat[valid_idx]
        track_ids = track_ids[valid_idx]
        time_ids = time_ids[valid_idx]
        logits = torch.matmul(flat, flat.T) / self.temperature
        logits = logits.masked_fill(torch.eye(logits.size(0), device=z.device, dtype=torch.bool), -1e9)

        same_track = track_ids[:, None].eq(track_ids[None, :])
        different_time = time_ids[:, None].ne(time_ids[None, :])
        positives = same_track & different_time
        usable = positives.any(dim=1)
        if not usable.any():
            return torch.zeros((), device=z.device, requires_grad=True)

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pos_log_prob = (log_prob * positives.float()).sum(dim=1) / positives.float().sum(dim=1).clamp_min(1.0)
        return -pos_log_prob[usable].mean()


class TemporalPartAlignmentLoss(nn.Module):
    """Slide 42: same learned part of the same object should persist across frames."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def _info_nce(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        if anchor.numel() == 0 or positive.numel() == 0:
            return anchor.new_zeros(())
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = torch.matmul(anchor, positive.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def forward(self, encoded_scene: list[dict]) -> torch.Tensor:
        losses = []
        for t in range(len(encoded_scene) - 1):
            frame_t = encoded_scene[t]
            frame_tp1 = encoded_scene[t + 1]
            lookup_t = {obj["instance_token"]: idx for idx, obj in enumerate(frame_t["objects"])}
            lookup_tp1 = {obj["instance_token"]: idx for idx, obj in enumerate(frame_tp1["objects"])}
            common = sorted(set(lookup_t) & set(lookup_tp1))
            if not common:
                continue
            for key in ("lidar_components", "rgb_components"):
                comp_t = frame_t["outputs"][key]
                comp_tp1 = frame_tp1["outputs"][key]
                anchor = torch.cat([comp_t[lookup_t[inst]] for inst in common], dim=0)
                positive = torch.cat([comp_tp1[lookup_tp1[inst]] for inst in common], dim=0)
                losses.append(self._info_nce(anchor, positive))
        if not losses:
            if not encoded_scene:
                return torch.zeros((), requires_grad=True)
            return encoded_scene[0]["embeddings"].new_zeros(())
        return torch.stack(losses).mean()


class GraphPredictionLoss(nn.Module):
    """Node cross-entropy plus a multi-label focal loss for 26 relation outputs."""

    def __init__(self, focal_gamma: float = 2.0, alpha: torch.Tensor | None = None, lambda_lse: float = 0.1):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.lambda_lse = lambda_lse
        self.register_buffer("alpha", alpha)

    def edge_focal_loss(self, edge_logits: torch.Tensor, edge_targets: torch.Tensor) -> torch.Tensor:
        if edge_logits.shape != edge_targets.shape:
            raise ValueError(
                f"Expected multi-label edge targets shaped {tuple(edge_logits.shape)}, got {tuple(edge_targets.shape)}."
            )
        targets = edge_targets.to(device=edge_logits.device, dtype=edge_logits.dtype)
        probabilities = torch.sigmoid(edge_logits)
        bce = F.binary_cross_entropy_with_logits(edge_logits, targets, reduction="none")
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        loss = (1.0 - pt).pow(self.focal_gamma) * bce

        if self.alpha is not None:
            positive_weights = self.alpha.to(device=edge_logits.device, dtype=edge_logits.dtype).view(1, -1)
            weights = targets * positive_weights + (1.0 - targets)
            loss = loss * weights
        return loss.mean()

    def forward(
        self,
        node_logits: torch.Tensor,
        node_labels: torch.Tensor,
        edge_logits: torch.Tensor | None = None,
        edge_labels: torch.Tensor | None = None,
        geom_reconstruction: torch.Tensor | None = None,
        geom_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        node_loss = F.cross_entropy(node_logits, node_labels) if node_logits.numel() else node_logits.sum()
        if edge_logits is None or edge_labels is None or edge_logits.numel() == 0:
            edge_loss = node_loss.new_zeros(())
        else:
            edge_loss = self.edge_focal_loss(edge_logits, edge_labels)
        if geom_reconstruction is None or geom_targets is None or geom_reconstruction.numel() == 0:
            lse_loss = node_loss.new_zeros(())
        else:
            lse_loss = F.l1_loss(geom_reconstruction, geom_targets.to(geom_reconstruction))
        graph_loss = node_loss + edge_loss + self.lambda_lse * lse_loss
        return {"node_loss": node_loss, "edge_loss": edge_loss, "lse_loss": lse_loss, "graph_loss": graph_loss}


class DynamicEdgeConsistencyLoss(nn.Module):
    """Slide 44: KL consistency of persistent edge relation distributions."""

    def forward(self, edge_probs_by_frame: list[dict[tuple[str, str], torch.Tensor]]) -> torch.Tensor:
        losses = []
        device = None
        for frame_probs in edge_probs_by_frame:
            for prob in frame_probs.values():
                device = prob.device
                break
            if device is not None:
                break
        for t in range(len(edge_probs_by_frame) - 1):
            current = edge_probs_by_frame[t]
            nxt = edge_probs_by_frame[t + 1]
            for key in sorted(set(current) & set(nxt)):
                p = current[key].clamp_min(1e-8)
                q = nxt[key].clamp_min(1e-8)
                losses.append(F.kl_div(p.log(), q, reduction="batchmean"))
        if losses:
            return torch.stack(losses).mean()
        if device is None:
            return torch.zeros((), requires_grad=True)
        return torch.zeros((), device=device)
