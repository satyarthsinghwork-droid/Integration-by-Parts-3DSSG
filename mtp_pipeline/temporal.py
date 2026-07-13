from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import stack_frame_tensors


def encode_frame(frame: dict[str, Any], scene_model: torch.nn.Module, device: torch.device | str) -> dict[str, Any]:
    rgb_tokens, lidar_tokens, text_features = stack_frame_tensors(frame, device=device)
    outputs = scene_model(rgb_tokens, lidar_tokens, text_features)
    return {
        "sample_token": frame["sample_token"],
        "timestamp": frame["timestamp"],
        "objects": frame["objects"],
        "outputs": outputs,
        "embeddings": outputs["fused_object"],
    }


def encode_scene(
    scene: dict[str, Any],
    scene_model: torch.nn.Module,
    device: torch.device | str,
    representation_loss: torch.nn.Module | None = None,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    encoded = []
    rep_losses = []
    for frame in scene["frames"]:
        frame_encoded = encode_frame(frame, scene_model, device=device)
        encoded.append(frame_encoded)
        if representation_loss is not None:
            rep_losses.append(representation_loss(frame_encoded["outputs"])["representation_loss"])

    if rep_losses:
        rep_loss = torch.stack(rep_losses).mean()
    else:
        first_device = encoded[0]["embeddings"].device if encoded else torch.device(device)
        rep_loss = torch.zeros((), device=first_device)
    return encoded, rep_loss


def build_tracklets(
    encoded_scene: list[dict[str, Any]],
    association_model: torch.nn.Module,
    aggregation_model: torch.nn.Module,
) -> tuple[dict[str, list[torch.Tensor]], list[dict[str, Any]]]:
    tracklets: dict[str, list[torch.Tensor]] = defaultdict(list)
    associations = []

    for t in range(len(encoded_scene) - 1):
        frame_t = encoded_scene[t]
        frame_tp1 = encoded_scene[t + 1]
        lookup_t = {obj["instance_token"]: idx for idx, obj in enumerate(frame_t["objects"])}
        lookup_tp1 = {obj["instance_token"]: idx for idx, obj in enumerate(frame_tp1["objects"])}
        common = sorted(set(lookup_t) & set(lookup_tp1))
        if not common:
            continue

        association = association_model(frame_t["embeddings"], frame_tp1["embeddings"])
        temporal = aggregation_model(association, frame_tp1["embeddings"])
        associations.append(
            {
                "t": t,
                "sample_t": frame_t["sample_token"],
                "sample_tp1": frame_tp1["sample_token"],
                "association": association,
                "instance_t": [obj["instance_token"] for obj in frame_t["objects"]],
                "instance_tp1": [obj["instance_token"] for obj in frame_tp1["objects"]],
            }
        )

        for instance in common:
            idx_t = lookup_t[instance]
            if instance not in tracklets:
                tracklets[instance].append(frame_t["embeddings"][idx_t])
            tracklets[instance].append(temporal[idx_t])

    return tracklets, associations


def pad_tracklets(tracklets: dict[str, list[torch.Tensor]]):
    sequences = []
    instance_tokens = []
    for instance, values in tracklets.items():
        if values:
            sequences.append(torch.stack(values, dim=0))
            instance_tokens.append(instance)

    if not sequences:
        return None, None, []

    padded = pad_sequence(sequences, batch_first=True)
    mask = torch.zeros(padded.size(0), padded.size(1), dtype=torch.bool, device=padded.device)
    for idx, sequence in enumerate(sequences):
        mask[idx, : sequence.size(0)] = True
    return padded, mask, instance_tokens


def temporal_forward(padded_tracklets: torch.Tensor | None, mask: torch.Tensor | None, temporal_model: torch.nn.Module):
    if padded_tracklets is None or mask is None:
        return None
    return temporal_model(padded_tracklets, src_key_padding_mask=~mask)


def process_scene(
    scene: dict[str, Any],
    scene_model: torch.nn.Module,
    association_model: torch.nn.Module,
    aggregation_model: torch.nn.Module,
    temporal_model: torch.nn.Module,
    temporal_loss: torch.nn.Module,
    representation_loss: torch.nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any] | None:
    encoded_scene, rep_loss = encode_scene(scene, scene_model, device=device, representation_loss=representation_loss)
    tracklets, associations = build_tracklets(encoded_scene, association_model, aggregation_model)
    padded, mask, instance_tokens = pad_tracklets(tracklets)
    if padded is None or mask is None:
        return None
    temporal_embeddings = temporal_forward(padded, mask, temporal_model)
    temp_loss = temporal_loss(temporal_embeddings, mask)
    return {
        "loss": rep_loss + temp_loss,
        "representation_loss": rep_loss,
        "temporal_loss": temp_loss,
        "encoded_scene": encoded_scene,
        "tracklets": tracklets,
        "associations": associations,
        "padded_tracklets": padded,
        "mask": mask,
        "instance_tokens": instance_tokens,
        "temporal_embeddings": temporal_embeddings,
    }


def evaluate_association_accuracy(associations: list[dict[str, Any]]) -> dict[str, float]:
    correct = 0
    total = 0
    mean_conf = []
    for item in associations:
        association = item["association"].detach().cpu()
        instance_t = item["instance_t"]
        instance_tp1 = item["instance_tp1"]
        pred = association.argmax(dim=1).tolist()
        conf = association.max(dim=1).values.tolist()
        for row, col in enumerate(pred):
            if row >= len(instance_t) or col >= len(instance_tp1):
                continue
            total += 1
            mean_conf.append(conf[row])
            correct += int(instance_t[row] == instance_tp1[col])
    return {
        "association_accuracy": correct / total if total else 0.0,
        "association_pairs": float(total),
        "mean_association_confidence": sum(mean_conf) / len(mean_conf) if mean_conf else 0.0,
    }
