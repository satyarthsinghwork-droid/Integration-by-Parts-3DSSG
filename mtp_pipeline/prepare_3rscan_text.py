from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .config import ProjectPaths


def prepare_3rscan_text_embeddings(
    objects_json: Path, 
    output_dir: Path, 
    model_name: str, 
    device: str | None
) -> None:
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device_obj).eval()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(objects_json, "r") as f:
        data = json.load(f)
    
    scans = data.get("scans", [])
    
    # Pre-compute category embeddings to save time
    categories = set()
    for scan in scans:
        for obj in scan.get("objects", []):
            categories.add(obj["label"])
    
    categories = sorted(categories)
    category_embeddings = {}
    
    for category in tqdm(categories, desc="Encoding 3DSSG category text"):
        # Base paper (2510) uses the prompt: "A point cloud of {object}."
        prompt = f"A point cloud of {category}."
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device_obj)
        with torch.no_grad():
            category_embeddings[category] = text_encoder(**inputs).pooler_output.squeeze(0).cpu()

    # Save individual tokens per object
    for scan in tqdm(scans, desc="Saving text embeddings for 3RScan objects"):
        scan_id = scan["scan"]
        for obj in scan.get("objects", []):
            obj_id = str(obj["id"])
            category_name = obj["label"]
            
            torch.save(
                {
                    "dataset_version": "3rscan",
                    "scene_token": scan_id,
                    "annotation_token": f"{scan_id}_{obj_id}",
                    "instance_token": obj_id,
                    "category_name": category_name,
                    "text_features": category_embeddings[category_name],
                },
                output_dir / f"{scan_id}_{obj_id}.pt",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CLIP text embeddings for 3DSSG objects.")
    parser.add_argument("--objects-json", type=Path, default=Path(r"D:\MTP_Project\3DSSG\objects.json"))
    parser.add_argument("--output-dir", type=Path, default=ProjectPaths().reference_root / "3rscan_text_embeddings")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_3rscan_text_embeddings(args.objects_json, args.output_dir, args.model_name, args.device)


if __name__ == "__main__":
    main()
