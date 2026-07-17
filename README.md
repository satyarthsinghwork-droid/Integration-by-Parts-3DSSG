# Integration by Parts for 3DSSG: Two-Stage Reproducible Run

This branch contains the complete code and verified checkpoint for the two-stage,
part-aware multimodal 3D scene graph experiment on the official OCRL/3DSSG
subset of 3RScan.

The only dataset input required from the user is the raw **3RScan** directory.
The official 160 object classes, 26 positive predicates, train/validation
annotations, scan lists, alignment metadata, and OCRL-pretrained PointNet object
encoder are included in this branch.

## Verified Result

The included best checkpoint was selected at graph fine-tuning epoch 5 on all
548 official validation entries.

| Task | Metric | Result (%) |
|---|---:|---:|
| Object | R@1 / R@5 | 56.16 / 76.44 |
| Object | mR@1 / mR@5 | 18.89 / 40.20 |
| Predicate | R@1 / R@3 | 81.59 / 97.20 |
| Predicate | mR@1 / mR@3 | 52.60 / 78.03 |
| Triplet | R@50 / R@100 | 87.21 / 89.78 |
| Triplet | mR@50 / mR@100 | 58.39 / 69.10 |
| SGCls with graph constraints | R@20 / R@50 / R@100 | 29.87 / 32.06 / 32.34 |
| PredCls with graph constraints | R@20 / R@50 / R@100 | 64.32 / 75.06 / 75.91 |
| SGCls without graph constraints | R@20 / R@50 / R@100 | 29.72 / 34.63 / 37.83 |
| PredCls without graph constraints | R@20 / R@50 / R@100 | 62.61 / 81.04 / 90.22 |

The complete Table 2, Table 3, and Table 10 values are stored in
`results/evaluation_official_graph_best_official_proxy.json`.



| Metric | OCRL | Ours | Difference |
|---|---:|---:|---:|
| Object R@1 | 60.10 | 56.16 | -3.94 |
| Object R@5 | 80.14 | 76.44 | -3.70 |
| Predicate R@1 | 92.41 | 81.59 | -10.82 |
| Predicate R@3 | 97.05 | **97.20** | **+0.15** |
| Triplet R@100 | 94.12 | 89.78 | -4.34 |
| Object mR@1 | 22.82 | 18.89 | -3.93 |
| Predicate mR@3 | 75.01 | **78.03** | **+3.02** |
| SGCls constrained R@50 | 36.98 | 32.06 | -4.92 |
| PredCls constrained R@50 | 86.15 | 75.06 | -11.09 |

## What Is Included

- Static top-3 RGB object views from CLIP ViT-B/32 patch tokens.
- Official OCRL-pretrained PointNet tokens from XYZ, RGB, and surface normals.
- CLIP category-text supervision without dividing text into artificial parts.
- Seven conditioned latent RGB-LiDAR parts with part alignment and diversity.
- Two-stage optimization: object/part pretraining, then graph fine-tuning.
- Hybrid global/spatial graph context with official ordered candidate pairs.
- Official 160/26 multi-label protocol and base-paper-style evaluator.
- Correct masking of unavailable RGB tokens and the official invalid-scan exclusion.

## Installation

Git LFS is required because the final graph checkpoint is about 184 MB.

```bash
git lfs install
git clone --branch two_staged https://github.com/satyarthsinghwork-droid/Integration-by-Parts-3DSSG.git
cd Integration-by-Parts-3DSSG
git lfs pull

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The recorded run used Python 3.11, PyTorch 2.6.0 with CUDA 12.4,
Transformers 5.12.1, and seed 42. A CUDA GPU is strongly recommended.

## Raw Dataset Layout

Point `--scan-root` to the directory containing the 3RScan UUID folders:

```text
3RScan/
  00d42bed-778d-2ac6-86a7-0e0e5f5f5660/
    labels.instances.annotated.v2.ply
    sequence.zip
    ...
  <other scan UUIDs>/
```

No separately downloaded 3DSSG `objects.json` or `relationships.json` is used.
The object manifest is deterministically generated from `official_splits/`.

## Reproduce the Data

From the repository root, run:

```bash
python -m mtp_pipeline.protocol_run_v6 prepare \
  --scan-root /path/to/3RScan \
  --device cuda
```

Preparation is resumable and performs these steps in order:

1. Generate the 1,335-scan, 34,833-object official manifest.
2. Extract CLIP text embeddings.
3. extract sampled RGB views and CLIP patch tokens.
4. Extract aligned pretrained PointNet tokens.
5. Build 4,393 graph entries: 3,845 training and 548 validation.

Each expensive extractor skips files already completed. Individual preparation
steps are also available as `prepare-text`, `prepare-rgb`, `prepare-lidar`, and
`build-database` actions.

Check the prepared counts with:

```bash
python -m mtp_pipeline.protocol_run_v6 verify \
  --scan-root /path/to/3RScan
```

## Evaluate the Included Checkpoint

After data preparation:

```bash
python -m mtp_pipeline.protocol_run_v6 evaluate \
  --scan-root /path/to/3RScan \
  --checkpoint artifacts/checkpoints/graph_best_official_proxy.pt \
  --device cuda
```

This evaluates all 548 validation entries. The output is written under
`pipeline_outputs/official_multimodal_parts_384_v6/stage2_graph/evaluation/`.

## Retrain from Scratch

Run both stages with the recorded configuration:

```bash
python -m mtp_pipeline.protocol_run_v6 train \
  --scan-root /path/to/3RScan \
  --stage1-epochs 20 \
  --stage2-epochs 80 \
  --device cuda
```

Training resumes automatically from each `latest` checkpoint after an
interruption. Add `--fresh` to start a new optimization run in an empty work
directory. For a completely isolated run, pass `--work-root /path/to/new/run`.

The notebook `run_two_stage_3dssg.ipynb` provides the same sequence as separate
cells. `evaluate_two_stage_metrics.ipynb` evaluates the included checkpoint.

GPU kernels and library versions can introduce small numerical differences in a
fresh training run. The included checkpoint provides the exact reported model;
its evaluation is deterministic for the prepared tensors.

## Protocol Notes

- Ground-truth object instances are used.
- All ordered non-self object pairs are relation candidates.
- No-relation is represented by an all-zero vector over 26 positive predicates.
- Seven entries from the OCRL-invalid aligned scan are excluded.
- RGB is an intentional additional inference modality, so this is not an
  input-identical ablation of OCRL even though the split and evaluator match.

## Important Files

- `mtp_pipeline/protocol_run_v6.py`: portable end-to-end command.
- `mtp_pipeline/train_two_stage.py`: Stage 1 and Stage 2 optimization.
- `mtp_pipeline/evaluate_3dssg.py`: official-style metrics.
- `official_splits/`: fixed OCRL/3DSSG annotations and class lists.
- `reference/pretrained/obj_enc.pth`: verified OCRL PointNet weights.
- `artifacts/checkpoints/graph_best_official_proxy.pt`: best graph checkpoint.
- `results/`: exact published metric output and compact CSV table.
