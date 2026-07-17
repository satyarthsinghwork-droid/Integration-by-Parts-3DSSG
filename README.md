# Integration by Parts on 3DSSG: Spatial-Gated 384-D Epoch-100 Archive

This branch is a frozen, reproducible archive of the completed
`official_static_spatial_gated_384_v4` experiment. It preserves the exact
100-epoch notebook, pipeline source, official OCRL/3DSSG split files,
pretrained PointNet weights, training history, final checkpoint, and official
evaluation output before the next architectural approach is developed.

The archived branch is `spatial-gated-384-v4-epoch100`. The repository's
`main` branch is intentionally left unchanged.

## Experiment Summary

- Protocol: official OCRL/3DSSG subset with 160 object classes and 26 positive
  multi-label relation classes.
- Split: 1,178 training scan IDs and 157 validation scan IDs (1,335 total).
- Static database: 3,852 training graph entries and 548 validation graph
  entries (4,400 total).
- Modalities: global CLIP text, official pretrained PointNet LiDAR tokens, and
  one valid CLIP RGB crop per object.
- Model: 384-D representation, seven latent RGB/LiDAR components, global text,
  and an 8-nearest-neighbor spatial-gated graph context layer.
- Optimization: 100 epochs, seed 42, AdamW at `1e-4`, cosine learning-rate
  decay, and edge-loss warmup from 0.20 to 1.00 over epochs 1-10.
- Static losses: temporal and dynamic terms are disabled for 3DSSG; node,
  edge, representation, part-alignment, and diversity objectives remain active
  through the implementation in `mtp_pipeline/`.

## Official Results

All values below are percentages. Published SGPN, SGFN, VL-SAT, and OCRL rows
are transcribed from the [OCRL paper](https://arxiv.org/abs/2510.04714)'s
official 3DSSG comparison tables. The two
Integration-by-Parts rows were produced by the evaluator included in this
branch on the same 160-object/26-relation label protocol. External rows were
not independently rerun in this repository.

| Method | Obj R@1 | Obj R@5 | Obj mR@1 | Obj mR@5 | Pred R@1 | Pred R@3 | Pred mR@1 | Pred mR@3 | Trip R@50 | Trip R@100 | Trip mR@50 | Trip mR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGPN | 49.46 | 73.99 | 10.86 | 27.54 | 86.92 | 94.76 | 32.01 | 55.22 | 85.38 | 88.59 | 41.52 | 51.92 |
| SGFN | 53.36 | 76.88 | 21.24 | 46.68 | 89.00 | 97.71 | 41.89 | 70.82 | 88.59 | 91.14 | 58.37 | 67.61 |
| VL-SAT | 55.93 | 78.06 | 21.41 | 46.14 | 89.81 | 98.46 | 54.03 | 77.67 | 89.35 | 92.20 | 65.09 | 73.59 |
| OCRL | 59.53 | 81.20 | 22.55 | 48.10 | 91.27 | 98.48 | 56.32 | 76.26 | 91.40 | 93.80 | 65.31 | 74.54 |
| Integration by Parts, initial 50 epochs | 51.43 | 71.00 | 15.40 | 32.72 | 83.76 | 93.25 | 36.42 | 57.99 | 84.64 | 86.46 | 43.05 | 53.14 |
| **Integration by Parts, spatial-gated v4, 100 epochs** | **55.04** | **75.14** | **18.83** | **40.49** | **76.26** | **92.09** | **35.17** | **57.61** | **86.83** | **88.70** | **49.17** | **59.40** |

The v4 run improves object, triplet, and most SGCls metrics over the initial
implementation, but predicate R@1/R@3 decreases. It should therefore be
reported as a completed ablation, not as a claim that every published baseline
has been beaten.

### Scene-Graph Metrics

`GC` means with graph constraints; `No GC` means without graph constraints.

| Run | Task | Setting | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Initial, epoch 50 | SGCls | GC | 25.17 | 26.53 | 26.56 | 18.93 | 20.18 | 20.18 |
| Initial, epoch 50 | SGCls | No GC | 26.33 | 30.13 | 32.20 | 20.62 | 24.75 | 26.50 |
| Initial, epoch 50 | PredCls | GC | 54.14 | 63.39 | 64.20 | 39.87 | 46.72 | 47.09 |
| Initial, epoch 50 | PredCls | No GC | 54.23 | 72.82 | 84.31 | 41.06 | 58.35 | 69.62 |
| **Spatial-gated v4, epoch 100** | **SGCls** | **GC** | **28.82** | **30.16** | **30.24** | **21.45** | **23.25** | **23.27** |
| **Spatial-gated v4, epoch 100** | **SGCls** | **No GC** | **29.34** | **33.68** | **36.94** | **22.96** | **28.60** | **32.69** |
| **Spatial-gated v4, epoch 100** | **PredCls** | **GC** | **59.95** | **67.37** | **67.87** | **43.11** | **48.82** | **48.97** |
| **Spatial-gated v4, epoch 100** | **PredCls** | **No GC** | **59.63** | **77.33** | **87.84** | **43.89** | **58.93** | **68.94** |

Raw, unrounded values are preserved in:

- `results/baseline_epoch50_evaluation.json`
- `results/spatial_gated_384_v4_epoch100/checkpoints/evaluation_official_integration_by_parts_3rscan_epoch_100.json`

## Archived Files

- `run_3rscan_pipeline.ipynb`: prepares CLIP text, pretrained PointNet LiDAR,
  RGB object crops, and the official static database.
- `run_3rscan_spatial_gated_384_v4.ipynb`: exact fresh/resume/evaluate notebook
  for the 100-epoch experiment.
- `mtp_pipeline/`: all preprocessing, model, loss, graph, training, splitting,
  resume, and official evaluation code used by the run.
- `official_splits/`: 160/26 label space and official train/validation files.
- `reference/pretrained/obj_enc.pth`: pretrained PointNet object encoder used to
  produce LiDAR tokens.
- `results/spatial_gated_384_v4_epoch100/training_history_3rscan.json`: all 100
  completed epoch records.
- `results/spatial_gated_384_v4_epoch100/checkpoints/integration_by_parts_3rscan_epoch_100.pt`:
  final trained model, stored with Git LFS.
- `results/spatial_gated_384_v4_epoch100/reproducibility_manifest.json`: exact
  configuration, expected data counts, environment, and SHA-256 checksums.

The raw 3RScan/3DSSG datasets, extracted modality caches, and generated 10 GB
static database are not committed. They must be regenerated from the licensed
datasets with the included notebook and source.

## Reproduce the Run

### 1. Clone the frozen branch and fetch the checkpoint

```powershell
git clone --branch spatial-gated-384-v4-epoch100 https://github.com/satyarthsinghwork-droid/Integration-by-Parts-3DSSG.git
cd Integration-by-Parts-3DSSG
git lfs pull
```

Verify that the epoch-100 checkpoint is approximately 159 MB. A small text
pointer means Git LFS has not fetched the model yet.

### 2. Create the environment

The completed run used Python 3.11.3, PyTorch 2.6.0 with CUDA 12.4, NumPy
2.4.6, tqdm 4.67.3, Transformers 5.12.1, OpenCV 4.11.0, Pillow 11.3.0, and
pandas 2.3.2. Install the CUDA-compatible PyTorch build for the target machine,
then install the remaining packages:

```powershell
python -m pip install -r requirements.txt
```

### 3. Place the raw data

The archived notebooks default to the original experiment layout:

```text
D:\MTP_Project\3RScan
D:\MTP_Project\3DSSG\objects.json
D:\MTP_Project\3DSSG\relationships.json
D:\MTP_Project\MTP_Pipeline_3RScan
```

If the repository is elsewhere, update `ProjectPaths` in
`mtp_pipeline/config.py` and the `project_root`/dataset paths in the first
setup cells. Do not change the official files in `official_splits/`.

### 4. Rebuild embeddings and the static database

Restart the kernel and run the preparation sections of
`run_3rscan_pipeline.ipynb` in order:

1. setup and official paths;
2. global CLIP text embeddings;
3. official pretrained PointNet LiDAR tokens;
4. CLIP RGB object crops;
5. official static database construction.

The expected output is `pipeline_outputs/3rscan_official_static_database_v2`
with 4,400 `.pt` graph entries and a `label_space.json` file. Preparation is
resumable and should reuse completed modality files.

### 5. Train or resume the exact experiment

Open `run_3rscan_spatial_gated_384_v4.ipynb` and run its setup cell. Use the
fresh-training cell for a new run or the resume cell after an interruption.
The total target remains 100 epochs; resume does not add another 100 epochs.

### 6. Evaluate the archived epoch-100 checkpoint

With the rebuilt database present, this evaluates the saved model directly:

```powershell
python -m mtp_pipeline.evaluate_3dssg `
  --checkpoint results\spatial_gated_384_v4_epoch100\checkpoints\integration_by_parts_3rscan_epoch_100.pt `
  --database D:\MTP_Project\MTP_Pipeline_3RScan\pipeline_outputs\3rscan_official_static_database_v2 `
  --reference-root D:\MTP_Project\MTP_Pipeline_3RScan\pipeline_data `
  --output-root D:\MTP_Project\MTP_Pipeline_3RScan\pipeline_outputs\official_static_spatial_gated_384_v4 `
  --train-scans official_splits\train_scans.txt `
  --val-scans official_splits\validation_scans.txt `
  --mode static --device cuda
```

The expected rounded output is the spatial-gated epoch-100 row and scene-graph
table shown above. Exact artifact checksums are in the reproducibility manifest.
Fresh training can show small hardware-dependent numerical variation even with
the same seed; evaluating the archived checkpoint should reproduce the metrics.
