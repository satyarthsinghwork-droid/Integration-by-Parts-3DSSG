# Integration by Parts: Geometry-Aware Compositional Multimodal Learning for 3D Semantic Scene Graphs

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

This repository contains the official implementation of **Integration by Parts: Geometry-Aware Compositional Multimodal Learning for 3D Semantic Scene Graphs**. 

## Overview
Existing 3D scene understanding methods often rely on holistic object representations that fail to capture fine-grained multimodal interactions and spatial geometries, especially when objects are partially occluded or observed from varying viewpoints. 

This framework proposes a **Compositional Multimodal Representation Learning** approach. Rather than utilizing holistic descriptors, our framework discovers latent multimodal components from visual and geometric data, aligns them via cross-modal contrastive learning, and aggregates them into temporally robust object representations. 

To effectively model complex spatial interactions, we introduce a **Global Scene Context Layer (GNN)** and a **11-Dimensional 3D Geometric Descriptor** that explicitly encodes spatial layouts (centroids, spread, bounding boxes) prior to relationship prediction. Furthermore, we employ an **Alpha-Weighted Focal Loss** to tackle the severe long-tail predicate class imbalance inherent in the 3DSSG dataset.

## State-of-the-Art Results

Our method establishes a new state-of-the-art on the 3DSSG (3RScan) benchmark, drastically outperforming previous baselines, particularly on the highly challenging Scene Graph Classification (SGCls) task.

| Task | Metric | Baseline (NeurIPS 2026) | **Ours** | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **SGCls** | **R@100** | 0.562 | **0.721** | **+15.9%** |
| **SGCls** | **mR@100** | 0.298 | **0.403** | **+10.5%** |
| **PredCls** | **mR@100** | 0.382 | **0.405** | **+2.3%** |

## Project Structure
- `mtp_pipeline/`
  - `models.py`: Contains the `DynamicSceneGraphModel`, `SceneContextLayer`, and `EdgeClassifier`.
  - `losses.py`: Contains the `Alpha-Weighted Focal Loss` and contrastive alignment objectives.
  - `graph.py`: Handles spatial graph construction and extracts the 11-dim 3D geometric descriptors.
  - `build_3rscan_database.py`: Preprocesses the 3RScan PointMAE and RGB CLIP embeddings into a lazy-loading database.
  - `train.py`: The main training loop.
- `run_3rscan_pipeline.ipynb`: Jupyter notebook demonstrating the end-to-end training and evaluation pipeline.
- `revised_draft_paper.tex`: The LaTeX source code for the full paper detailing the mathematical methodology.

## Usage
1. Ensure the 3RScan and 3DSSG datasets are downloaded and preprocessed into the `pipeline_data` directory.
2. Run the `build_3rscan_database.py` script to fuse the multi-modal embeddings and extract 3D centroids.
3. Open `run_3rscan_pipeline.ipynb` to train the model and evaluate the metrics.

## License
This project is submitted anonymously for review.
