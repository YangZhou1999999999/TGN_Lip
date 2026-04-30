# TGN_Lip

Utilizing Lipschitz Bound to Mitigate Temporal Discontinuity in Dynamic Graph Neural Networks.

## Overview

This repository contains the implementation of a Lipschitz-bound-based regularization framework for dynamic graph neural networks (GNNs), designed to mitigate temporal discontinuity during training.

Memory-based dynamic GNNs can become unstable when trained with large batches or under rapidly evolving interaction patterns. This project studies how to preserve temporal coherence in dynamic representations while maintaining scalability and performance.

## Key Idea

The main idea is to constrain representation changes across time by introducing a Lipschitz-bound-inspired regularization term.

This helps the model:
- stabilize memory updates,
- preserve temporal consistency,
- reduce representation drift under dynamic graph evolution,
- remain robust under large-batch training.

## Repository Contents

- `train_supervised.py` — supervised training script
- `train_self_supervised.py` — self-supervised training script
- `model/` — core model definitions
- `modules/` — model components and building blocks
- `evaluation/` — evaluation utilities
- `utils/` — helper functions

## Dataset

This project uses the JODIE benchmark dataset.

Dataset source:
http://snap.stanford.edu/jodie/

## Requirements

Typical dependencies include:
- Python
- PyTorch
- NumPy
- SciPy
- tqdm

Install the required packages before running the code.

## Training

### Supervised Training

```bash
python train_supervised.py
```
### Supervised Training
```bash
python train_self_supervised.py
```
## Parameters
Key parameters include:
- modify_coefficient: coefficient used to modify the attention model dimension d_k
- modify_regularization: coefficient used to control the regularization term

## Research Motivation
This project is related to:

- dynamic representation learning,
- temporal stability in evolving graphs,
- robustness under structural change,
- graph learning under non-stationary conditions.
