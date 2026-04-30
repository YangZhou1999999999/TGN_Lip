# TGN_Lip

Utilizing Lipschitz Bound to Mitigate Temporal Discontinuity in Dynamic Graph Neural Networks.

## Overview

This repository contains an implementation of a Lipschitz-bound-based regularization framework for dynamic graph neural networks (GNNs), designed to mitigate temporal discontinuity during training. The framework addresses the instability problems that occur when memory-based dynamic GNNs are trained with large batches or under rapidly evolving interaction patterns.

The key contribution is a novel regularization approach that preserves temporal coherence in dynamic representations by constraining representation changes across time intervals using Lipschitz bounds.

## Motivation

Memory-based dynamic GNNs can suffer from temporal discontinuity and representation drift when:
- Training with large batch sizes
- Processing rapidly evolving graphs
- Experiencing significant structural changes over time
- Operating under non-stationary conditions

This project addresses these challenges through principled regularization inspired by Lipschitz bounds.

## Key Features

The main idea is to constrain representation changes across time by introducing a Lipschitz-bound-inspired regularization term. This helps the model:

- **Stabilize** memory updates across temporal steps
- **Preserve** temporal consistency and smoothness
- **Reduce** representation drift under dynamic graph evolution
- **Remain robust** under large-batch training scenarios
- **Maintain** numerical stability during backpropagation

## Repository Structure

```
TGN_Lip/
├── train_supervised.py           # Supervised training script
├── train_self_supervised.py      # Self-supervised training script
├── model/                        # Core model definitions
├── modules/                      # Model components and building blocks
├── evaluation/                   # Evaluation utilities and metrics
├── utils/                        # Helper functions and utilities
└── README.md
```

## Requirements

- Python 3.7+
- PyTorch >= 1.9
- NumPy
- SciPy
- tqdm
- scikit-learn (for evaluation metrics)

## Training

### Supervised Training

```bash
python train_supervised.py \
    --modify_coefficient 0.1 \
    --modify_regularization 0.01
```

### Self-Supervised Training

```bash
python train_self_supervised.py \
    --modify_coefficient 0.1 \
    --modify_regularization 0.01
```


## Results

[Add your experimental results, performance metrics, and comparisons with baselines here]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhou2025batch,
  title={Batch-agnostic dynamic GNN for mitigating temporal discontinuity},
  author={Zhou, Yang and Ren, Xiaoning and He, Jing},
  journal={Neurocomputing},
  pages={131447},
  year={2025},
  publisher={Elsevier}
}
```

## References

- [Temporal Graph Networks for Deep Learning on Dynamic Graphs (ICLR 2020)](https://arxiv.org/abs/2006.10637)
- [Lipschitz Neural Networks](https://arxiv.org/abs/1805.05957)
- Related work on dynamic representation learning and temporal stability in evolving graphs

## License

This project is open source. Please see the LICENSE file for details (if applicable).

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions or issues, please open an issue on GitHub or contact the repository owner.
