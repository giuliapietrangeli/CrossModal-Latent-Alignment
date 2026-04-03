# Cross-Modal Latent Alignment: Text-to-Vision Embedding Mapping

This repository provides a comprehensive pipeline for aligning disparate latent spaces in cross-modal retrieval tasks. The primary objective is to map 1024-dimensional text embeddings (RoBERTa) into a 1536-dimensional target vision latent space (DINOv2).

The framework implements multiple adapter architectures and optimization strategies to ensure high-fidelity mapping and robust retrieval performance.

## Technical Methodology

The project explores three distinct neural architectures designed to handle the complexity of cross-modal translation:

### 1. RMLPA (Residual Bottleneck Adapter)
A residual neural network utilizing a bottleneck design (ratio D_in/4) to learn non-linear mappings while preserving gradient flow through shortcut connections.

### 2. LatentMapper (Linear Projection)
A streamlined linear layer that leverages Orthogonal Procrustes Analysis for weight initialization. By computing the optimal rotation between source and target distributions on the training set, this model achieves rapid convergence and high baseline accuracy.

### 3. Stitcher (Parallel Path Adapter)
A hybrid architecture that fuses the outputs of a direct linear projection and a deep, non-linear MLP path, allowing the model to capture both global alignment and local semantic nuances.

## Optimization and Loss Functions

Models are trained using state-of-the-art loss functions tailored for retrieval:
* **Symmetric InfoNCE Loss**: Implemented with a learnable temperature parameter to maximize mutual information between modalities.
* **Triplet Margin Loss**: Utilized with in-batch hard negative mining (margin m=0.2) to refine the decision boundaries in the embedding space.

## Repository Structure

* `models/`: Core adapter architectures and model-specific logic (e.g., `utils_mlp.py`, `utils_rmlpa.py`, `utils_stitcher.py`).
* `utils/`: Support utilities, including hyperparameter tuning (`utils_tuning.py`) and shared validation metrics.
* `checkpoints/`: Optimized model weights (.pth) for the RMLPA, LatentMapper, and Stitcher architectures.
* `Technical_Report.pdf`: Documentation detailing the experimental setup, hyperparameter tuning, and comparative analysis.
* `main_pipeline.ipynb`: The primary execution environment for training, validation, and inference.

## Performance and Results

The implemented ensembling strategy—calculated via a weighted average of L2-normalized outputs—reaches an internal Mean Reciprocal Rank (MRR) of 0.88. This demonstrates the effectiveness of combining diverse loss functions and architectures to reduce uncorrelated errors.

## Requirements

* Python 3.8+
* PyTorch
* Scikit-learn
* SciPy
* Pandas/NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
*Developed as part of the AML Challenge 2025.*
