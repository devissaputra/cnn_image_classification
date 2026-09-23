# Convolutional Neural Network for Handwritten Digits

## Abstract

This experiment compares a compact CNN with logistic regression on the scikit-learn handwritten-digits benchmark. The comparison tests whether preserving two-dimensional image structure improves classification relative to a flattened linear model.

## Method

Images are scaled to [0,1] and split 75/25 with stratification and seed 42. The CNN contains two convolution-pooling stages and a linear head, totaling 6,090 trainable parameters. It trains for 14 epochs with Adam and cross-entropy loss. Batch shuffling uses a seeded PyTorch generator.

## Results

Logistic regression reaches 0.9622 accuracy and 0.9620 Macro-F1. The CNN reaches 0.9733 accuracy and 0.9729 Macro-F1.

## Interpretation

The CNN improves on the flattened linear baseline in this run, suggesting that a spatial inductive bias is useful even on this small 8×8 dataset. The result complements the MLP project, where extra nonlinear complexity did not improve on the linear baseline.

## Limitations

The benchmark is very small. Repeated seeds, validation-based tuning, robustness analysis, and larger image datasets are needed for stronger claims.
