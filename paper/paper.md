# Convolutional Neural Network: Scientific-Style Technical Report

**Status:** reproducible portfolio report, not peer reviewed.  
**Difficulty:** ★★★★  
**Dataset:** Optical Recognition of Handwritten Digits dataset

## Abstract
This project studies a concrete AI Engineering problem using a real public dataset and a fully inspectable pipeline. The project focuses on PyTorch, CNN, computer vision, training loop. Its central engineering goal is to make data preparation, model fitting, evaluation, and limitations reproducible rather than treating the model as a black box.

## 1. Research objective
Train a compact PyTorch CNN on real handwritten digit images with explicit batching and evaluation.

## 2. Data
The dataset is **Optical Recognition of Handwritten Digits dataset**. Provenance and the original reference are documented in [`DATA.md`](../DATA.md).

## 3. Method
The implemented pipeline is:
1. Load digits
2. Normalize images
3. Batch tensors
4. CNN
5. Confusion analysis

## 4. Evaluation
**Primary metric(s):** Accuracy / macro F1.  
**Validation design:** stratified hold-out.  
The experiment saves machine-readable metrics and visual diagnostics so claims can be traced to an executable run.

## 5. Results
Generated metrics:
```json
{
  "accuracy": 0.96,
  "macro_f1": 0.9596695483330727,
  "epochs": 14
}
```

## 6. Limitations and validity
Key concern: small spatial resolution. Benchmark performance on one dataset does not imply universal performance. The project is intended to demonstrate research engineering discipline and to provide a base for stronger comparative studies.

## 7. Reproducibility
Run `python src/run_experiment.py` from the repository root after installing `requirements.txt`.

## 8. Next research extension
Add repeated cross-validation or temporal/external validation, stronger baselines, hyperparameter sensitivity, confidence intervals, and a domain-specific error analysis.

## References
- Dataset/reference page: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
