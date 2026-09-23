# Convolutional Neural Network for Handwritten Digits

## Question

Can a small convolutional network use the spatial structure of 8 × 8 digit images effectively without becoming a large computer-vision project?

## Data

I use scikit-learn's digits dataset with a stratified 75/25 train/test split.

Pixel values are divided by 16 and reshaped to `1 × 8 × 8` tensors.

## Method

The PyTorch model is:

```text
Conv2d 1→16
ReLU
MaxPool
Conv2d 16→32
ReLU
MaxPool
Flatten 128
Linear 128→10
```

Training uses mini-batches of 64, Adam with learning rate 0.003, cross-entropy loss, and 14 epochs.

## Results

The recorded run produced:

| Metric | Result |
|---|---:|
| Accuracy | 0.9600 |
| Macro-F1 | 0.9597 |
| Epochs | 14 |

## Interpretation

Accuracy and macro-F1 are nearly the same, so the model performs consistently across the ten classes in the recorded run.

The result is close to the MLP result. That is not surprising on such a small and clean image dataset. The main value of this project is understanding the convolutional pipeline rather than claiming a large performance gain.

## Limitations

The dataset is small, the network is intentionally compact, and the experiment uses one split. Small numerical differences can also appear across PyTorch and CPU-library versions.

A stronger comparison would run both MLP and CNN models across several seeds with the same evaluation protocol.

## Reproduce

```bash
pip install -r requirements.txt
python src/run_experiment.py
```
