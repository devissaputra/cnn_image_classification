# Convolutional Neural Network for Handwritten Digits

[![CI](https://github.com/devissaputra/cnn_image_classification/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/cnn_image_classification/actions/workflows/ci.yml)


**Category:** AI Engineering
![Project overview](assets/01_cover.svg)

A compact PyTorch image-classification experiment that asks whether preserving **2D spatial structure** helps beyond a strong flattened linear baseline.

## Data

- 1,797 handwritten digit images
- 8 × 8 grayscale pixels
- 10 classes
- stratified 75/25 train/test split
- seed 42
- pixels scaled to [0, 1]

## Baseline and CNN

![Training pipeline](assets/02_data_pipeline.svg)

### Logistic baseline
The 8×8 image is flattened to 64 values and classified with logistic regression.

### CNN
```text
1 × 8 × 8
  ↓
Conv 1→16 + ReLU + MaxPool
  ↓
Conv 16→32 + ReLU + MaxPool
  ↓
32 × 2 × 2
  ↓
Flatten (128)
  ↓
Linear 128→10
```

The network has **6,090 trainable parameters**.

Training uses Adam, learning rate 0.003, batch size 64, cross-entropy loss, and 14 epochs. The DataLoader uses a seeded generator so batch ordering is reproducible.

## Recorded results

![CNN architecture](assets/03_data_or_model.svg)

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic regression | 0.9622 | 0.9620 |
| CNN | **0.9733** | **0.9729** |

![Held-out evaluation](assets/04_evaluation_or_results.svg)

Unlike the MLP project, the spatial inductive bias helps here: the CNN improves on the flattened linear baseline. The point is not that CNNs always win, but that architecture should match the structure of the data.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

## Test

```bash
pip install pytest
pytest
```

CI uses a one-epoch smoke run plus shape and determinism tests, so the repository checks real behaviour without retraining the full experiment on every commit.

## Why the CNN helps here

The CNN wins by a modest margin, which is exactly the kind of result I wanted to test rather than assume. Keeping the image in two dimensions gives the model access to local spatial structure that the flattened logistic baseline cannot represent directly.

The gain is real on this split, but it is not large enough to justify broad claims about CNN superiority.

## What I would test next

The next version would use repeated seeds, a validation set for model selection, augmentation, calibration, robustness checks, and a larger image dataset. I would also compare parameter efficiency so the accuracy gain is weighed against the added computational cost.
