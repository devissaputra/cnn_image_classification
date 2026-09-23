# Convolutional Neural Network for Handwritten Digits

![Project overview](assets/01_cover.svg)

I built this project to compare a small convolutional network with the fully connected MLP from the previous project.

The main difference is that the CNN keeps the two-dimensional image structure intact while it learns local patterns such as edges and strokes.

## Data

I use scikit-learn's handwritten digits dataset.

- 1,797 grayscale images
- image size: 8 × 8
- 10 classes, digits 0 through 9
- 75% training, 25% test
- stratified split with random state 42

Pixel values are divided by 16 so the inputs are roughly in the 0 to 1 range.

## How the experiment works

![Training pipeline](assets/02_data_pipeline.svg)

The network is implemented in PyTorch.

```text
1 × 8 × 8 input
      ↓
Conv2d: 1 → 16
ReLU
MaxPool2d
      ↓
Conv2d: 16 → 32
ReLU
MaxPool2d
      ↓
32 × 2 × 2
      ↓
Flatten to 128 values
      ↓
Linear: 128 → 10
```

Training uses:

- batch size 64;
- Adam optimizer;
- learning rate 0.003;
- cross-entropy loss;
- 14 epochs.

## Model structure

![CNN architecture](assets/03_data_or_model.svg)

The first convolution learns 16 feature maps. After pooling, the second convolution expands that to 32 feature maps. The final pooled representation contains 128 values, which are passed to a linear layer for the ten digit classes.

## Results

![Training and evaluation](assets/04_evaluation_or_results.svg)

The recorded run produced:

| Metric | Result |
|---|---:|
| Accuracy | 0.9600 |
| Macro-F1 | 0.9597 |
| Epochs | 14 |

Macro-F1 is close to accuracy, so the model is not getting its score from only a few classes.

Because PyTorch kernels and package versions can affect small numerical details, I treat these as the results of the recorded run rather than a promise that every machine will reproduce the final decimals exactly.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

On Windows, use `.venv\Scripts\activate`.

## Repository notes

- [DATA.md](DATA.md) explains the dataset.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) records the settings that matter for reruns.
- [paper/paper.md](paper/paper.md) contains the longer write-up.
