# Data

This project uses scikit-learn's Optical Recognition of Handwritten Digits dataset.

- 1,797 images
- image size: 8 × 8
- 10 classes
- pixel values range from 0 to 16

Source documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

Before training, pixel values are divided by 16 and each sample is reshaped to a `1 × 8 × 8` tensor for PyTorch.
