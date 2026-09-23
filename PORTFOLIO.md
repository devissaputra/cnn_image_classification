# Portfolio Summary

## Convolutional Neural Network for Handwritten Digits

I train a small PyTorch CNN on the same handwritten digit dataset used in the MLP project. This time the image remains two-dimensional while the model learns local feature maps.

### Images

![Project overview](assets/01_cover.svg)

![Training pipeline](assets/02_data_pipeline.svg)

![CNN architecture](assets/03_data_or_model.svg)

![Training and evaluation](assets/04_evaluation_or_results.svg)

**Architecture:** 1×8×8 → Conv16 → Pool → Conv32 → Pool → 128 features → 10 classes.

**Key result:** 0.9600 accuracy and 0.9597 macro-F1 in the recorded run.
