# Web Portfolio Card

## Convolutional Neural Network

**Track:** AI Engineering  
**Difficulty:** ★★★★  
**Dataset:** Optical Recognition of Handwritten Digits  
**Quick description:** Train a compact PyTorch CNN on real handwritten digit images with explicit batching and held-out evaluation.

### Suggested website image gallery

![Cover](assets/01_cover.svg)

![CNN training pipeline](assets/02_data_pipeline.svg)

![CNN architecture](assets/03_data_or_model.svg)

![Training and held-out results](assets/04_evaluation_or_results.svg)

### Suggested portfolio copy
This project trains a compact PyTorch convolutional neural network on real 8×8 handwritten digit images. The model learns 16 then 32 spatial feature maps, compresses them through max-pooling, and maps 128 learned features to 10 digit classes. With Adam optimization over 14 epochs, it reaches 96.0% held-out accuracy and 95.97% macro-F1. The repository includes executable code, empirical metrics, reproducibility documentation, and a scientific-style technical report.
