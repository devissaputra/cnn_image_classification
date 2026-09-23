# Convolutional Neural Network for Handwritten Digits

**Focus:** whether spatial inductive bias improves on a flattened linear model.

The project compares logistic regression on flattened 8×8 digit images with a compact two-stage PyTorch CNN. The CNN has 6,090 trainable parameters and uses a deterministic seeded DataLoader.

The linear baseline reaches 0.9622 accuracy and 0.9620 Macro-F1. The CNN improves to 0.9733 accuracy and 0.9729 Macro-F1. Together with the MLP repo, this shows a more useful lesson than “deep learning wins”: architecture should match data structure, and every complex model should face a strong baseline.

The repository includes behavioural tests, CI, reproducibility notes, and generated learning diagnostics.
