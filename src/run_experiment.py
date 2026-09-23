from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SEED = 42


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Linear(32 * 2 * 2, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def load_split(seed: int = SEED):
    X, y = load_digits(return_X_y=True)
    images = (X.astype("float32") / 16.0).reshape(-1, 1, 8, 8)
    return train_test_split(
        images,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )


def classification_metrics(y_true, prediction):
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
    }


def train_cnn(X_train, y_train, epochs: int = 14, seed: int = SEED):
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )

    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    model.train()
    for _ in range(epochs):
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))

    return model, losses


def run_experiment(
    results_dir: str | Path = "results",
    seed: int = SEED,
    epochs: int = 14,
    make_plots: bool = True,
):
    X_train, X_test, y_train, y_test = load_split(seed)

    baseline = LogisticRegression(max_iter=3000, random_state=seed)
    baseline.fit(X_train.reshape(len(X_train), -1), y_train)
    baseline_prediction = baseline.predict(X_test.reshape(len(X_test), -1))

    model, losses = train_cnn(X_train, y_train, epochs=epochs, seed=seed)
    model.eval()
    with torch.no_grad():
        cnn_prediction = (
            model(torch.tensor(X_test))
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

    results = {
        "seed": int(seed),
        "epochs": int(epochs),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "logistic": classification_metrics(y_test, baseline_prediction),
        "cnn": classification_metrics(y_test, cnn_prediction),
    }

    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    if make_plots:
        figures = output / "figures"
        figures.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 5))
        plt.plot(range(1, len(losses) + 1), losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.title("CNN learning curve")
        plt.tight_layout()
        plt.savefig(figures / "cnn_loss.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            cnn_prediction,
            ax=ax,
            colorbar=False,
        )
        ax.set_title("CNN held-out confusion matrix")
        fig.tight_layout()
        fig.savefig(figures / "cnn_confusion_matrix.png", dpi=150)
        plt.close(fig)

    return results


def main() -> None:
    print(json.dumps(run_experiment(), indent=2))


if __name__ == "__main__":
    main()
