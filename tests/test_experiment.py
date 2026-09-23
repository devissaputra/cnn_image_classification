from pathlib import Path

import torch

from src.run_experiment import CNN, load_split, run_experiment


def test_cnn_output_shape():
    model = CNN()
    output = model(torch.zeros(4, 1, 8, 8))
    assert output.shape == (4, 10)


def test_split_is_deterministic():
    first = load_split()
    second = load_split()
    for left, right in zip(first, second):
        assert (left == right).all()


def test_one_epoch_smoke_run(tmp_path):
    result = run_experiment(
        tmp_path,
        epochs=1,
        make_plots=False,
    )
    assert result["trainable_parameters"] == 6090
    assert 0.0 <= result["cnn"]["accuracy"] <= 1.0
    assert 0.0 <= result["logistic"]["accuracy"] <= 1.0


def test_repository_structure():
    root = Path(__file__).resolve().parents[1]
    assert (root / ".github/workflows/ci.yml").exists()
    assert (root / "paper/paper.md").exists()
