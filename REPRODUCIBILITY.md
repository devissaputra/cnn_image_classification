# Reproducing the experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The data split uses seed 42. PyTorch uses seed 42 and one CPU thread. The training DataLoader receives its own seeded generator, which stabilizes shuffled batch order.

The full recorded run uses 14 epochs, Adam at learning rate 0.003, batch size 64, and cross-entropy loss.

Outputs are written to `results/metrics.json` and `results/figures/`.

CI intentionally runs a shorter smoke experiment through the test suite instead of retraining all 14 epochs on every push. Exact floating-point values can still vary slightly across PyTorch builds.
