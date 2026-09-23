# Reproducing the Experiment

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The script sets the PyTorch seed to 42 and uses one CPU thread. The train/test split is stratified with `random_state=42`.

Training uses batch size 64, Adam with learning rate 0.003, cross-entropy loss, and 14 epochs.

The recorded metrics are saved to `results/metrics.json`.

PyTorch and low-level math-library versions can produce small numerical differences even with the same seed. If exact comparison matters, record the Python, PyTorch, NumPy, and scikit-learn versions.
