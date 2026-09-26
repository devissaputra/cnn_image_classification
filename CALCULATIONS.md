# Calculation guide

## Question and evidence

Does preserving image structure help digit classification?

Handwritten digits represented as small 2D images; a shared held-out split.

**Status:** RECORDED SMALL-SAMPLE BENCHMARK | see validation scope.

## Design

Compare flattened logistic regression with a 6,090-parameter CNN trained for 14 epochs.

## Calculation and interpretation

`Accuracy difference = CNN accuracy - logistic accuracy.`

The recorded difference is about 1.11 percentage points on one split. This is not an uncertainty interval or evidence of broad image-recognition superiority. The MLP repository uses a different preprocessing setup and is not a direct cross-repository benchmark.

## Evidence table

Selected recorded values (units and context shown). Full precision below is for traceability, not a claim of measurement precision.

| Quantity | Value | Unit / meaning | JSON path |
|---|---:|---|---|
| logistic | 0.9622222222222222 | accuracy ↑ | `logistic.accuracy` |
| cnn | 0.9733333333333334 | accuracy ↑ | `cnn.accuracy` |

Source: [results/metrics.json](results/metrics.json). Values resolve directly from this file when figures are regenerated.

This PyTorch experiment compares a compact convolutional network with a flattened logistic-regression baseline on handwritten digits. The committed run reports 0.9733 accuracy for the CNN and 0.9622 for logistic regression after 14 epochs. The small gain illustrates a possible benefit from preserving spatial structure, while the single-split design and lack of an uncertainty estimate limit the strength of the architectural conclusion.

## Verification performed in this review

The existing suite requires unavailable dependencies; no full-suite pass is claimed. The complete data/model experiment was not rerun in this review.

The figure-generation check verifies agreement between the selected source values and SVGs. It does not validate the raw dataset, fitted model, identification assumptions, or external generalization.

```bash
python scripts/build_review_figures.py
python scripts/build_review_figures.py --check
```

## Implementation map

Follow these functions to inspect each transformation. Validation helpers and private functions remain visible in the linked modules.

| Function | Purpose / documented behavior |
|---|---|
| [`load_split`](src/run_experiment.py#L41) | Inspect the explicit implementation and its callers. |
| [`classification_metrics`](src/run_experiment.py#L53) | Inspect the explicit implementation and its callers. |
| [`train_cnn`](src/run_experiment.py#L60) | Inspect the explicit implementation and its callers. |
| [`run_experiment`](src/run_experiment.py#L94) | Inspect the explicit implementation and its callers. |
| [`main`](src/run_experiment.py#L161) | Inspect the explicit implementation and its callers. |
| [`forward`](src/run_experiment.py#L37) | Inspect the explicit implementation and its callers. |

## What remains before a stronger research claim

The recorded difference is about 1.11 percentage points on one split. This is not an uncertainty interval or evidence of broad image-recognition superiority. The MLP repository uses a different preprocessing setup and is not a direct cross-repository benchmark. A successful software test is not validation of a scientific construct. New experiments should state their split unit, comparator, outcome, uncertainty procedure and failure criteria before examining final test results.
