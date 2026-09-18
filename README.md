# Ray-based sampling for transition-line extraction in DQD stability diagrams

Recovering the charge-transition lines of a double quantum dot from a **small
fraction** of the gate-voltage plane, instead of a full raster scan.

Current is measured along a fan of rays from one corner of the (V₁, V₂)
window. A ray crossing a transition line produces a local maximum in the
charge-sensor current, so the peaks along the rays are candidate points *on*
the lines. A small U-Net turns those sparse traces into a dense transition-line
map over the whole plane, and we ask one question at several measurement
budgets:

> how little of the grid can we measure and still recover every line?

E. Alizadeh Kashtiban, T. Fujita, A. Oiwa — Osaka University (SANKEN).

## Getting started

```bash
pip install torch numpy matplotlib scipy pillow qarray python-pptx
python scripts/run_0_full_sweep.py          # build, train, score, compare
```

`run_0` is the whole study in one command. To do it step by step — which is
the normal way to work — see **[`scripts/README.md`](scripts/README.md)**;
that is the document to read first.

```
python scripts/run_1_generate_dataset.py    simulate the devices, split them
python scripts/run_2_train_model.py         train the U-Net
python scripts/run_3_evaluate_model.py      score it on held-out devices
python scripts/run_4_compare_configs.py     every budget side by side
python scripts/run_9_make_slides.py         build the presentation deck
```

One more, outside the sequence — the geometry benchmark, in a single file:

```
python scripts/benchmark.py                 four ways to spend one budget
```

## Layout

```
scripts/          the programs you run — a settings block and a few lines each
src/dqd/          the library.  Nothing here has a command line.
    config/       parameter space, paths, figure house style
    simulation/   the device model
    ml/           the ray cutting, the U-Net, training, metrics
    study/        the four stages, the sweep, and the figure gallery
    visualization/ drawing measurements and predictions over a diagram
training_data/    one folder per configuration (+ the cached device pool)
results/          cross-configuration tables, figures and the deck
```

## What makes the numbers trustworthy

* **Devices are simulated once and shared.** Changing the number of rays
  changes how a device is measured, never which device it is — so a
  comparison across budgets is a comparison of the measurement and nothing
  else.
* **The split is made on the device, not the image**, once, and stored with
  the device pool. No device can land on both sides in any budget.
* **Every simulated diagram enters the dataset.** The capacitances are drawn
  at random and nothing is filtered — there is no acceptance test.
* **Training constants are constants**, not settings — they live once, in
  `src/dqd/ml/grid_train.py`, so every cell of a sweep is trained
  identically.
* **Nothing is tuned on the test devices.** The binarisation threshold is
  chosen on a validation split carved out of the training devices and stored
  inside the checkpoint.

## Reporting

Scores are reported as **F1@τ**: a predicted pixel counts as correct when a
true line pixel lies within τ pixels. F1@1 is the headline, and F1@0 … F1@3
are all reported, so how much of the error is sub-pixel placement rather than
a missed line is visible instead of assumed.
