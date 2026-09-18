# The programs you run

Everything in this folder is a program **you** run: a settings block at the
top, then a few lines that call the library under `src/dqd/`. Nothing in
`src/` has a command line.

## The sequence — four steps, one measurement budget at a time

```
python scripts/run_1_generate_dataset.py     make the devices + split them
python scripts/run_2_train_model.py          train the U-Net
python scripts/run_3_evaluate_model.py       score it on the held-out devices
python scripts/run_4_compare_configs.py      put every budget side by side
```

Step 1 builds **one** configuration. Steps 2–4 take a **list** of
configuration folder names, so a whole sweep is one command each:

```python
CONFIG_NAMES = ["3_rays_40_points_500_samples",
                "8_rays_40_points_500_samples"]
CONFIG_NAMES = "ALL"          # every folder in training_data/
```

The list is an instruction about order too — the table and figures come out
in the order you wrote them. A name that does not exist is an error listing
the ones that do, rather than being silently skipped. `run_3` finishes by
comparing what you listed (`COMPARE_AFTER = True`), so for most sweeps you
never run step 4 by hand.

## `run_0` — the lazy button

```
python scripts/run_0_full_sweep.py
```

Give it ray counts, point counts, a training-set size and a test-set size,
and it does steps 1–4 for every combination and prints one table. Every
*other* setting comes from `run_1`'s `CONFIG`, and it calls the same library
functions in the same order, so it cannot disagree with running them by hand.
Restartable: anything already on disk is reused, so adding one more ray count
costs one training run.

## The extras — not part of the sequence

| | |
|---|---|
| `run_5_render_device_figures.py` | redraw the per-device pictures — more devices, 300 dpi — without regenerating data or retraining |
| `run_6_threshold_report.py` | the probability maps, and where the binarisation threshold falls |
| `benchmark.py` | one budget, four scenarios — our fan of rays vs Hernandes-style line cuts vs parallel oblique lines vs a lattice.  Self-contained: measure, train, score, table, figure, all in one file |
| `run_9_make_slides.py` | build the presentation deck from whatever results are on disk |

`_common.py` is shared boilerplate (import path, headless plotting, the
"resolve names → do a thing to each → report" loop). It holds no settings.

## One folder per configuration

A configuration is one measurement budget plus one training-set size, and it
owns a folder named after exactly those numbers:

```
training_data/3_rays_40_points_500_samples/
    config.json          the settings; steps 2-4 read them back, so the
                         programs cannot silently disagree
    train.npz test.npz   the measurements and the answers
    dataset_summary.txt  <- the numbers to quote in the paper
    figures/             per-device pictures
    model/               unet.pt, training_curve.png, model_structure.yaml
    evaluation/          results.txt, metrics.json, per_device.csv, figures/
```

`results/<sweep name>/` is where step 4 puts the cross-configuration table
and figures, e.g. `results/3-5-8-12_rays_40_points_500_samples/`.

Nothing is ever overwritten across configurations, and the second and later
ones cost almost no simulation time: **the simulated devices are stored
once**, in `training_data/_device_pools/`, and reused. The number of rays
changes how a device is measured, never which device it is — which is what
makes the comparison a comparison of measurement and nothing else. The pool
folder name carries a fingerprint of the capacitance intervals, so changing
`src/dqd/config/capacitance_config.py` builds a new pool instead of silently
reusing the old one.

## How train and test are separated

**The thing that is split is the device, not the image.** Capacitance
configurations are drawn first, from one distribution, and each gets an ID.
The IDs are split once and the split is stored **with the device pool**
(`_device_pools/<pool>/device_split.json`), not with the configuration. Every
image and every measurement budget inherits the ID of the device it came
from, so a device's images all land on one side — never both.

Storing it with the pool is what makes a sweep safe: every cell reads the
same stored assignment, so device 37 cannot be a training device in the
3-ray cell and a test device in the 5-ray cell.

`dataset_summary.txt` (written by step 1) reports three things:

1. **Every diagram is an unfiltered capacitance draw.** There is no
   acceptance test: the capacitances are drawn at random and whatever the
   simulator returns enters the pool, honeycomb or not. The file reports how
   many draws were made and how many the simulator itself failed on.
2. **No device contributes to both sets** — checked by ID and by capacitance
   hash on the generated data.
3. **No test device is a near-duplicate of a training one** — the minimum
   Euclidean distance between any train and any test configuration vector in
   normalised parameter space, next to the same statistic computed *within*
   the training set as a yardstick.

## Per-device pictures

The `device_figures` block in `run_1` is a switch per picture, and
`figure_devices` says which devices to draw them for:

```python
figure_devices={"train": "ALL",  "test": "ALL"}      # every device
figure_devices={"train": [1, 2, 5], "test": "NONE"}  # those, and none
```

`"ALL"` writes a lot of files on a big pool — the count and a time estimate
are printed before anything is drawn.

| figure | what it is |
|---|---|
| `charge_sensor` | the coloured charge-sensor image, with colorbar and voltage axes |
| `charge_sensor_gradient` | the same data with the smooth gate background differenced away — **this is where the honeycomb is actually visible** |
| `stability_diagram` | the binary DQD stability diagram (ground truth) |
| `rays` | the rays and their detected peaks, over the sensor image |
| `rays_on_truth` | the same rays over the binary diagram — which lines the measurement went near, and which it missed |
| `measurement` | only the visited pixels: what the network is shown |
| `ray_traces` | the 1-D signal along each ray, peaks marked |
| `panel` | four of them side by side, in one file |
| `summary_total_all_crosses` | ground truth + measured points + every peak as one magenta X — the paper figure |

The first three are properties of the **device** and look the same in every
configuration; the rest depend on the measurement budget, which is why they
live inside the configuration folder.

*On the raw sensor image:* the simulated signal carries a large smooth
background from direct gate-to-sensor cross-talk, and the charge steps ride
on top of it — so the honeycomb is faint in `charge_sensor` and obvious in
`charge_sensor_gradient`. Real experiments subtract the same background for
the same reason. `charge_sensor` is left un-rescaled because a figure with a
colorbar should show the quantity that was actually simulated.

## Where the rules live

| what | file |
|---|---|
| the capacitance parameter space | `src/dqd/config/capacitance_config.py` |
| the train/test split, made once on device IDs | `src/dqd/study/device_split.py` |
| how a device is simulated | `src/dqd/simulation/device_factory.py` |
| how the rays are cut | `src/dqd/ml/ray_peaks.py` |
| the network | `src/dqd/ml/grid_model.py` |
| training constants (lr, batch size, loss, validation split) | `src/dqd/ml/grid_train.py` |
| the four stages themselves | `src/dqd/study/` |
| the per-device figures | `src/dqd/study/device_figures.py` |
| the shared figure house style | `src/dqd/config/figure_style.py` |

Training hyperparameters other than the number of epochs are deliberately
**not** settings in `scripts/`. They must be identical in every configuration
or the comparison between budgets stops meaning anything, so they live once,
in `src/dqd/ml/grid_train.py`.
