"""
sweep.py — stages 1-4 for a whole grid of measurement budgets.

This is the machinery behind scripts/run_0_full_sweep.py.  It calls exactly
the same library functions the four run_* programs do, in the same order, so
a sweep cannot produce a different answer than running them by hand.

    configs(template, rays, points, ...)   one StudyConfig per cell
    plan(configs, skip_existing)           print what will be built or reused
    run(...)                               do it, then print the final table
"""
import os
import shutil
import time
from dataclasses import replace

from . import comparison, dataset, evaluation, training
from ..config import log, paths
from ..ml.grid_train import training_description
from ..ml.model_report import save_model_report, to_yaml

_RULE = "=" * 74


def configs(template, rays, points, n_train, n_test, epochs, figure_devices):
    """One StudyConfig per cell, all identical except the measurement budget.

    Everything that is not swept comes from `template` (run_1's CONFIG), so
    run_0 and run_1 cannot drift apart.
    """
    return [replace(template,
                    n_rays=r, n_points=p,
                    n_train=n_train, n_test=n_test,
                    epochs=epochs,
                    figure_devices=dict(figure_devices))
            for p in points for r in rays]


def run_dir(cfgs):
    """
    The one folder this sweep owns, under results/ :

        results/4-7-8_rays_40_points_150_samples/

    Named by the same rule the comparison table uses, so the folder holding
    the run and the folder holding its table are the same folder.  Everything
    the run produces - the datasets, the models, the evaluations, the
    comparison and the model/hyperparameter report - goes inside it.
    """
    rows = [{"configuration": c.name, "n_rays": c.n_rays,
             "n_points": c.n_points, "n_train": c.n_train,
             "split_seed": c.split_seed, "resolution": c.resolution}
            for c in cfgs]
    return os.path.join(paths.RESULTS, comparison.sweep_name(rows))


def wipe(results=True, training_data=False):
    """
    Empty the output folders before a run, so what is on disk afterwards came
    from THIS run and nothing else.  Off by default everywhere except run_0's
    FRESH_START switch - deleting results is not something to do by accident.

    `training_data` also throws away the simulated DEVICE POOLS, which is the
    expensive part: the next run re-simulates every device from scratch.
    """
    targets = []
    if results:
        targets.append(paths.RESULTS)
    if training_data:
        targets.append(paths.TRAINING_DATA)
    for folder in targets:
        if os.path.isdir(folder):
            log.warn(f"FRESH START - deleting {os.path.abspath(folder)}")
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)


def _describe_run(out_dir, template, cfgs, rays, points, n_train, n_test,
                  epochs):
    """
    Write, into the run folder, what the model IS and what it was run with:

        model_structure.yaml / .json   the network, layer by layer
        hyperparameters.yaml           every setting of this run

    The network is fully convolutional and therefore identical in every cell
    of the sweep, so one report describes all of them - which is also what
    makes the cells comparable.
    """
    os.makedirs(out_dir, exist_ok=True)
    # The swept knobs are listed once, under "sweep", with their real values;
    # leaving the template's defaults for them in "shared_settings" too would
    # put two different numbers for n_rays in one file.
    shared = template.to_dict()
    for key in ("name", "n_rays", "n_points", "n_train", "n_test", "epochs"):
        shared.pop(key, None)
    settings = {
        "_about": ("Every setting this run was started with.  The sweep "
                   "settings come from scripts/run_0_full_sweep.py; "
                   "everything else is run_1_generate_dataset.py's CONFIG, "
                   "shared by every cell."),
        "sweep": {"rays": list(rays), "points": list(points),
                  "n_train": n_train, "n_test": n_test, "epochs": epochs,
                  "configurations": [c.name for c in cfgs]},
        "shared_settings": shared,
        "training": training_description(template.train_seed),
        "folders": {"run": os.path.abspath(out_dir),
                    "device_pools": os.path.abspath(
                        paths.training_data("_device_pools"))},
    }
    path = os.path.join(out_dir, "hyperparameters.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# The settings of this run - see _about below.\n")
        f.write(to_yaml(settings))
    log.detail(f"wrote {os.path.abspath(path)}")

    save_model_report(out_dir,
                      input_hw=(template.resolution, template.resolution),
                      extra={"run": os.path.basename(out_dir),
                             "sweep": settings["sweep"],
                             "shared_settings": shared},
                      quiet=True)
    log.detail(f"wrote {os.path.abspath(os.path.join(out_dir, 'model_structure.yaml'))}")


def plan(cfgs, skip_existing=True):
    """Print, per cell, whether its dataset and model already exist."""
    log.say(f"{'':4}{'configuration':<36}{'dataset':>12}{'model':>12}"
            f"{'scored':>10}")
    todo = {"dataset": 0, "model": 0}
    for i, cfg in enumerate(cfgs, 1):
        has_data = os.path.isfile(cfg.train_npz) and os.path.isfile(cfg.test_npz)
        has_model = os.path.isfile(cfg.checkpoint)
        has_eval = os.path.isfile(os.path.join(cfg.eval_dir, "metrics.json"))
        reuse_model = has_model and skip_existing
        todo["dataset"] += not has_data
        todo["model"] += not reuse_model
        log.say(f"{i:>3} {cfg.name:<36}"
                f"{('reuse' if has_data else 'BUILD'):>12}"
                f"{('reuse' if reuse_model else 'TRAIN'):>12}"
                f"{('done' if has_eval else '-'):>10}")
    # The devices are the expensive part and they are shared, so the cost is
    # dominated by however many pools are genuinely new.
    log.say(f"\nto do: {todo['dataset']} dataset(s), {todo['model']} training "
            f"run(s)")
    return todo


def _one_cell(cfg, skip_existing):
    """Stages 1-3 for a single configuration.  Returns its metrics."""
    log.say("\n--- step 1: dataset " + "-" * 50)
    report = dataset.build(cfg)
    if not report["all_passed"]:
        raise RuntimeError("a dataset split check FAILED — see "
                           "dataset_summary.json")

    log.say("\n--- step 2: train " + "-" * 52)
    if skip_existing and os.path.isfile(cfg.checkpoint):
        log.say(f"reusing {os.path.abspath(cfg.checkpoint)}")
    else:
        training.train(cfg)

    log.say("\n--- step 3: evaluate " + "-" * 49)
    return evaluation.run(cfg)


def _final_table(done, failed, out_dir=None):
    log.say("\n" + _RULE)
    log.say("FINAL RESULT")
    log.say(_RULE)
    log.say(f"{'rays':>5}{'points':>8}{'train':>7}{'test':>6}{'coverage':>10}"
            f"{'F1@1':>8}{'+/- sd':>8}{'IoU':>8}")
    for cfg, m in done:
        log.say(f"{cfg.n_rays:>5}{cfg.n_points:>8}{cfg.n_train:>7}"
                f"{m['n_test_devices']:>6}{100 * m['coverage']:>9.2f}%"
                f"{m['f1@1']:>8.3f}{m['f1@1_std']:>8.3f}{m['iou']:>8.3f}")
    for name, why in failed:
        log.say(f"{name}  FAILED: {why.splitlines()[0]}")

    if not done:
        return
    best_cfg, best = max(done, key=lambda d: d[1]["f1@1"])
    log.say(f"\nbest: {best_cfg.n_rays} rays x {best_cfg.n_points} points "
            f"— F1@1 {best['f1@1']:.3f} at "
            f"{100 * best['coverage']:.2f}% of the grid measured")
    log.say(f"\neverything from this run : "
            f"{os.path.abspath(out_dir or paths.CONFIG_ROOT)}")
    log.say(f"  the table, the figures, model_structure.yaml and "
            f"hyperparameters.yaml are in that folder")
    log.say(f"  best configuration      : {os.path.abspath(best_cfg.dir)}  "
            f"(and the other cells beside it)")


def run(template, rays, points, n_train, n_test, epochs,
        skip_existing=True, figure_devices=None,
        fresh_start=False, wipe_training_data=False):
    """Stages 1-4 for every (rays, points) combination.

    Restartable: anything already on disk is reused, so adding one more ray
    count and re-running costs only the new cell.  A cell that fails does not
    throw away the cells already finished — they are on disk and still get
    compared.
    """
    t0 = time.time()
    if fresh_start:
        wipe(results=True, training_data=wipe_training_data)
    cfgs = configs(template, rays, points, n_train, n_test, epochs,
                   figure_devices or {})

    # One run, one folder: the configuration folders are written INSIDE the
    # run folder under results/, beside the comparison they end up in.
    out_dir = run_dir(cfgs)
    paths.set_config_root(out_dir)

    log.say(_RULE)
    log.say("run_0 — the whole sweep, steps 1 to 4")
    log.say(_RULE)
    log.detail(f"settings taken from run_1_generate_dataset.py: "
               f"split_seed={template.split_seed}, "
               f"resolution={template.resolution}, seed={template.seed}")
    log.say(f"swept here: rays={rays}, points={points}, "
            f"n_train={n_train}, n_test={n_test}, epochs={epochs}\n")
    log.say(f"run folder : {os.path.abspath(out_dir)}")
    _describe_run(out_dir, template, cfgs, rays, points, n_train,
                  n_test, epochs)
    plan(cfgs, skip_existing)

    done, failed = [], []
    for i, cfg in enumerate(cfgs, 1):
        log.say(f"\n{'#' * 74}\n#  [{i}/{len(cfgs)}] {cfg.name}")
        try:
            done.append((cfg, _one_cell(cfg, skip_existing)))
        except Exception as exc:
            log.warn(f"\n[FAILED] {cfg.name}: {exc}")
            failed.append((cfg.name, str(exc)))

    if done:
        log.say(f"\n{'#' * 74}\n#  step 4: comparison")
        comparison.run(out_dir=out_dir, configs=[cfg for cfg, _ in done])

    _final_table(done, failed, out_dir)
    log.say(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")
    log.say(_RULE)
    return done, failed
