"""Paired cluster bootstrap with identical resamples across masks and methods."""
import copy
import json
from pathlib import Path
import numpy as np
from scripts.trainable_or_3way.reinforcement.common import (
    require, digest, atomic_npz, load_npz, Progress)
from scripts.trainable_or_3way.reinforcement.statistics import Metric
from .inputs import METHODS, COHORTS

METRICS = ("roc_auc", "fake_f1", "real_fpr")


def calculators(data, masks):
    # Share calculators for identical masks (the two AVDF1M filters currently coincide).
    unique, lookup, objects = [], [], []
    for mask in masks:
        match = next((i for i, m in enumerate(unique) if np.array_equal(mask, m)), None)
        if match is None:
            match = len(unique)
            unique.append(mask)
            objects.append([Metric(data["y"][mask], data["scores"][mask, j]) for j in range(len(METHODS))])
        lookup.append(match)
    return unique, lookup, objects


def points(data, masks, recorded=None):
    unique, lookup, objects = calculators(data, masks)
    rows = [[m.calculate() for m in group] for group in objects]
    result = {name: {method: rows[lookup[i]][j] for j, method in enumerate(METHODS)}
              for i, name in enumerate(COHORTS)}
    if recorded is not None:
        require(recorded["n"] == len(data["y"]), "Historical total changed")
        for method in METHODS:
            for metric in METRICS:
                require(abs(result["full"][method][metric] - recorded["metrics"][method][metric]) < 1e-10,
                        f"Full-cohort point estimate differs from committed result: {method}/{metric}")
    return result


def bootstrap(data, masks, folder, identity, *, replicates=5000, seed=1701,
              max_attempts=50000, checkpoint_every=25, stop_after=None):
    """stop_after is synthetic-test only; whole original clusters are sampled before filtering."""
    require(replicates >= 2 and max_attempts >= replicates and checkpoint_every > 0, "Invalid bootstrap settings")
    folder = Path(folder)
    y = data["y"]
    require(masks.shape == (len(COHORTS), len(y)) and masks.dtype == bool and masks[0].all(),
            "Invalid fixed cohort masks")
    clusters, inverse = np.unique(data["clusters"], return_inverse=True)
    require(len(clusters) > 1, "Need at least two clusters")
    unique, lookup, objects = calculators(data, masks)
    require(all(set(y[m].tolist()) == {0, 1} for m in unique), "Each original cohort needs both classes")
    bound = digest(dict(identity=identity, replicates=replicates, seed=seed,
                        max_attempts=max_attempts, checkpoint_every=checkpoint_every,
                        methods=METHODS, metrics=METRICS, cohorts=COHORTS,
                        sampling="original manifest clusters jointly across masks; reject any one-class draw"))
    state_path = folder/"bootstrap_state.npz"
    shape = (replicates, len(COHORTS), len(METHODS), len(METRICS))
    values = np.empty(shape, dtype=np.float64)
    rng = np.random.default_rng(seed)
    valid = attempts = 0
    if state_path.exists():
        state = load_npz(state_path)
        require(str(state["identity"]) == bound, "Bootstrap source/configuration changed")
        valid, attempts = int(state["valid"]), int(state["attempts"])
        require(0 <= valid <= replicates and valid <= attempts <= max_attempts, "Corrupt counters")
        require(state["values"].shape == (valid, *shape[1:]) and np.isfinite(state["values"]).all()
                and ((state["values"] >= 0) & (state["values"] <= 1)).all(), "Corrupt bootstrap state")
        values[:valid] = state["values"]
        rng.bit_generator.state = json.loads(str(state["rng"]))
    progress = Progress("SENSITIVITY bootstrap " + folder.name, replicates, folder, resumed=valid)

    def save():
        atomic_npz(state_path, identity=np.array(bound), values=values[:valid],
                   valid=np.array(valid), attempts=np.array(attempts),
                   rng=np.array(json.dumps(rng.bit_generator.state)))

    pending = None
    try:
        while valid < replicates and attempts < max_attempts:
            pending = (copy.deepcopy(rng.bit_generator.state), valid, attempts)
            counts = np.bincount(rng.integers(0, len(clusters), size=len(clusters)), minlength=len(clusters))
            weight = counts[inverse]
            attempts += 1
            eligible = all(0 < np.sum(weight[m]*y[m]) < np.sum(weight[m]) for m in unique)
            if eligible:
                # Calculate each metric bundle once, not once per reported metric.
                bundles = []
                for mask, group in zip(unique, objects):
                    result = [m.calculate(weight[mask]) for m in group]
                    bundles.append([[r[k] for k in METRICS] for r in result])
                values[valid] = np.asarray(bundles)[lookup]
                valid += 1
            pending = None
            if attempts % checkpoint_every == 0 or valid == replicates:
                save()
            progress.update(valid, detail=f"attempts={attempts} invalid={attempts-valid}")
            if stop_after is not None and valid >= stop_after:
                save()
                raise InterruptedError("Synthetic requested interruption")
    except BaseException:
        if pending is not None:
            rng.bit_generator.state, valid, attempts = pending
        save()
        raise
    save()
    require(valid == replicates, "Insufficient two-class bootstrap draws; no successful completion")
    progress.update(valid, force=True, detail=f"attempts={attempts} invalid={attempts-valid}")
    return values, dict(valid_replicates=valid, attempts=attempts, invalid_replicates=attempts-valid,
                        original_clusters=len(clusters), seed=seed)


def interval(values, level=.95):
    tail = (1-level)/2
    return np.quantile(values, [tail, 1-tail]).tolist()


def summarize(point, values, counters):
    comparisons = {}
    for ci, cohort in enumerate(COHORTS):
        comparisons[cohort] = {}
        for right in ("M1", "M0"):
            ri, pi = METHODS.index(right), METHODS.index("M4")
            draws = values[:, ci, pi] - values[:, ci, ri]
            full_draws = values[:, 0, pi] - values[:, 0, ri]
            entry = {}
            for mi, metric in enumerate(METRICS):
                delta = point[cohort]["M4"][metric] - point[cohort][right][metric]
                full_delta = point["full"]["M4"][metric] - point["full"][right][metric]
                primary = cohort == "metadata_unshared" and right == "M1" and metric == "roc_auc"
                entry[metric] = dict(delta=delta, ci95=interval(draws[:, mi]),
                                     family_ci=interval(draws[:, mi], .9875) if primary else None,
                                     delta_change_from_full=delta-full_delta,
                                     change_ci95=interval(draws[:, mi]-full_draws[:, mi]))
            comparisons[cohort]["M4-"+right] = entry
    return dict(**counters, comparisons=comparisons, primary_family_size=4,
                primary_individual_ci_level=.9875, secondary_ci_level=.95,
                method="paired whole-original-cluster percentile bootstrap, then apply fixed masks",
                rejection="draws with either class absent in any unique cohort are redrawn",
                scope="fixed trained models and audited test sample clusters; post-hoc sensitivity",
                cluster_definition="saved manifest group_id: FAV source person, AVDF1M identity")
