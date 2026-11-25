"""Paired whole-cluster bootstrap. Sorted-score sufficient statistics avoid repeated sorting."""
import numpy as np
from .common import require, digest, read, atomic_npz, load_npz, Progress

BOOT_METRICS = ("roc_auc", "fake_f1", "real_fpr", "fake_recall")
PAIRS = (("T2", "T0"), ("T2", "T1"))

class Metric:
    def __init__(self, labels, score):
        self.y = np.asarray(labels, dtype=np.int64)
        self.p = np.asarray(score, dtype=np.float64)
        require(self.y.shape == self.p.shape and self.y.ndim == 1 and len(self.y) > 0, "Invalid metric shapes")
        require(np.isin(self.y, [0,1]).all() and np.isfinite(self.p).all()
                and ((self.p>=0)&(self.p<=1)).all(), "Invalid metric input")
        self.order = np.argsort(self.p, kind="stable")
        sorted_p = self.p[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_p))+1]
        self.code = self.y*2+(self.p>=.5).astype(int)

    def calculate(self, weight=None):
        w = np.ones(len(self.y)) if weight is None else np.asarray(weight, dtype=np.float64)
        require(w.shape == self.y.shape and np.isfinite(w).all() and (w>=0).all(), "Invalid bootstrap weights")
        tn, fp, fn, tp = np.bincount(self.code, weights=w, minlength=4)
        real, fake = tn+fp, fn+tp
        positive = np.add.reduceat((w*self.y)[self.order], self.starts)
        negative = np.add.reduceat((w*(1-self.y))[self.order], self.starts)
        auc = float(np.sum(positive*(np.cumsum(negative)-.5*negative))/(real*fake)) if real and fake else None
        cp, ct = np.cumsum(positive[::-1]), np.cumsum((positive+negative)[::-1])
        precision = np.divide(cp, ct, out=np.zeros_like(cp), where=ct>0)
        ap = float(np.sum(positive[::-1]*precision)/fake) if fake else None
        f1 = float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.
        real_f1 = float(2*tn/(2*tn+fp+fn)) if 2*tn+fp+fn else 0.
        recall, fpr = (float(tp/fake) if fake else None), (float(fp/real) if real else None)
        return dict(n=int(real+fake), real=int(real), fake=int(fake), roc_auc=auc, fake_f1=f1,
                    real_fpr=fpr, fake_recall=recall, balanced_accuracy=(recall+1-fpr)/2 if real and fake else None,
                    macro_f1=(f1+real_f1)/2, average_precision=ap,
                    all_fake_f1=float(2*fake/(2*fake+real)) if real+fake else None,
                    confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])

def metric_table(y, scores):
    return {k: Metric(y,p).calculate() for k,p in scores.items()}

def bootstrap(y, scores, clusters, folder, identity, *, replicates=5000, seed=1701,
              max_attempts=50000, checkpoint_every=25, stop_after=None, pairs=PAIRS):
    """stop_after is test-only: exercise exact resume at a valid checkpoint."""
    import json
    from pathlib import Path
    folder = Path(folder)
    keys = list(scores)
    y = np.asarray(y)
    require(set(y.tolist()) == {0,1}, "Bootstrap needs both classes in the original cohort")
    unique, inverse = np.unique(clusters, return_inverse=True)
    require(len(unique)>1 and len(inverse)==len(y), "Need at least two verified clusters")
    config = dict(input_identity=identity, methods=keys, replicates=replicates, seed=seed,
                  max_attempts=max_attempts, metrics=list(BOOT_METRICS), pairs=list(pairs))
    bound = digest(config)
    state_path = folder/"bootstrap_state.npz"
    rng = np.random.default_rng(seed)
    results = np.empty((replicates,len(keys),len(BOOT_METRICS)), dtype=np.float64)
    valid, attempts = 0, 0
    if state_path.exists():
        state = load_npz(state_path)
        require(str(state["identity"])==bound, "Bootstrap input/config changed")
        valid, attempts = int(state["valid"]), int(state["attempts"])
        require(0<=valid<=replicates and valid<=attempts<=max_attempts, "Corrupt bootstrap counters")
        require(state["values"].shape==(valid,len(keys),len(BOOT_METRICS)) and np.isfinite(state["values"]).all(),
                "Corrupt bootstrap values")
        results[:valid] = state["values"]
        rng.bit_generator.state = json.loads(str(state["rng"]))
    metrics = [Metric(y,scores[k]) for k in keys]
    progress = Progress("TEMPORAL bootstrap "+folder.name, replicates, folder, resumed=valid)
    def save():
        atomic_npz(state_path, identity=np.array(bound), valid=np.array(valid), attempts=np.array(attempts),
                   values=results[:valid], rng=np.array(json.dumps(rng.bit_generator.state)))
    pending = None
    try:
        while valid<replicates and attempts<max_attempts:
            pending = (rng.bit_generator.state, valid, attempts)
            counts = np.bincount(rng.integers(0,len(unique),size=len(unique)), minlength=len(unique))
            weights = counts[inverse]
            attempts += 1
            pos = np.sum(weights*y)
            total = weights.sum()
            if not (0<pos<total):
                if attempts % checkpoint_every == 0:
                    save()
                continue
            for j,m in enumerate(metrics):
                r = m.calculate(weights)
                results[valid,j] = [r[k] for k in BOOT_METRICS]
            valid += 1
            pending = None
            progress.update(valid, detail=f"attempts={attempts} invalid={attempts-valid}")
            if valid%checkpoint_every==0 or valid==replicates:
                save()
            if stop_after is not None and valid>=stop_after:
                save()
                raise InterruptedError("Synthetic requested interruption")
    except BaseException:
        # Redraw an interrupted, uncommitted draw from its pre-draw RNG state.
        if pending is not None:
            rng.bit_generator.state, valid, attempts = pending
        save()
        raise
    save()
    progress.update(valid, force=True, detail=f"attempts={attempts} invalid={attempts-valid}")
    point = metric_table(y,scores)
    comparisons = {}
    for left,right in pairs:
        i,j = keys.index(left),keys.index(right)
        values = results[:valid,i]-results[:valid,j]
        comparisons[left+"-"+right] = {
            k: dict(delta=point[left][k]-point[right][k],
                    ci95=np.quantile(values[:,m],[.025,.975]).tolist() if valid==replicates else None,
                    draws_column=[i,j,m])
            for m,k in enumerate(BOOT_METRICS)}
    return dict(status="COMPLETE" if valid==replicates else "INSUFFICIENT_VALID_REPLICATES",
                valid_replicates=valid, attempts=attempts, invalid_replicates=attempts-valid,
                clusters=len(unique), seed=seed, comparisons=comparisons,
                inference_scope="fixed model and cohort only; not training/checkpoint/subset variability",
                simultaneous_family="Separate scope-locked aggregation; this file contains marginal 95% CIs only",
                method="paired whole-cluster percentile bootstrap; no target threshold tuning")
