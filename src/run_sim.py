"""Detection experiments on the IEEE 14-bus FDI simulation -> results/sim.json

Detectors: chi-square bad data detection (BDD) on the WLS residual, a training-free
max.-deviation detector, and four lightweight classifiers on four feature sets:
  meas       magnitude and angle of every PMU channel
  meas+res   + residual statistic J and the largest normalized residual
  temporal   + deviation of every channel and estimated state from its trailing
               60-minute median (causal)
  aggregate  J, largest normalized residual and the largest trailing-median deviations
Splits: random 70/30; chronological (first 70% of the days for training); and two
chronological splits with stealthy attacks on unseen target buses. For the
chronological splits, alarm thresholds are also calibrated to a 1% false-alarm rate
on the normal samples of the last VAL_DAYS training days.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SEED = 0
WINDOW_MIN = 60    # length of the trailing median window in minutes
TRAIN_FRAC = 0.7   # chronological split: first 70% of the days for training
VAL_DAYS = 6       # last days of the training period used to calibrate alarm thresholds


def models():
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "DT": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "LGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                               random_state=SEED, verbose=-1, n_jobs=4),
    }


def features(s):
    Z, Xh, J, H = s["Z"], s["Xh"], s["J"], s["H"]
    W = np.diag(2 / (float(s["sigma"]) * np.abs(Z).mean(0)) ** 2)
    R = Z - Xh @ H.T
    rn = np.abs(R) * np.sqrt(np.diag(W).real)[None, :]
    meas = np.column_stack([np.abs(Z), np.angle(Z)])
    res = np.column_stack([J, rn.max(1)])
    st = np.column_stack([np.abs(Xh), np.angle(Xh)])
    base = np.column_stack([meas, st])
    w = WINDOW_MIN // int(s["step_min"])
    med = pd.DataFrame(base).shift(1).rolling(w, min_periods=max(3, w // 3)).median().values
    dev = base - med
    temporal = np.column_stack([meas, res, dev])
    m = Z.shape[1]
    n = Xh.shape[1]
    rel_meas = np.abs(dev[:, :m]) / (np.abs(med[:, :m]) + 1e-9)
    agg = np.column_stack([
        res,
        np.nanmax(rel_meas, 1),                         # largest relative change of a PMU magnitude
        np.nanmax(np.abs(dev[:, m:2 * m]), 1),          # largest change of a PMU angle
        np.nanmax(np.abs(dev[:, 2 * m:2 * m + n]), 1),  # largest change of an estimated |V|
        np.nanmax(np.abs(dev[:, 2 * m + n:]), 1),       # largest change of an estimated angle
    ])
    return {"meas": meas, "meas+res": np.column_stack([meas, res]), "temporal": temporal,
            "aggregate": agg}, dev[:, 2 * m:]


def max_deviation_detector(state_dev, normal_train, test):
    """Training-free detector: largest robust z-score of the change of any estimated
    state from its trailing median; scales and threshold use normal training data only."""
    ref = state_dev[normal_train]
    med = np.nanmedian(ref, 0)
    mad = np.nanmedian(np.abs(ref - med), 0) * 1.4826 + 1e-12
    score = np.nanmax(np.abs(state_dev - med) / mad, 1)
    thr = np.nanquantile(score[normal_train], 0.99)
    return (score[test] > thr).astype(int), float(thr)


def episode_ids(label):
    """Number contiguous attack episodes on the full timeline (-1 for normal samples)."""
    ep = np.full(len(label), -1)
    start = np.r_[True, label[1:] != label[:-1]] & (label > 0)
    ep[label > 0] = np.cumsum(start)[label > 0] - 1
    return ep


def evaluate(y3, p, ep):
    y = (y3 > 0).astype(int)
    out = {"f1": float(f1_score(y, p)),
           "det_random": float(p[y3 == 1].mean()), "det_stealthy": float(p[y3 == 2].mean()),
           "far": float(p[y3 == 0].mean())}
    for k, name in [(1, "random"), (2, "stealthy")]:
        e = ep[y3 == k]
        hit = {i: False for i in np.unique(e)}
        for i, pi in zip(e, p[y3 == k]):
            hit[i] = hit[i] or bool(pi)
        out[f"episodes_{name}"] = len(hit)
        out[f"ep_det_{name}"] = float(np.mean(list(hit.values()))) if hit else float("nan")
    return out


def main():
    s = dict(np.load(ROOT / "data" / "sim" / "sim.npz", allow_pickle=True))
    T = len(s["J"])
    keep = s["ok"].copy()
    keep[:WINDOW_MIN // int(s['step_min'])] = False  # trailing median needs history
    F, state_dev = features(s)
    y3 = s["label"]
    y = (y3 > 0).astype(int)
    day = (np.arange(T) * int(s["step_min"])) // 1440
    target = s["target"]
    R = {"T": int(T), "dof": int(s["dof"]),
         "counts": {"normal": int((y3[keep] == 0).sum()), "random": int((y3[keep] == 1).sum()),
                    "stealthy": int((y3[keep] == 2).sum())}}

    # chi-square BDD at 1% false-alarm level
    thr = chi2.ppf(0.99, int(s["dof"]))
    p_bdd = (s["J"] > thr).astype(int)
    R["BDD"] = evaluate(y3[keep], p_bdd[keep], episode_ids(y3)[keep])
    R["BDD"]["threshold"] = float(thr)

    idx = np.where(keep)[0]
    tr_r, te_r = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=y3[idx])
    split_day = int(TRAIN_FRAC * (day.max() + 1))
    tr_c, te_c = idx[day[idx] < split_day], idx[day[idx] >= split_day]
    # unseen targets: stealthy attacks on one half of the buses appear only in the test
    # period; both halves are used in turn (0-based odd = 1-based even buses, and vice versa)
    stealth = y3 == 2
    splits = {"random": (tr_r, te_r), "chronological": (tr_c, te_c)}
    for name, test_buses in [("unseen_even", np.arange(1, 14, 2)), ("unseen_odd", np.arange(2, 14, 2))]:
        unseen = np.isin(target, test_buses)
        splits[name] = (tr_c[~(stealth[tr_c] & unseen[tr_c])], te_c[~(stealth[te_c] & ~unseen[te_c])])
    ep = episode_ids(y3)
    R["split_sizes"] = {k: {"train": int(len(tr)), "test": int(len(te)),
                            "test_stealthy_samples": int(stealth[te].sum()),
                            "test_stealthy_episodes": int(len(np.unique(ep[te][stealth[te]])))} for k, (tr, te) in splits.items()}
    R["BDD_by_split"] = {k: evaluate(y3[te], p_bdd[te], ep[te]) for k, (tr, te) in splits.items()}
    R["maxdev_by_split"] = {}
    for k, (tr, te) in splits.items():
        p, thr = max_deviation_detector(state_dev, tr[y3[tr] == 0], te)
        R["maxdev_by_split"][k] = evaluate(y3[te], p, ep[te])
        R["maxdev_by_split"][k]["threshold"] = thr
        R["maxdev_by_split"][k + "_or_BDD"] = evaluate(y3[te], np.maximum(p, p_bdd[te]), ep[te])
    for sp, (tr, te) in splits.items():
        for fs, X in F.items():
            Xf = np.nan_to_num(X)
            for m in models():
                mdl = models()[m].fit(Xf[tr], y[tr])
                r = evaluate(y3[te], mdl.predict(Xf[te]), ep[te])
                # threshold-free comparison: ROC-AUC of stealthy vs. normal test samples
                score = mdl.predict_proba(Xf[te])[:, 1]
                sn = (y3[te] == 2) | (y3[te] == 0)
                r["auc_stealthy"] = float(roc_auc_score(y3[te][sn] == 2, score[sn]))
                if sp != "random":
                    # operational threshold: refit without the last VAL_DAYS of the training
                    # period and set the threshold to a 1% false-alarm rate on its normal samples
                    val = day[tr] >= split_day - VAL_DAYS
                    cal = models()[m].fit(Xf[tr[~val]], y[tr[~val]])
                    vn = tr[val & (y3[tr] == 0)]
                    thr_c = np.quantile(cal.predict_proba(Xf[vn])[:, 1], 0.99)
                    pc = (cal.predict_proba(Xf[te])[:, 1] > thr_c).astype(int)
                    rc = evaluate(y3[te], pc, ep[te])
                    r.update({f"cal_{k}": v for k, v in rc.items()})
                R.setdefault(sp, {}).setdefault(fs, {})[m] = r
            print(sp, fs, {m: (round(v["f1"], 3), round(v["det_stealthy"], 3)) for m, v in R[sp][fs].items()})
    json.dump(R, open(ROOT / "results" / "sim.json", "w"), indent=1)
    print("max-deviation detector", {k: (round(v["f1"], 3), round(v["det_stealthy"], 3), round(v["far"], 3))
                                     for k, v in R["maxdev_by_split"].items()})
    print("split sizes", R["split_sizes"])
    print("BDD", R["BDD"])


if __name__ == "__main__":
    main()
