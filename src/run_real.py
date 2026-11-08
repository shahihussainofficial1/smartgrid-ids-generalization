"""Experiments on the MSU/ORNL power system attack data -> results/real.json

Protocols: random stratified 70/30 split of samples; leave-one-set-out (LOSO, 15 folds).
Feature sets: raw PMU, raw PMU + logs, physics, set-normalized PMU, physics + set-normalized.
Models: logistic regression, decision tree, random forest, LightGBM.
"""
import json
import sys
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data import ATTACK_TYPE, feature_groups, load
from features import matrix, physics, setnorm

warnings.filterwarnings("ignore")
SEED = 0
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def models():
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "DT": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "LGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                               random_state=SEED, verbose=-1, n_jobs=4),
    }


def scores(y, p, scen=None):
    out = {"acc": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}
    if set(np.unique(y)) <= {0, 1}:
        out["det_rate"] = float((p[y == 1] == 1).mean())
        out["far"] = float((p[y == 0] == 1).mean())
        out["majority_acc"] = float(max(y.mean(), 1 - y.mean()))
        if scen is not None:
            out["det_by_type"] = {t: float((p[(y == 1) & np.isin(scen, [s for s, v in ATTACK_TYPE.items() if v == t])] == 1).mean())
                                  for t in sorted(set(ATTACK_TYPE.values()))}
    return {k: (float(v) if not isinstance(v, dict) else v) for k, v in out.items()}


def run(X, y, groups, scen, model_names):
    res = {}
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=y)
    for m in model_names:
        mdl = models()[m].fit(X[tr], y[tr])
        res.setdefault("random", {})[m] = scores(y[te], mdl.predict(X[te]), scen[te])
        pred = np.zeros_like(y)
        for s in np.unique(groups):
            mdl = models()[m].fit(X[groups != s], y[groups != s])
            pred[groups == s] = mdl.predict(X[groups == s])
        r = scores(y, pred, scen)
        r["per_set_macro_f1"] = [float(f1_score(y[groups == s], pred[groups == s], average="macro")) for s in np.unique(groups)]
        res.setdefault("loso", {})[m] = r
    return res


def main():
    t0 = time.time()
    df = load()
    pmu, logs = feature_groups(df)
    phys = physics(df)
    sn = setnorm(df, pmu)
    F = {
        "raw": matrix(df[pmu]),
        "raw+logs": matrix(df[pmu + logs]),
        "physics": matrix(phys),
        "setnorm": matrix(sn),
        "physics+setnorm": matrix(pd.concat([phys, sn], axis=1)),
    }
    g = df["set"].values
    scen = df["scenario"].values
    resume = "--resume" in sys.argv and (OUT / "real.json").exists()
    R = json.load(open(OUT / "real.json")) if resume else {}
    R.update({"n": int(len(df)), "n_features": {k: int(v.shape[1]) for k, v in F.items()}})

    # sanity check: identify the set (operating condition) from normal-operation samples only
    if "set_identification_acc" not in R:
        nm = (scen == 41)
        tr, te = train_test_split(np.where(nm)[0], test_size=0.3, random_state=SEED, stratify=g[nm])
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED).fit(F["raw"][tr], g[tr])
        R["set_identification_acc"] = float(accuracy_score(g[te], rf.predict(F["raw"][te])))
        print("set id acc", R["set_identification_acc"], round(time.time() - t0))

    for fs, X in F.items():
        if fs in R.get("binary", {}):
            continue
        R.setdefault("binary", {})[fs] = run(X, df["binary"].values, g, scen, list(models()))
        print("binary", fs, round(time.time() - t0))
        json.dump(R, open(OUT / "real.json", "w"), indent=1)
    # three-class and 37-scenario labels (random forest only, to bound the run time)
    for fs in ["raw", "physics+setnorm"]:
        for lab in ["three", "scenario"]:
            if fs in R.get(lab, {}):
                continue
            R.setdefault(lab, {})[fs] = run(F[fs], df[lab].values, g, scen, ["RF"])
            print(lab, fs, round(time.time() - t0))
            json.dump(R, open(OUT / "real.json", "w"), indent=1)

    # footprint (binary, physics+setnorm)
    fp = {}
    X, y = F["physics+setnorm"], df["binary"].values
    for m in models():
        mdl = models()[m].fit(X, y)
        t = time.perf_counter()
        for i in range(300):
            mdl.predict(X[i:i + 1])
        fp[m] = {"size_kB": len(pickle.dumps(mdl)) / 1024, "latency_ms": (time.perf_counter() - t) / 300 * 1e3}
    R["footprint"] = fp
    json.dump(R, open(OUT / "real.json", "w"), indent=1)
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
