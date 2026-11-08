"""Load the MSU/ORNL power system attack dataset (multiclass ARFF files) into one table.

Each of the 15 files ("sets") is a 1% random sample of 37 scenarios. The marker
column gives the scenario number; binary and three-class labels follow the
dataset documentation.
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "multiclass"
CACHE = Path(__file__).resolve().parents[1] / "data" / "interim" / "power.parquet"

NATURAL = [1, 2, 3, 4, 5, 6, 13, 14]
NORMAL = [41]
ATTACK = [7, 8, 9, 10, 11, 12] + list(range(15, 31)) + list(range(35, 41))
ATTACK_TYPE = {**{s: "Data injection" for s in range(7, 13)},
               **{s: "Command injection" for s in range(15, 21)},
               **{s: "Relay setting change" for s in list(range(21, 31)) + list(range(35, 41))}}
LOG_COLS_PATTERN = r"(log)"


def read_arff(path):
    txt = Path(path).read_text()
    hdr, data = txt.split("@data", 1)
    names = [line.split()[1] for line in hdr.splitlines() if line.lower().startswith("@attribute")]
    return pd.read_csv(io.StringIO(data), header=None, names=names)


def load(refresh=False):
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    frames = []
    for f in RAW.glob("*.arff"):
        d = read_arff(f)
        d["set"] = int(re.search(r"data(\d+)", f.name).group(1))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).sort_values(["set"], kind="stable").reset_index(drop=True)
    df = df.rename(columns={"marker": "scenario"})
    df["binary"] = df["scenario"].isin(ATTACK).astype(int)
    df["three"] = np.select([df["scenario"].isin(ATTACK), df["scenario"].isin(NATURAL)], [2, 1], 0)
    num = df.columns.difference(["set", "scenario", "binary", "three"])
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def feature_groups(df):
    meta = {"set", "scenario", "binary", "three"}
    logs = [c for c in df.columns if re.search(LOG_COLS_PATTERN, c)]
    pmu = [c for c in df.columns if c not in meta and c not in logs]
    return pmu, logs


if __name__ == "__main__":
    df = load(refresh=True)
    pmu, logs = feature_groups(df)
    print(df.shape, "PMU features", len(pmu), "log features", len(logs))
    print("rows per set", df.groupby("set").size().to_dict())
    print("class balance (binary)", df["binary"].value_counts().to_dict())
    print("NaN (from inf) per column, top:", df[pmu].isna().sum().sort_values(ascending=False).head(6).to_dict())
