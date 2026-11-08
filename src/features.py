"""Feature sets for the MSU/ORNL power system data.

raw     : the 116 PMU columns as recorded (infinite impedances -> NaN -> sentinel)
logs    : the 12 control-panel, relay and Snort log columns
physics : load-independent and line-end consistency features derived from the PMUs
setnorm : raw PMU columns robustly standardized per set (median / IQR of all rows of
          the set, no labels), i.e., relative to the operating condition of the set
"""
import numpy as np
import pandas as pd

RELAYS = ["R1", "R2", "R3", "R4"]
LINES = {"L1": ("R1", "R2"), "L2": ("R3", "R4")}
EPS = 1e-6
SENTINEL = -1.0


def _c(df, r, name):
    sep = ":" if name in ("F", "DF", "S") else "-"
    return df[f"{r}{sep}{name}"].astype(float)


def physics(df):
    f = {}
    per = {}
    for r in RELAYS:
        i1, i2, i0 = (_c(df, r, f"PM{k}:I") for k in (10, 11, 12))
        v1, v2, v0 = (_c(df, r, f"PM{k}:V") for k in (7, 8, 9))
        ia, ib, ic = (_c(df, r, f"PM{k}:I") for k in (4, 5, 6))
        va, vb, vc = (_c(df, r, f"PM{k}:V") for k in (1, 2, 3))
        iph = np.column_stack([ia, ib, ic])
        vph = np.column_stack([va, vb, vc])
        z = _c(df, r, "PA:Z")
        d = {
            "I2_I1": i2 / (i1 + EPS), "I0_I1": i0 / (i1 + EPS),
            "V2_V1": v2 / (v1 + EPS), "V0_V1": v0 / (v1 + EPS),
            "I_unb": (iph.max(1) - iph.min(1)) / (iph.mean(1) + EPS),
            "V_unb": (vph.max(1) - vph.min(1)) / (vph.mean(1) + EPS),
            "Imax_Imean": iph.max(1) / (iph.mean(1) + EPS),
            "Vmin_Vmean": vph.min(1) / (vph.mean(1) + EPS),
            "f_dev": (_c(df, r, "F") - 60.0).abs(),
            "dfdt": _c(df, r, "DF").abs(),
            "Z_inf": z.isna().astype(float),
            "status": _c(df, r, "S"),
        }
        d = {k: pd.Series(np.asarray(v, float), index=df.index) for k, v in d.items()}
        per[r] = d
        for k, v in d.items():
            f[f"{r}_{k}"] = v
    # consistency between the two ends of each line
    for line, (a, b) in LINES.items():
        for k in ["I2_I1", "I0_I1", "V2_V1", "I_unb", "V_unb", "Vmin_Vmean"]:
            f[f"{line}_d_{k}"] = (per[a][k] - per[b][k]).abs()
        ia1, ib1 = _c(df, a, "PM10:I"), _c(df, b, "PM10:I")
        f[f"{line}_I1_ratio"] = np.log((ia1 + 1) / (ib1 + 1)).abs()
    out = pd.DataFrame(f, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def setnorm(df, cols):
    x = df[cols].astype(float)
    out = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for s, g in x.groupby(df["set"]):
        med = g.median()
        iqr = (g.quantile(0.75) - g.quantile(0.25)).replace(0, np.nan)
        out.loc[g.index] = ((g - med) / iqr).clip(-50, 50).values
    return out


def matrix(frame):
    """Fill NaN with a sentinel for models without native missing-value support."""
    return frame.astype(float).fillna(SENTINEL).values
