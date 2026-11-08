"""Figures and LaTeX table rows for the paper."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
RE = json.load(open(ROOT / "results" / "real.json"))
SI = json.load(open(ROOT / "results" / "sim.json"))

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300, "savefig.bbox": "tight"})
COL = 3.45
FS_REAL = [("raw", "Raw PMU"), ("raw+logs", "Raw PMU\n+ logs"), ("physics", "Physics"),
           ("setnorm", "Set-\nnormalized"), ("physics+setnorm", "Physics +\nset-norm.")]
FS_SIM = [("meas", "Meas."), ("meas+res", "Meas. + residual"), ("temporal", "Temporal"), ("aggregate", "Aggregate")]


def unseen(fs, m, key):
    return np.mean([SI[s][fs][m][key] for s in ("unseen_even", "unseen_odd")])


def fig_real():
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    x = np.arange(len(FS_REAL))
    w = 0.38
    rnd = [RE["binary"][f]["random"]["RF"]["macro_f1"] for f, _ in FS_REAL]
    lso = [RE["binary"][f]["loso"]["RF"]["macro_f1"] for f, _ in FS_REAL]
    b1 = ax.bar(x - w / 2, rnd, w, color="#9e9e9e", label="Random split")
    b2 = ax.bar(x + w / 2, lso, w, color="#1f77b4", label="Leave-one-set-out")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=5.2)
    ax.axhline(0.5, color="k", ls=":", lw=0.7)
    ax.set_xticks(x, [l for _, l in FS_REAL], fontsize=6.5)
    ax.set_ylabel("Macro-F1 (RF, binary)")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "real_protocols.pdf")


def fig_timeseries():
    s = dict(np.load(ROOT / "data" / "sim" / "sim.npz", allow_pickle=True))
    J, y3 = s["J"], s["label"]
    step = int(s["step_min"])
    thr = chi2.ppf(0.99, int(s["dof"]))
    # pick a 24 h window in the test period containing both attack types
    per_day = 1440 // step
    best, start = -1, 0
    first_test_day = int(0.7 * (len(J) * step // 1440))
    for d in range(first_test_day, len(J) // per_day):
        sl = y3[d * per_day:(d + 1) * per_day]
        score = min((sl == 1).sum(), (sl == 2).sum())
        if score > best:
            best, start = score, d * per_day
    sl = slice(start, start + per_day)
    t = np.arange(per_day) * step / 60
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    for k, col, lab in [(1, "#fdae6b", "Random FDI"), (2, "#9ecae1", "Stealthy FDI")]:
        m = y3[sl] == k
        ax.fill_between(t, 0, 1, where=m, color=col, alpha=0.8, transform=ax.get_xaxis_transform(), label=lab, lw=0)
    ax.semilogy(t, J[sl], lw=0.6, color="k", label=r"$J$ (WLS residual)")
    ax.axhline(thr, color="tab:red", ls="--", lw=0.8, label=r"$\chi^2_{0.99}$ threshold")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(r"$J$")
    ax.set_xlim(0, 24)
    ax.legend(frameon=False, fontsize=5.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "sim_timeseries.pdf")


def fig_sim():
    U = ("unseen_even", "unseen_odd")
    bdd = lambda sp: (SI["BDD_by_split"][sp]["det_stealthy"] if sp != "unseen"
                      else np.mean([SI["BDD_by_split"][u]["det_stealthy"] for u in U]))
    get = lambda fs, m, key: (lambda sp: SI[sp][fs][m][key] if sp != "unseen" else unseen(fs, m, key))
    rows = [(r"$\chi^2$ BDD", bdd),
            ("RF, default threshold", get("meas+res", "RF", "det_stealthy")),
            ("RF, calibrated", get("meas+res", "RF", "cal_det_stealthy")),
            ("LGBM, calibrated", get("meas+res", "LGBM", "cal_det_stealthy"))]
    sps = [("chronological", "Chronological split"), ("unseen", "Unseen target buses")]
    cols = ["#d62728", "#c6dbef", "#3182bd", "#08519c"]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    x = np.arange(len(sps))
    w = 0.2
    for k, ((lab, fn), c) in enumerate(zip(rows, cols)):
        vals = [fn(sp) for sp, _ in sps]
        bars = ax.bar(x + (k - 1.5) * w, vals, w * 0.95, color=c, label=lab)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                    ha="center", fontsize=5.5)
    ax.set_xticks(x, [l for _, l in sps])
    ax.set_ylabel("Stealthy FDI detection rate")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, ncol=2, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "sim_detection.pdf")


def tables():
    lines = []
    for f, lab in FS_REAL:
        cells = []
        for m in ["LR", "DT", "RF", "LGBM"]:
            r, l = RE["binary"][f]["random"][m], RE["binary"][f]["loso"][m]
            cells.append(f"{r['macro_f1']:.2f}/{l['macro_f1']:.2f}")
        l = RE["binary"][f]["loso"]["RF"]
        cells.append(f"{l['det_rate']:.2f}/{l['far']:.2f}")
        clean = lab.replace("-\n", "-").replace("\n", " ")
        lines.append(f"{clean} & " + " & ".join(cells) + r" \\")
    (TAB / "real.tex").write_text("\n".join(lines) + "\n")

    U = ("unseen_even", "unseen_odd")
    um = lambda d, k: np.mean([d[u][k] for u in U])
    lines = []
    b = SI["BDD_by_split"]
    lines.append(r"\multicolumn{2}{@{}l}{$\chi^2$ BDD} & " +
                 f"{b['chronological']['det_random']:.2f} & {b['chronological']['det_stealthy']:.2f} & "
                 f"{b['chronological']['far']:.3f} & {um(b, 'det_stealthy'):.2f} & {um(b, 'far'):.3f} & -- \\\\")
    md = SI["maxdev_by_split"]
    lines.append(r"\multicolumn{2}{@{}l}{Max.-deviation} & " +
                 f"{md['chronological']['det_random']:.2f} & {md['chronological']['det_stealthy']:.2f} & "
                 f"{md['chronological']['far']:.3f} & {um(md, 'det_stealthy'):.2f} & {um(md, 'far'):.3f} & -- \\\\")
    lines.append(r"\midrule")
    for fs, flab in FS_SIM:
        for m in ["LR", "DT", "RF", "LGBM"]:
            c = SI["chronological"][fs][m]
            lines.append(f"{flab} & {m} & {c['cal_det_random']:.2f} & {c['cal_det_stealthy']:.2f} & {c['cal_far']:.3f} & "
                         f"{unseen(fs, m, 'cal_det_stealthy'):.2f} & {unseen(fs, m, 'cal_far'):.3f} & "
                         f"{unseen(fs, m, 'auc_stealthy'):.3f}" + r" \\")
        lines.append(r"\midrule")
    (TAB / "sim.tex").write_text("\n".join(lines[:-1]) + "\n")


if __name__ == "__main__":
    fig_real(); fig_timeseries(); fig_sim(); tables()
    print("figures and tables written")
