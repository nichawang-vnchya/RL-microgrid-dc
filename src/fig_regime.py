"""
Figures for the journal paper (dense-peak, LANTA-calibrated runs, 30 test days).

  Fig R  regime map: (a) saving vs status quo for best rule / PPO / oracle in each
         tariff; (b) the two screening variables with PPO-minus-rule annotated.
  Fig D  daily cost, PPO vs best rule, days sorted by status-quo cost (SG and TH).

    python fig_regime.py --out ../docs
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# colour slots (validated categorical palette): blue, orange, aqua; neutral grey for B0
C_RULE, C_PPO, C_ORC, C_B0 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"

ARMS = {
    "SG": dict(label="Singapore\nspot (USEP)", rl="../results_sg_final/results_raw.csv",
               orc="../results_oracle3_sg/oracle3_paired.csv",
               prof="../data/profiles_real.csv", peak_charge=8.0),
    "TH": dict(label="Thailand\nTOU (MEA 4.2)", rl="../results_th_final/results_raw.csv",
               orc="../results_oracle3_th/oracle3_paired.csv",
               prof="../data/profiles_th.csv", peak_charge=3.91),
    "CA": dict(label="California\nspot (CAISO)", rl="../results_ca_dp_v2/results_raw.csv",
               orc="../results_oracle3_ca/oracle3_paired.csv",
               prof="../data/profiles_ca.csv", peak_charge=8.0),
}


def load_arm(a):
    d = pd.read_csv(a["rl"])
    day = d.groupby(["controller", "episode"]).cost.mean().unstack(0)
    rules = [c for c in day.columns if c.startswith("B") and c != "B0 status-quo"]
    best = day[rules].sum().idxmin()
    orc = pd.read_csv(a["orc"]).set_index("episode").cost if os.path.exists(a["orc"]) else None
    prof = pd.read_csv(a["prof"], index_col=0, parse_dates=True)
    b0 = d[d.controller == "B0 status-quo"]
    return dict(day=day, best=best, orc=orc,
                price_cv=float(prof.price.std() / prof.price.mean()),
                price_tail=float(prof.price.quantile(0.99) / prof.price.mean()),
                peak_share=float((b0.peak_cost / b0.cost).mean()))


def bill(day, col, ref="B0 status-quo"):
    return 100 * (day[col].sum() / day[ref].sum() - 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../docs"); args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#b9b8b2",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    R = {k: load_arm(a) for k, a in ARMS.items()}

    # ---------------- Fig R: regime map ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.25, 1]))
    keys = list(ARMS); x = np.arange(len(keys)); w = 0.26
    vals = {"rule": [], "ppo": [], "orc": []}
    for k in keys:
        r = R[k]; d = r["day"]
        vals["rule"].append(bill(d, r["best"]))
        vals["ppo"].append(bill(d, "R1 PPO"))
        vals["orc"].append(100 * (r["orc"].sum() / d["B0 status-quo"].sum() - 1) if r["orc"] is not None else np.nan)
    for j, (key, col, lab) in enumerate([("rule", C_RULE, "best rule"), ("ppo", C_PPO, "safe PPO"),
                                         ("orc", C_ORC, "perfect-foresight oracle")]):
        bars = ax1.bar(x + (j - 1) * w, vals[key], w * 0.92, color=col, label=lab, zorder=3)
        for bx, v in zip(x + (j - 1) * w, vals[key]):
            if not np.isnan(v):
                ax1.text(bx, v + (0.25 if v >= 0 else -0.25), f"{v:+.1f}", ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=7, color="#0b0b0b")
    ax1.axhline(0, color="#52514e", lw=0.8, zorder=2)
    ax1.set_xticks(x); ax1.set_xticklabels([ARMS[k]["label"] for k in keys])
    ax1.set_ylabel("30-day bill vs status quo (%)")
    ax1.yaxis.grid(True, color="#e6e5e0", lw=0.6, zorder=0); ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax1.set_title("(a) What each controller saves", fontsize=9, loc="left")
    lo = min(min(v for v in vals["orc"] if not np.isnan(v)), min(vals["ppo"])) - 1.8
    hi = max(max(vals["ppo"]), 0) + 1.5
    ax1.set_ylim(lo, hi)

    # (b) screening variables
    for k, col in zip(keys, [C_B0, C_B0, C_B0]):
        r = R[k]; d = r["day"]
        gain = bill(d, "R1 PPO", ref=r["best"])
        ax2.scatter(100 * r["peak_share"], r["price_tail"], s=160, color=C_PPO if gain < 0 else C_RULE,
                    edgecolor="#fcfcfb", lw=1.5, zorder=3)
        ax2.annotate(f"{k}\nPPO vs rule {gain:+.1f} %", (100 * r["peak_share"], r["price_tail"]),
                     xytext=(8, 0), textcoords="offset points", fontsize=7.5, va="center")
    ax2.set_xlabel("demand charge, share of status-quo bill (%)")
    ax2.set_ylabel("price tail  (p99 / mean)")
    ax2.set_xlim(0, 32); ax2.set_ylim(0, 9)
    ax2.grid(True, color="#e6e5e0", lw=0.6, zorder=0); ax2.set_axisbelow(True)
    ax2.set_title("(b) Where learning pays", fontsize=9, loc="left")
    ax2.text(31, 0.4, "orange: PPO beats the best rule\nblue: rule is cheaper", fontsize=7, color="#52514e",
             va="bottom", ha="right")
    ax2.set_ylim(0, 9.5)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(args.out, "figR_regime_map.png"), dpi=300)
    fig.savefig(os.path.join(args.out, "figR_regime_map.pdf"))

    # ---------------- Fig D: daily cost PPO vs rule ----------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for ax, k in zip(axes, ["SG", "TH"]):
        r = R[k]; d = r["day"]; order = d["B0 status-quo"].sort_values().index
        xx = np.arange(len(order))
        ax.plot(xx, d.loc[order, "B0 status-quo"].values, color=C_B0, lw=1.2, label="status quo (B0)")
        ax.plot(xx, d.loc[order, r["best"]].values, color=C_RULE, lw=2, label=f"best rule ({r['best'].split(' ')[0]})")
        ax.plot(xx, d.loc[order, "R1 PPO"].values, color=C_PPO, lw=2, label="safe PPO", marker="o", ms=3.5,
                markerfacecolor="#fcfcfb", markeredgewidth=1.2)
        if r["orc"] is not None:
            ax.plot(xx, r["orc"].loc[order].values, color=C_ORC, lw=1.4, ls="--", label="oracle")
        wins = int((d.loc[order, "R1 PPO"] < d.loc[order, r["best"]]).sum())
        short = {"SG": "(a) Singapore spot", "TH": "(b) Thailand TOU"}[k]
        ax.set_title(f"{short} — PPO beats rule {wins}/30 days", fontsize=8.5, loc="left")
        ax.set_xlabel("test day, sorted by status-quo cost"); ax.set_ylabel("daily cost (USD)")
        ax.grid(True, color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(args.out, "figD_daily_cost.png"), dpi=300)
    fig.savefig(os.path.join(args.out, "figD_daily_cost.pdf"))

    # numbers used
    rows = []
    for k in keys:
        r = R[k]; d = r["day"]
        rows.append(dict(arm=k, best_rule=r["best"], rule_vs_b0=bill(d, r["best"]), ppo_vs_b0=bill(d, "R1 PPO"),
                         ppo_vs_rule=bill(d, "R1 PPO", ref=r["best"]),
                         oracle_vs_b0=(100 * (r["orc"].sum() / d["B0 status-quo"].sum() - 1)) if r["orc"] is not None else np.nan,
                         ppo_share_of_attainable=((d["B0 status-quo"].sum() - d["R1 PPO"].sum())
                                                  / (d["B0 status-quo"].sum() - r["orc"].sum())) if r["orc"] is not None else np.nan,
                         price_cv=r["price_cv"], price_tail=r["price_tail"], peak_share=r["peak_share"],
                         ppo_win_days=int((d["R1 PPO"] < d[r["best"]]).sum())))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(args.out, "regime_table.csv"), index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
