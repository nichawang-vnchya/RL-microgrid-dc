"""
Publication figures for dcmg-gym.

    python plots.py --results ../results --out ../figures

Figures
    fig1_dispatch.png   24-h dispatch trace: PV, load, battery, price/carbon, T_room
    fig2_bars.png       cost / carbon / cooling-energy vs the status quo, with 95 % CI
    fig3_pareto.png     cost-carbon Pareto front from the preference-conditioned policy
    fig4_tradeoff.png   cooling energy vs thermal headroom -- the safety/efficiency tension
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import REGISTRY
from env import DCConfig, DCMicrogridEnv, calibrate
from profiles import DT_H
from train import build_profiles

# Colour system: one categorical hue per series, consistent across every figure.
C = {"pv": "#E8A33D", "it": "#3B6FA8", "cool": "#7FA8D0", "bess": "#5E9E7E",
     "grid": "#8A8F98", "price": "#C0554E", "carbon": "#6B5B95",
     "temp": "#C0554E", "ok": "#5E9E7E", "warn": "#C0554E"}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def _ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


# --------------------------------------------------------------------------
def fig_dispatch(out: str, day: int = 3, results: str | None = None):
    tr, te = build_profiles(seed=1)
    cfg = calibrate(DCConfig(), profiles=tr, n_episodes=8)
    env = DCMicrogridEnv(cfg=cfg, profiles=te, seed=0, eval_mode=True)
    env.start_index = day * env.ep_len

    model, label = None, "MPC-4h"
    if results:
        p = os.path.join(results, "R1_s0.zip")
        if os.path.exists(p):
            from stable_baselines3 import PPO
            model, label = PPO.load(p), "trained PPO agent"
    pol = None if model else REGISTRY["B5"](env)

    obs, _ = env.reset(seed=7)
    done = False
    while not done:
        a = model.predict(obs, deterministic=True)[0] if model else pol.act(obs, env)
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
    d = pd.DataFrame(env.log)
    t = np.arange(len(d)) * DT_H

    fig, ax = plt.subplots(4, 1, figsize=(7.4, 8.2), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1.3, 1.3, 1.3]})

    a = ax[0]
    a.stackplot(t, d.p_it, d.p_cool, labels=["IT", "Cooling"],
                colors=[C["it"], C["cool"]], alpha=.9, lw=0)
    a.plot(t, d.p_pv, color=C["pv"], lw=2.0, label="PV available")
    a.plot(t, d.imp, color="#333", lw=1.3, ls="--", label="Grid import")
    a.set_ylabel("Power (kW)")
    a.legend(ncol=4, loc="upper left", fontsize=8)
    a.set_title(f"Joint dispatch over one held-out test day — {label}", loc="left",
                fontsize=10.5, fontweight="bold")

    a = ax[1]
    a.plot(t, d.u_off, color=C["grid"], lw=1.5, ls="--", label="Offered utilisation")
    a.plot(t, d.u_exec, color=C["it"], lw=1.9, label="Executed utilisation")
    a.set_ylabel("Utilisation (–)")
    a.set_ylim(0, 1)
    ab = a.twinx()
    ab.fill_between(t, d.backlog, color=C["pv"], alpha=.28, lw=0, label="Deferred backlog")
    ab.set_ylabel("Backlog (util·h)")
    ab.grid(False)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = ab.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, ncol=3, loc="upper left", fontsize=8)

    a = ax[2]
    a.plot(t, d.soc, color=C["bess"], lw=1.8, label="Battery SoC")
    a.fill_between(t, cfg.soc_min, cfg.soc_max, color=C["bess"], alpha=.07)
    a.set_ylabel("SoC (-)")
    a.set_ylim(0, 1)
    a2 = a.twinx()
    a2.plot(t, d.price, color=C["price"], lw=1.2, label="Price")
    a2.plot(t, d.carbon_int, color=C["carbon"], lw=1.2, ls=":", label="Carbon intensity")
    a2.set_ylabel("Price ($/kWh) · CI (kg/kWh)")
    a2.grid(False)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, ncol=3, loc="upper left", fontsize=8)

    a = ax[3]
    a.plot(t, d.t_room, color=C["temp"], lw=1.8, label="Room temperature")
    a.plot(t, d.t_sup, color=C["it"], lw=1.4, ls="--", label="CRAC supply setpoint")
    a.axhline(cfg.t_room_max, color=C["warn"], lw=1.0, ls=":")
    a.text(t[-1], cfg.t_room_max + .12, "ASHRAE recommended limit",
           ha="right", fontsize=7.5, color=C["warn"])
    a.set_ylabel("Temperature (°C)")
    a.set_xlabel("Hour of day")
    a.legend(ncol=2, loc="lower left", fontsize=8)
    a.set_xlim(0, t[-1])

    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_dispatch.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_bars(results: str, out: str):
    df = pd.read_csv(os.path.join(results, "results_raw.csv"))
    b0 = df[df.controller.str.startswith("B0")]
    metrics = [("cost", "Electricity cost"), ("carbon_kg", "Operational CO₂"),
               ("energy_cool_kwh", "Cooling energy")]
    order = sorted(df.controller.unique())
    order = [c for c in order if not c.startswith("B0")]

    # PAIRED differences. Every controller runs the same held-out days, so the
    # comparison is paired: differencing against B0 on the SAME episode removes
    # day-to-day weather/price variance, which otherwise dominates and makes every
    # controller look statistically indistinguishable. Unpaired CIs here would be
    # a real statistical error, not just a conservative one.
    base_by_ep = b0.set_index("episode")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        vals, errs, sig = [], [], []
        for c in order:
            s = df[df.controller == c]
            d = ((s[m].to_numpy()
                  / base_by_ep.loc[s.episode, m].to_numpy() - 1) * 100)
            vals.append(d.mean())
            errs.append(_ci(d))
            sig.append(abs(d.mean()) > _ci(d))       # CI excludes zero
        import matplotlib.colors as mcolors
        cols = [mcolors.to_rgba(C["ok"] if v < 0 else C["warn"], .9 if s else .3)
                for v, s in zip(vals, sig)]
        y = np.arange(len(order))
        ax.barh(y, vals, xerr=errs, color=cols,
                error_kw=dict(lw=1, capsize=2.5, ecolor="#444"))
        pad = 0.06 * (max(np.array(vals) + np.array(errs)).item()
                      - min(np.array(vals) - np.array(errs)).item())
        for i, (v, e, s) in enumerate(zip(vals, errs, sig)):
            # place the label just OUTSIDE the error bar, on the far side from zero
            ax.text(v + np.sign(v) * (e + pad), i,
                    f"{v:+.1f}" + ("" if s else "  n.s."),
                    va="center", ha="right" if v < 0 else "left", fontsize=7.4,
                    color="#333")
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8)
        ax.axvline(0, color="#333", lw=1)
        ax.set_xlabel("Change vs status quo (%)")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.margins(x=.22)
        ax.invert_yaxis()
    fig.suptitle("Paired performance vs the status-quo controller — same held-out days, "
                 "mean ± 95 % CI\nfaded bars marked n.s. are not distinguishable from zero",
                 x=.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(os.path.join(out, "fig2_bars.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_pareto(results: str, out: str):
    p = os.path.join(results, "pareto_raw.csv")
    if not os.path.exists(p):
        print("no pareto_raw.csv -- run train.py --pareto")
        return
    dfp = pd.read_csv(p)
    dfb = pd.read_csv(os.path.join(results, "results_raw.csv"))
    g = dfp.groupby("controller").agg(cost=("cost", "mean"), co2=("carbon_kg", "mean")).reset_index()
    g["lam"] = g.controller.str.extract(r"=([\d.]+)").astype(float)
    g = g.sort_values("lam")

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    sc = ax.scatter(g.cost, g.co2, c=g.lam, cmap="viridis", s=52, zorder=3,
                    edgecolor="white", lw=.7)
    ax.plot(g.cost, g.co2, color="#555", lw=1.1, alpha=.6, zorder=2)
    plt.colorbar(sc, ax=ax, label="carbon preference weight $w_{carbon}$")

    mk = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, c in enumerate(sorted(dfb.controller.unique())):
        s = dfb[dfb.controller == c]
        ax.scatter(s.cost.mean(), s.carbon_kg.mean(), marker=mk[i % len(mk)],
                   color="#C0554E", s=44, zorder=4, edgecolor="white", lw=.7)
        ax.annotate(c.split()[0], (s.cost.mean(), s.carbon_kg.mean()),
                    textcoords="offset points", xytext=(6, 3), fontsize=7.5,
                    color="#C0554E")
    ax.set_xlabel("Electricity cost per episode ($)")
    ax.set_ylabel("Operational CO₂ per episode (kg)")
    ax.set_title("Cost–carbon Pareto front from one preference-conditioned policy",
                 loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_pareto.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_tradeoff(out: str):
    """Cooling energy vs thermal headroom, swept over a fixed setpoint. Shows the
    physical tension the agent must resolve, independent of any RL result."""
    tr, te = build_profiles(seed=1)
    cfg = calibrate(DCConfig(), profiles=tr, n_episodes=6)
    cfg.safety_layer = False           # sweep the raw setpoint
    rows = []
    for tset in np.arange(18, 27.1, 0.5):
        env = DCMicrogridEnv(cfg=cfg, profiles=te, seed=0, eval_mode=True)
        for ep in range(5):
            env.start_index = ep * env.ep_len
            obs, _ = env.reset(seed=11 + ep)
            a = np.array([1.0, 0.0,
                          2 * (tset - cfg.t_sup_min) / (cfg.t_sup_max - cfg.t_sup_min) - 1])
            done = False
            while not done:
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
            s = env.summary()
            s["tset"] = tset
            rows.append(s)
    d = pd.DataFrame(rows).groupby("tset").mean(numeric_only=True).reset_index()

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(d.tset, d.energy_cool_kwh, color=C["cool"], lw=2, marker="o", ms=3.5)
    ax.set_xlabel("CRAC supply-air setpoint (°C)")
    ax.set_ylabel("Cooling energy per day (kWh)", color=C["cool"])
    ax2 = ax.twinx()
    ax2.plot(d.tset, d.thermal_viol_rate * 100, color=C["warn"], lw=2, ls="--",
             marker="s", ms=3.5)
    ax2.set_ylabel("Steps above the ASHRAE limit (%)", color=C["warn"])
    ax2.grid(False)
    safe = d[d.thermal_viol_rate < 1e-9]
    if len(safe):
        ax.axvline(safe.tset.max(), color="#333", lw=1, ls=":")
        ax.text(safe.tset.max() - .1, d.energy_cool_kwh.max(),
                "highest safe fixed setpoint", rotation=90, ha="right", va="top",
                fontsize=7.5)
    ax.set_title("Why the cooling and IT levers must be co-optimised",
                 loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_tradeoff.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_dispatch(a.out, results=a.results)
    fig_tradeoff(a.out)
    if os.path.exists(os.path.join(a.results, "results_raw.csv")):
        fig_bars(a.results, a.out)
        fig_pareto(a.results, a.out)
    print("figures ->", a.out)
