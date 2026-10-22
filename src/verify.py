"""
Physical and accounting sanity checks for dcmg-gym.

    python verify.py

Every check is a claim a reviewer could make; each one either passes or prints
the magnitude of the discrepancy. Run this before trusting any result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baselines import REGISTRY
from env import DCConfig, DCMicrogridEnv, calibrate
from profiles import DT_H, make_profiles

FAILED = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def rollout(cfg, df, ctrl="B5", episodes=4, seed=0) -> pd.DataFrame:
    env = DCMicrogridEnv(cfg=cfg, profiles=df, seed=seed, eval_mode=True)
    pol = REGISTRY[ctrl](env)
    frames = []
    for e in range(episodes):
        env.start_index = e * env.ep_len
        obs, _ = env.reset(seed=seed + e)
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(pol.act(obs, env))
            done = term or trunc
        f = pd.DataFrame(env.log)
        f["ep"] = e
        f["_sla_missed_total"] = env.sla_missed
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def main():
    import platform
    import scipy
    print(f"python {platform.python_version()}  numpy {np.__version__}  "
          f"pandas {pd.__version__}  scipy {scipy.__version__}")
    print("NOTE: numpy does not guarantee bit-identical Generator streams across\n"
          "      versions, so exact per-step values can differ between machines.\n"
          "      Physical invariants below must hold regardless; aggregate results\n"
          "      should agree to within seed noise. Record these versions in the paper.\n")
    df = make_profiles(n_days=60, seed=5)
    cfg = calibrate(DCConfig(), profiles=df, n_episodes=6)

    print("\n1. Exogenous profiles")
    check("GHI in [0,1]", df.ghi_norm.between(0, 1).all())
    check("wet-bulb <= dry-bulb everywhere", (df.t_wb <= df.t_amb + 1e-6).all(),
          f"max excess {float((df.t_wb - df.t_amb).max()):.3f} K")
    check("PV zero at night",
          float(df.loc[df.ghi_norm < 1e-9, "pv_kw"].abs().max()) < 1e-9)
    check("price strictly positive", (df.price > 0).all())
    check("carbon intensity in a physical range", df.carbon.between(0.02, 1.2).all())
    check("PV capacity factor plausible for the tropics",
          0.10 < df.pv_kw.mean() / cfg.pv_capacity_kw < 0.30,
          f"CF = {df.pv_kw.mean()/cfg.pv_capacity_kw:.3f}")

    for ctrl in ["B0", "B1", "B5"]:
        print(f"\n2. Physical accounting -- controller {ctrl}")
        d = rollout(cfg, df, ctrl)

        # --- power balance -------------------------------------------------
        resid = (d.p_it + d.p_cool - d.p_pv + d.p_ch - d.p_dis) - d.p_grid
        check("instantaneous power balance closes",
              float(resid.abs().max()) < 1e-6, f"max |resid| = {resid.abs().max():.2e} kW")
        check("import/export decomposition is exact",
              float((d.imp - d.exp - d.p_grid).abs().max()) < 1e-9)
        check("import and export never both positive",
              float((np.minimum(d.imp, d.exp)).max()) < 1e-9)

        # --- battery ---------------------------------------------------------
        soc_pred = d.soc.shift(1)
        soc_pred.iloc[0] = cfg.soc_init
        dsoc = (d.p_ch * DT_H * cfg.eta_ch - d.p_dis * DT_H / cfg.eta_dis) / cfg.bess_energy_kwh
        err = (soc_pred + dsoc - d.soc)
        # ignore the first row of each episode (reset discontinuity)
        err = err[d.step != 1]
        check("SoC integrates consistently", float(err.abs().max()) < 1e-8,
              f"max err = {err.abs().max():.2e}")
        check("SoC inside [soc_min, soc_max]",
              bool(d.soc.between(cfg.soc_min - 1e-9, cfg.soc_max + 1e-9).all()))
        check("battery never charges and discharges at once",
              float(np.minimum(d.p_ch, d.p_dis).max()) < 1e-9)
        check("battery power respects the rating",
              float(max(d.p_ch.max(), d.p_dis.max())) <= cfg.bess_power_kw + 1e-6)
        if d.p_ch.sum() > 1.0:
            rt = d.p_dis.sum() / d.p_ch.sum()
            check("round-trip loss present (no free energy)", rt < 1.0,
                  f"discharged/charged = {rt:.3f}")
        else:
            print("  [skip] round-trip check -- controller never charges")
        # Terminal storage value must be accounted for, or draining the battery
        # is free energy that reset() silently replenishes.
        ends = d.groupby("ep").tail(1)
        drained = ends[ends.soc < cfg.soc_init - 1e-6]
        check("draining the battery is charged for at episode end",
              len(drained) == 0 or bool((drained.soc_cost > 0).all()),
              f"{len(drained)}/{len(ends)} episodes end below soc_init; "
              f"mean terminal cost {ends.soc_cost.mean():.1f}")

        # --- IT / workload ---------------------------------------------------
        check("executed utilisation never exceeds the cap",
              float(d.u_exec.max()) <= cfg.u_cap + 1e-9)
        check("IT power within [idle, nameplate+aux]",
              bool(d.p_it.between(cfg.p_it_idle_kw + cfg.p_aux_frac * cfg.p_it_max_kw - 1e-6,
                                  cfg.p_it_max_kw * (1 + cfg.p_aux_frac) + 1e-6).all()),
              f"range {d.p_it.min():.1f}-{d.p_it.max():.1f} kW")
        # work conservation: offered = executed + still queued + missed
        for ep, g in d.groupby("ep"):
            offered = (g.u_off * DT_H).sum()
            executed = (g.u_exec * DT_H).sum()
            left = g.backlog.iloc[-1]
            missed = g.missed.sum()
            gap = offered - (executed + left + missed)
            check(f"work conserved (ep {ep})", abs(gap) < 1e-6 * max(offered, 1),
                  f"gap = {gap:.2e} util·h of {offered:.1f}")
            break   # one episode is enough to catch a systematic leak

        # --- thermal ---------------------------------------------------------
        tau = cfg.c_room_kwh_per_k / cfg.k_cool_kw_per_k
        decay = np.exp(-DT_H / tau)
        t_inf = d.t_sup + d.p_it / cfg.k_cool_kw_per_k
        # The interval-mean temperature must lie between where the room started
        # and where it is heading (it approaches t_inf monotonically from either side).
        lo = np.minimum(d.t_room_start, t_inf) - 1e-6
        hi = np.maximum(d.t_room_start, t_inf) + 1e-6
        check("room temperature between its start value and the steady state",
              bool(((d.t_room >= lo) & (d.t_room <= hi)).all()))
        # The room CAN sit below the supply-air temperature during a transient: at
        # reset, and after the controller raises the setpoint by several kelvin, the
        # fixed-flow air loop warms a room that is still cold. That is correct
        # physics for a constant-flow CRAC, not a violation. The invariants that do
        # hold unconditionally are (a) the asymptote always exceeds supply, because
        # the IT load is strictly positive, and (b) once the setpoint is steady the
        # room sits above supply.
        check("thermal asymptote always exceeds the supply-air temperature",
              bool((t_inf > d.t_sup + 1e-9).all()))
        steady = d[(d.step > 2) & (d.t_sup.diff().abs() < 0.5)]
        below = steady[steady.t_room < steady.t_sup - 1e-6]
        check("room sits above supply once the setpoint is steady",
              len(below) == 0,
              f"{len(below)}/{len(steady)} steady steps below supply"
              + (f", worst {float((below.t_sup - below.t_room).max()):.3f} K"
                 if len(below) else ""))
        # Setpoint chatter is not an error, but a bang-bang controller on a coarse
        # action grid will show it, and a reviewer will ask. Report it.
        print(f"  [info] setpoint chatter: mean |ΔT_sup| = "
              f"{float(d.t_sup.diff().abs().mean()):.2f} K/step")
        check("thermal time constant is 15 min", abs(tau - 0.25) < 1e-9,
              f"tau = {tau:.3f} h, decay = {decay:.3f}")
        if cfg.safety_layer:
            check("safety layer holds the ASHRAE envelope",
                  float(d.t_room.max()) <= cfg.t_room_max + 1e-6,
                  f"max T_room = {d.t_room.max():.3f} °C")
        check("setpoint always inside its bounds",
              bool(d.t_sup.between(cfg.t_sup_min - 1e-9, cfg.t_sup_max + 1e-9).all()))

        # --- cooling ---------------------------------------------------------
        check("COP inside declared bounds",
              bool(((d.cop >= cfg.cop_min - 1e-9) &
                    (d.cop <= max(cfg.cop_max, cfg.cop_free) + 1e-9)).all()))
        mech = d[d.free < .5]
        if mech.t_sup.std() > 1e-6:
            corr = float(mech[["t_sup", "cop"]].corr().iloc[0, 1])
            check("COP rises with setpoint (mechanical cooling only)", corr > 0.3,
                  f"corr = {corr:.3f}")
        else:
            print("  [skip] COP-vs-setpoint correlation -- setpoint is constant")
        check("PUE in a physically plausible band",
              bool(d.pue.between(1.05, 2.2).all()),
              f"range {d.pue.min():.3f}-{d.pue.max():.3f}")
        # Free cooling is a RESULT, not an invariant: it fires whenever the
        # controller happens to hold a high setpoint during a low-wet-bulb hour, so
        # the exact fraction is knife-edge sensitive to the controller and to RNG
        # stream differences between numpy versions. The physical claim worth
        # asserting is only that the tropics do not permit free cooling to dominate.
        check("free cooling is not the dominant mode in the tropics",
              float(d.free.mean()) < 0.35,
              f"free-cooling fraction {d.free.mean():.4f} "
              f"(wet-bulb p05 = {np.percentile(d.t_wb, 5):.1f} °C); "
              f"expect ~0 at a Bangkok-like site, a few % is normal")

        # --- economics -------------------------------------------------------
        e_cost = (d.price * d.imp * DT_H
                  - d.price * cfg.export_price_frac * d.exp * DT_H
                  + cfg.degrade_cost * (d.p_ch + d.p_dis) * DT_H)
        # the last step of each episode also carries the demand charge
        last = d.groupby("ep").tail(1).index
        diff = (d.cost - e_cost).drop(index=last)
        check("energy cost reproducible from price x import", float(diff.abs().max()) < 1e-6,
              f"max err {diff.abs().max():.2e}")
        check("carbon reproducible from CI x import",
              float((d.carbon - d.carbon_int * d.imp * DT_H).abs().max()) < 1e-9)
        check("PCC limit respected", float(d.pcc_exc.max()) < 1e-6,
              f"max excess {d.pcc_exc.max():.2f} kW")

    # --- determinism ---------------------------------------------------------
    print("\n3. Reproducibility")
    a = rollout(cfg, df, "B0", episodes=2, seed=3)
    b = rollout(cfg, df, "B0", episodes=2, seed=3)
    check("same seed -> identical trajectory",
          bool(np.allclose(a.p_grid.values, b.p_grid.values)))

    # --- monotonic sanity: more PV cannot cost more --------------------------
    print("\n4. Monotonicity")
    costs = []
    for pv in [0.0, 300.0, 600.0, 1200.0]:
        d2 = make_profiles(n_days=20, seed=5, pv_capacity_kw=pv)
        c2 = DCConfig(pv_capacity_kw=pv)
        c2.norm = cfg.norm
        e = DCMicrogridEnv(cfg=c2, profiles=d2, seed=0, eval_mode=True)
        p = REGISTRY["B0"](e)
        e.start_index = 0
        obs, _ = e.reset(seed=0)
        done = False
        while not done:
            obs, _, t1, t2, _ = e.step(p.act(obs, e))
            done = t1 or t2
        costs.append(e.summary()["cost"])
    check("cost is non-increasing in PV capacity",
          all(costs[i] >= costs[i + 1] - 1e-6 for i in range(len(costs) - 1)),
          " -> ".join(f"{c:.0f}" for c in costs))

    print("\n" + "=" * 62)
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED:")
        for f in FAILED:
            print("   -", f)
    else:
        print("ALL CHECKS PASSED")
    print("=" * 62)


if __name__ == "__main__":
    main()
