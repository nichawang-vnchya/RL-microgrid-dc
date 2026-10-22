"""
Simulator-verified perfect-foresight oracle (replaces oracle2 for the TOU arms).

    python oracle3.py --profiles ../data/profiles_th.csv --episodes 30 \
        --idle-frac 0.52 --peak-charge 3.91 --out ../results_oracle3_th

Why a third version. oracle.py holds the cooling set-point constant over the
day (fine under the Singapore spot tariff, where 22 C won every day, but it is
beaten by a simple rule under a flat time-of-use tariff, where cooling is the
main lever). oracle2.py made the set-point a per-interval binary choice, but the
resulting MILP is too large for CBC to solve to optimality in reasonable time,
and a time-limited incumbent is not an optimum.

This version exploits the structure instead: the coefficient of performance
rises monotonically with the set-point, so for any workload/battery schedule
the cost-minimising set-point is the HIGHEST one the ASHRAE constraint allows,
which is exactly what the environment's own thermal projection computes. The
oracle therefore
  1. fixes a per-step set-point trajectory, solves the dispatch + battery MILP
     with foresight (exact SPECpower curve, deadlines as hard constraints),
  2. recomputes the maximum feasible set-point from the resulting IT power,
  3. repeats until the trajectory stops changing (2-3 rounds), and
  4. REPLAYS the final schedule through the environment itself, commanding
     the set-point at its upper bound and letting the projection clip it.
The reported cost is the environment's own accounting of an achievable
schedule, so it is a conservative (upper-bound-on-cost) oracle that is
guaranteed consistent with every other number in the paper.
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import pulp

warnings.filterwarnings("ignore")

from env import DCConfig, DCMicrogridEnv, calibrate      # noqa: E402
from profiles import DT_H, STEPS_PER_DAY                 # noqa: E402
from train import build_profiles                         # noqa: E402

SPEC_X = [0.0, 0.10, 0.50, 1.00]
SPEC_Y = [0.0, 0.2504, 0.5704, 1.00]


def cop_at(c, t_sup, t_wb):
    if t_wb < (t_sup - c.econ_delta_t):
        return c.cop_free
    v = (c.cop_nom * (1.0 + c.cop_gamma * (t_sup - c.t_sup_ref))
         * (1.0 - c.cop_delta * (t_wb - c.t_wb_ref)))
    return float(np.clip(v, c.cop_min, c.cop_max))


def max_feasible_tsup(c, p_it, t_room0):
    """Forward pass of the environment's thermal projection at full command."""
    tau = c.c_room_kwh_per_k / c.k_cool_kw_per_k
    decay = float(np.exp(-DT_H / tau))
    t = t_room0
    out = []
    for p in p_it:
        feas = (c.t_room_max - t * decay) / (1.0 - decay) - p / c.k_cool_kw_per_k
        ts = float(np.clip(min(c.t_sup_max, feas), c.t_sup_min, c.t_sup_max))
        out.append(ts)
        t_inf = ts + p / c.k_cool_kw_per_k
        t = t_inf + (t - t_inf) * decay
    return np.array(out)


def solve_fixed_tsup(c, day, t_sup, msg=False):
    T = len(day)
    price = day["price"].to_numpy(); pv = day["pv_kw"].to_numpy()
    t_wb = day["t_wb"].to_numpy(); u_off = day["u_demand"].to_numpy()
    u_now = c.alpha_interactive * u_off
    arrival = (1.0 - c.alpha_interactive) * u_off * DT_H
    D = int(c.deadline_h / DT_H)
    cop = np.array([cop_at(c, t_sup[t], t_wb[t]) for t in range(T)])
    p_fan = c.fan_frac * c.p_it_max_kw
    P_span = c.p_it_max_kw - c.p_it_idle_kw
    P_base = c.p_it_idle_kw + c.p_aux_frac * c.p_it_max_kw

    m = pulp.LpProblem("oracle3", pulp.LpMinimize)
    seg_w = [SPEC_X[i + 1] - SPEC_X[i] for i in range(3)]
    seg_m = [(SPEC_Y[i + 1] - SPEC_Y[i]) / seg_w[i] for i in range(3)]
    d = [[pulp.LpVariable(f"d{i}_{t}", lowBound=0, upBound=seg_w[i]) for i in range(3)] for t in range(T)]
    b = [[pulp.LpVariable(f"b{i}_{t}", cat="Binary") for i in range(2)] for t in range(T)]
    for t in range(T):
        m += d[t][0] >= seg_w[0] * b[t][0]
        m += d[t][1] <= seg_w[1] * b[t][0]
        m += d[t][1] >= seg_w[1] * b[t][1]
        m += d[t][2] <= seg_w[2] * b[t][1]
    u_exec = [pulp.lpSum(d[t]) for t in range(T)]
    p_it = [P_base + P_span * pulp.lpSum(seg_m[i] * d[t][i] for i in range(3)) for t in range(T)]
    p_load = [p_it[t] * (1.0 + 1.0 / cop[t]) + p_fan for t in range(T)]

    served = [pulp.LpVariable(f"s_{t}", lowBound=0, upBound=None) for t in range(T)]
    for t in range(T):
        m += u_exec[t] >= u_now[t]
        m += u_exec[t] <= c.u_cap
        m += served[t] == (u_exec[t] - u_now[t]) * DT_H
        m += pulp.lpSum(served[:t + 1]) <= float(arrival[:t + 1].sum())
        m += pulp.lpSum(served[:t + 1]) >= float(arrival[:max(t - D + 1, 0)].sum())
    # NOTE: no end-of-day "serve everything" equality. The environment resets
    # at midnight and work that arrived in the last deadline window is neither
    # served nor penalised in cost; every controller (rules and RL) benefits from
    # this horizon effect equally, so the oracle must be given the same freedom
    # or it is over-constrained and loses to a rule (which is how this was found).

    ch = [pulp.LpVariable(f"ch_{t}", lowBound=0, upBound=c.bess_power_kw) for t in range(T)]
    dis = [pulp.LpVariable(f"di_{t}", lowBound=0, upBound=c.bess_power_kw) for t in range(T)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=c.soc_min, upBound=c.soc_max) for t in range(T)]
    imp = [pulp.LpVariable(f"im_{t}", lowBound=0, upBound=None) for t in range(T)]
    exp = [pulp.LpVariable(f"ex_{t}", lowBound=0, upBound=None) for t in range(T)]
    pk = pulp.LpVariable("peak", lowBound=0, upBound=None)
    for t in range(T):
        prev = c.soc_init if t == 0 else soc[t - 1]
        m += soc[t] == prev + (ch[t] * c.eta_ch - dis[t] * (1.0 / c.eta_dis)) * (DT_H / c.bess_energy_kwh)
        m += imp[t] - exp[t] == p_load[t] - pv[t] + ch[t] - dis[t]
        m += imp[t] <= c.pcc_limit_kw
        m += pk >= imp[t]
    price_ref = float(price.mean())
    m += (pulp.lpSum(price[t] * imp[t] * DT_H - price[t] * c.export_price_frac * exp[t] * DT_H
                     + c.degrade_cost * (ch[t] + dis[t]) * DT_H for t in range(T))
          + c.peak_charge * pk / 30.0 * c.episode_days
          + (c.soc_init - soc[T - 1]) * c.bess_energy_kwh * price_ref
          + (float(arrival.sum()) - pulp.lpSum(served)) * P_span * ((1.0 - 0.5704) / 0.5) * price_ref
          * (1.0 if c.terminal_backlog else 0.0))
    m.solve(pulp.PULP_CBC_CMD(msg=msg))
    if pulp.LpStatus[m.status] != "Optimal":
        return None
    p_it_v = np.array([P_base + P_span * sum(seg_m[i] * d[t][i].value() for i in range(3)) for t in range(T)])
    return dict(cost=float(pulp.value(m.objective)), p_it=p_it_v,
                served=np.array([served[t].value() for t in range(T)]),
                ch=np.array([ch[t].value() for t in range(T)]),
                dis=np.array([dis[t].value() for t in range(T)]))


def solve_joint(c, day, t_bar, p_bar, t_room0, msg=False):
    """Dispatch + battery + CONTINUOUS set-point MILP with the thermal coupling
    exact (linear) and 1/COP linearised around (t_bar, p_bar) per step.
    Sequential linearisation: call repeatedly, updating t_bar/p_bar."""
    T = len(day)
    price = day["price"].to_numpy(); pv = day["pv_kw"].to_numpy()
    t_wb = day["t_wb"].to_numpy(); u_off = day["u_demand"].to_numpy()
    u_now = c.alpha_interactive * u_off
    arrival = (1.0 - c.alpha_interactive) * u_off * DT_H
    D = int(c.deadline_h / DT_H)
    p_fan = c.fan_frac * c.p_it_max_kw
    P_span = c.p_it_max_kw - c.p_it_idle_kw
    P_base = c.p_it_idle_kw + c.p_aux_frac * c.p_it_max_kw
    tau = c.c_room_kwh_per_k / c.k_cool_kw_per_k
    decay = float(np.exp(-DT_H / tau))
    # 1/COP and its slope in t_sup at the linearisation point (finite difference)
    h0 = np.array([1.0 / cop_at(c, t_bar[t], t_wb[t]) for t in range(T)])
    eps = 0.25
    h1 = np.array([(1.0 / cop_at(c, min(t_bar[t] + eps, c.t_sup_max), t_wb[t])
                    - 1.0 / cop_at(c, max(t_bar[t] - eps, c.t_sup_min), t_wb[t])) / (2 * eps)
                   for t in range(T)])

    m = pulp.LpProblem("oracle3j", pulp.LpMinimize)
    seg_w = [SPEC_X[i + 1] - SPEC_X[i] for i in range(3)]
    seg_m = [(SPEC_Y[i + 1] - SPEC_Y[i]) / seg_w[i] for i in range(3)]
    d = [[pulp.LpVariable(f"d{i}_{t}", lowBound=0, upBound=seg_w[i]) for i in range(3)] for t in range(T)]
    b = [[pulp.LpVariable(f"b{i}_{t}", cat="Binary") for i in range(2)] for t in range(T)]
    for t in range(T):
        m += d[t][0] >= seg_w[0] * b[t][0]
        m += d[t][1] <= seg_w[1] * b[t][0]
        m += d[t][1] >= seg_w[1] * b[t][1]
        m += d[t][2] <= seg_w[2] * b[t][1]
    u_exec = [pulp.lpSum(d[t]) for t in range(T)]
    p_it = [P_base + P_span * pulp.lpSum(seg_m[i] * d[t][i] for i in range(3)) for t in range(T)]

    t_sup = [pulp.LpVariable(f"ts_{t}", lowBound=c.t_sup_min, upBound=c.t_sup_max) for t in range(T)]
    t_room = [pulp.LpVariable(f"tr_{t}", lowBound=10.0, upBound=c.t_room_max) for t in range(T + 1)]
    m += t_room[0] == t_room0
    for t in range(T):
        t_inf = t_sup[t] + p_it[t] * (1.0 / c.k_cool_kw_per_k)
        m += t_room[t + 1] == decay * t_room[t] + (1 - decay) * t_inf   # end-of-step <= max via bounds
    # linearised cooling power: p_it*h(t) ~ p_it*h0 + h1*p_bar*(t - t_bar)
    p_cool = [p_it[t] * h0[t] + h1[t] * p_bar[t] * (t_sup[t] - t_bar[t]) + p_fan for t in range(T)]
    p_load = [p_it[t] + p_cool[t] for t in range(T)]

    served = [pulp.LpVariable(f"s_{t}", lowBound=0, upBound=None) for t in range(T)]
    for t in range(T):
        m += u_exec[t] >= u_now[t]
        m += u_exec[t] <= c.u_cap
        m += served[t] == (u_exec[t] - u_now[t]) * DT_H
        m += pulp.lpSum(served[:t + 1]) <= float(arrival[:t + 1].sum())
        m += pulp.lpSum(served[:t + 1]) >= float(arrival[:max(t - D + 1, 0)].sum())
    # NOTE: no end-of-day "serve everything" equality. The environment resets
    # at midnight and work that arrived in the last deadline window is neither
    # served nor penalised in cost; every controller (rules and RL) benefits from
    # this horizon effect equally, so the oracle must be given the same freedom
    # or it is over-constrained and loses to a rule (which is how this was found).

    ch = [pulp.LpVariable(f"ch_{t}", lowBound=0, upBound=c.bess_power_kw) for t in range(T)]
    dis = [pulp.LpVariable(f"di_{t}", lowBound=0, upBound=c.bess_power_kw) for t in range(T)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=c.soc_min, upBound=c.soc_max) for t in range(T)]
    imp = [pulp.LpVariable(f"im_{t}", lowBound=0, upBound=None) for t in range(T)]
    exp = [pulp.LpVariable(f"ex_{t}", lowBound=0, upBound=None) for t in range(T)]
    pk = pulp.LpVariable("peak", lowBound=0, upBound=None)
    for t in range(T):
        prev = c.soc_init if t == 0 else soc[t - 1]
        m += soc[t] == prev + (ch[t] * c.eta_ch - dis[t] * (1.0 / c.eta_dis)) * (DT_H / c.bess_energy_kwh)
        m += imp[t] - exp[t] == p_load[t] - pv[t] + ch[t] - dis[t]
        m += imp[t] <= c.pcc_limit_kw
        m += pk >= imp[t]
    price_ref = float(price.mean())
    m += (pulp.lpSum(price[t] * imp[t] * DT_H - price[t] * c.export_price_frac * exp[t] * DT_H
                     + c.degrade_cost * (ch[t] + dis[t]) * DT_H for t in range(T))
          + c.peak_charge * pk / 30.0 * c.episode_days
          + (c.soc_init - soc[T - 1]) * c.bess_energy_kwh * price_ref
          + (float(arrival.sum()) - pulp.lpSum(served)) * P_span * ((1.0 - 0.5704) / 0.5) * price_ref
          * (1.0 if c.terminal_backlog else 0.0))
    m.solve(pulp.PULP_CBC_CMD(msg=msg))
    if pulp.LpStatus[m.status] != "Optimal":
        return None
    p_it_v = np.array([P_base + P_span * sum(seg_m[i] * d[t][i].value() for i in range(3)) for t in range(T)])
    return dict(cost=float(pulp.value(m.objective)), p_it=p_it_v,
                t_sup=np.array([t_sup[t].value() for t in range(T)]),
                served=np.array([served[t].value() for t in range(T)]),
                ch=np.array([ch[t].value() for t in range(T)]),
                dis=np.array([dis[t].value() for t in range(T)]))


def replay(env, e, sol, c, clip_tsup=False):
    """Drive the environment with the oracle's schedule; return env.summary()."""
    env.start_index = e * env.ep_len
    env.reset(seed=1000 + e)
    day = env._arr["u_demand"][env.t0: env.t0 + env.ep_len]
    done = False; k = 0
    while not done:
        arrival = (1.0 - c.alpha_interactive) * day[k] * DT_H
        backlog_total = sum(b[0] for b in env.backlog) + arrival
        phi = float(np.clip(sol["served"][k] / max(backlog_total, 1e-9), 0.0, 1.0))
        bess = float(np.clip((sol["dis"][k] - sol["ch"][k]) / c.bess_power_kw, -1.0, 1.0))
        if clip_tsup or "t_sup" not in sol:
            a2 = 1.0                                          # at bound; projection clips
        else:
            a2 = 2.0 * (sol["t_sup"][k] - c.t_sup_min) / (c.t_sup_max - c.t_sup_min) - 1.0
        a = np.array([2 * phi - 1, bess, a2], dtype=np.float32)
        _, _, term, trunc, _ = env.step(a)
        done = term or trunc; k += 1
    return env.summary()


def main():
    from baselines import REGISTRY, run_controller
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--idle-frac", type=float, default=None)
    ap.add_argument("--peak-charge", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--baselines", default="B0,B4,B6")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tr, te = build_profiles(profiles_csv=args.profiles)
    kw = {}
    if args.idle_frac is not None:
        kw["p_idle_w"] = args.idle_frac * DCConfig().p_max_w
    if args.peak_charge is not None:
        kw["peak_charge"] = args.peak_charge
    cfg = calibrate(DCConfig(sla_floor=False, **kw), profiles=tr, n_episodes=12)
    env = DCMicrogridEnv(cfg=cfg, profiles=te, seed=0, eval_mode=True)
    env.reset(seed=1000); t0 = env.t_room
    base = {k: pd.DataFrame(run_controller(REGISTRY[k], env, n_episodes=args.episodes))
            for k in args.baselines.split(",")}

    rows = []
    for e in range(args.episodes):
        day = te.iloc[e * STEPS_PER_DAY:(e + 1) * STEPS_PER_DAY]
        # round 0: fixed-set-point solve gives a starting (t_bar, p_bar);
        # then sequential linearisation of the joint problem.
        t_bar = np.full(STEPS_PER_DAY, 24.0)
        sol = solve_fixed_tsup(cfg, day, t_bar)
        if sol is None:
            print(f"ep {e}: infeasible", flush=True); continue
        t_bar = max_feasible_tsup(cfg, sol["p_it"], t0); p_bar = sol["p_it"]
        best = None
        for r in range(args.rounds):
            new = solve_joint(cfg, day, t_bar, p_bar, t0)
            if new is None:
                break
            s_try = replay(env, e, new, cfg)
            if best is None or s_try["cost"] < best[0]["cost"]:
                best = (s_try, new)
            if np.abs(new["t_sup"] - t_bar).max() < 0.05:
                break
            t_bar, p_bar = new["t_sup"], new["p_it"]
        # also try the fixed-set-point schedule with the projection clipping
        s_fix = replay(env, e, sol, cfg, clip_tsup=True)
        if best is None or s_fix["cost"] < best[0]["cost"]:
            best = (s_fix, sol)
        s, sol = best
        t_sup = sol.get("t_sup", t_bar)
        row = dict(episode=e, milp_cost=sol["cost"], cost=s["cost"], carbon=s["carbon_kg"],
                   peak_kw=s["peak_kw"], sla_missed=s["sla_missed"],
                   thermal_viol=s["thermal_viol_rate"], t_sup_mean=float(t_sup.mean()))
        for k in base:
            row[f"{k}_cost"] = float(base[k].cost.iloc[e])
        rows.append(row)
        print(f"ep {e:3d}: milp {sol['cost']:8.1f} replay {s['cost']:8.1f}  B0 {row['B0_cost']:8.1f} "
              f"gap {100*(s['cost']/row['B0_cost']-1):6.2f}%  t_sup {t_sup.mean():.1f} C  "
              f"sla {s['sla_missed']:.3f} therm {s['thermal_viol_rate']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "oracle3_paired.csv"), index=False)
    print(f"\n=== simulator-verified oracle, {len(df)} days ===")
    for k in base:
        g = 100 * (df.cost / df[f"{k}_cost"] - 1)
        print(f"oracle vs {k}: paired {g.mean():6.2f}% +/- {1.96*g.std(ddof=1)/np.sqrt(len(g)):.2f}"
              f"   bill {100*(df.cost.sum()/df[f'{k}_cost'].sum()-1):6.2f}%")


if __name__ == "__main__":
    main()
