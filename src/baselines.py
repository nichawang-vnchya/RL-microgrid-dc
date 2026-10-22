"""
Baseline controllers B0-B5 for the DC-microgrid environment.

Every controller returns an action in the environment's [-1, 1]^3 space:
    a[0] -> dispatch fraction of the deferrable backlog
    a[1] -> battery power  (+ discharge, - charge)
    a[2] -> CRAC supply-air setpoint

See docs/research_design.md §5. The MILP oracle lives in oracle.py / oracle2.py.
"""
from __future__ import annotations

import numpy as np

from profiles import DT_H


def _map_tsup(env, t_c: float) -> float:
    lo, hi = env.cfg.t_sup_min, env.cfg.t_sup_max
    return float(np.clip(2.0 * (t_c - lo) / (hi - lo) - 1.0, -1.0, 1.0))


class Controller:
    name = "base"

    def __init__(self, env):
        self.env = env

    def act(self, obs, env) -> np.ndarray:
        raise NotImplementedError

    # --- shared helpers -------------------------------------------------
    @staticmethod
    def _idx(env) -> int:
        return min(env.t0 + env.k, env._n - 1)

    @staticmethod
    def _urgent(env) -> float:
        """Dispatch fraction needed so nothing misses its deadline this step."""
        if not env.backlog:
            return 0.0
        total = sum(b[0] for b in env.backlog)
        soon = sum(b[0] for b in env.backlog
                   if (b[1] - env.k) * DT_H <= env.cfg.deadline_h * 0.34)
        return float(np.clip(soon / max(total, 1e-9), 0.0, 1.0))

    @staticmethod
    def _thermal_guard(env, t_target: float) -> float:
        """Clamp the setpoint down if the room is close to the ASHRAE limit."""
        slack = env.cfg.t_room_max - env.t_room
        if slack < 1.0:
            t_target -= (1.0 - slack) * 3.0
        return float(np.clip(t_target, env.cfg.t_sup_min, env.cfg.t_sup_max))


class StatusQuo(Controller):
    """B0 -- fixed 22 degC setpoint, no deferral (run everything immediately),
    battery idle. The reference against which all savings are reported."""
    name = "B0 status-quo"

    def act(self, obs, env):
        return np.array([1.0, 0.0, _map_tsup(env, 22.0)], dtype=np.float32)


class GreedySolar(Controller):
    """B1 -- run backlog whenever there is PV surplus; charge the battery on
    surplus and discharge it when PV cannot cover the load."""
    name = "B1 greedy-solar"

    def act(self, obs, env):
        i = self._idx(env)
        pv = env._arr["pv_kw"][i]
        # Load estimate from the previous realised step (a real controller has
        # this from its meter); fall back to the nameplate estimate on step 0.
        if env.log:
            load_est = env.log[-1]["p_it"] + env.log[-1]["p_cool"]
        else:
            load_est = env.cfg.p_it_max_kw * 0.85
        surplus = pv - load_est
        sun = pv > 0.05 * env.cfg.pv_capacity_kw
        phi = 1.0 if sun else self._urgent(env)
        if surplus > 0:                       # bank the excess
            bess = -min(surplus, env.cfg.bess_power_kw) / env.cfg.bess_power_kw
        elif not sun:                         # cover the night from storage
            bess = min(-surplus, env.cfg.bess_power_kw) / env.cfg.bess_power_kw
        else:
            bess = 0.0
        t = self._thermal_guard(env, 25.0)
        return np.array([2 * phi - 1, np.clip(bess, -1, 1), _map_tsup(env, t)],
                        dtype=np.float32)


class PriceThreshold(Controller):
    """B2 -- defer when the price is above its trailing median; discharge the
    battery on-peak and charge off-peak. This is the heuristic Sukprasert et al.
    (EuroSys '24) say already captures most of the achievable benefit."""
    name = "B2 price-threshold"

    def __init__(self, env, window_h: float = 24.0, q_hi: float = 0.65, q_lo: float = 0.35):
        super().__init__(env)
        self.n = int(window_h / DT_H)
        self.q_hi, self.q_lo = q_hi, q_lo

    def act(self, obs, env):
        i = self._idx(env)
        hist = env._arr["price"][max(0, i - self.n): i + 1]
        p = env._arr["price"][i]
        hi, lo = np.quantile(hist, self.q_hi), np.quantile(hist, self.q_lo)
        if p >= hi:
            phi, bess = self._urgent(env), 0.7
        elif p <= lo:
            # charge at part power: full-power charging creates a new peak that
            # costs more than the arbitrage earns.
            phi, bess = 1.0, -0.5
        else:
            phi, bess = 0.5, 0.0
        t = self._thermal_guard(env, 26.0 if p >= hi else 24.0)
        return np.array([2 * phi - 1, bess, _map_tsup(env, t)], dtype=np.float32)


class CarbonThreshold(PriceThreshold):
    """B3 -- identical logic on the carbon-intensity signal."""
    name = "B3 carbon-threshold"

    def act(self, obs, env):
        i = self._idx(env)
        hist = env._arr["carbon"][max(0, i - self.n): i + 1]
        cq = env._arr["carbon"][i]
        hi, lo = np.quantile(hist, self.q_hi), np.quantile(hist, self.q_lo)
        if cq >= hi:
            phi, bess = self._urgent(env), 0.7
        elif cq <= lo:
            phi, bess = 1.0, -0.5
        else:
            phi, bess = 0.5, 0.0
        t = self._thermal_guard(env, 26.0 if cq >= hi else 24.0)
        return np.array([2 * phi - 1, bess, _map_tsup(env, t)], dtype=np.float32)


class TOUWarm(PriceThreshold):
    """B6 -- the rule a TOU-tariff operator would actually write: serve only
    deadline-urgent work on-peak and everything off-peak, hold the supply-air
    set-point at 26 degC (the thermal guard / projection clamps it near the
    ASHRAE limit), and leave the battery idle. The battery is idle on purpose:
    under a 1.5x TOU ratio the round-trip loss and degradation cost cancel the
    arbitrage margin, and any charging raises the billed peak. Added because,
    under a constant grid emission factor, B3 degenerates into this warm +
    defer behaviour by accident; B6 states it explicitly and honestly."""
    name = "B6 TOU-warm"

    def act(self, obs, env):
        i = self._idx(env)
        hist = env._arr["price"][max(0, i - self.n): i + 1]
        p = env._arr["price"][i]
        hi = np.quantile(hist, self.q_hi)
        phi = self._urgent(env) if p >= hi else 1.0
        t = self._thermal_guard(env, 26.0)
        return np.array([2 * phi - 1, 0.0, _map_tsup(env, t)], dtype=np.float32)


class RuleBasedCooling(Controller):
    """B4 -- standard supply-air-temperature reset from ambient wet-bulb.
    No workload deferral, no battery. Isolates the cooling-only contribution."""
    name = "B4 cooling-reset"

    def act(self, obs, env):
        i = self._idx(env)
        t_wb = env._arr["t_wb"][i]
        t = self._thermal_guard(env, np.clip(18.0 + 0.45 * (t_wb - 12.0), 18.0, 26.0))
        return np.array([1.0, 0.0, _map_tsup(env, t)], dtype=np.float32)


class MPC(Controller):
    """B5 -- receding-horizon controller with an imperfect 4-hour forecast.

    Enumerates a small action grid, rolls a light-weight internal model forward
    over the horizon, and applies the first action of the best sequence
    (constant-action approximation). Deliberately uses the same noisy forecast
    the RL agent sees, so the comparison is fair.
    """
    name = "B5 MPC-4h"

    def __init__(self, env, horizon_h: float = 4.0, n_phi: int = 4,
                 n_bess: int = 5, n_t: int = 4):
        super().__init__(env)
        self.H = int(horizon_h / DT_H)
        self.grid = [(p, b, t)
                     for p in np.linspace(0, 1, n_phi)
                     for b in np.linspace(-1, 1, n_bess)
                     for t in np.linspace(env.cfg.t_sup_min, env.cfg.t_sup_max, n_t)]

    def act(self, obs, env):
        c = env.cfg
        i = self._idx(env)
        H = min(self.H, env.ep_len - env.k, env._n - i - 1)
        if H <= 0:
            return np.array([1.0, 0.0, _map_tsup(env, 22.0)], dtype=np.float32)

        rng = np.random.default_rng(env.t0 * 1000 + env.k)
        err = np.exp(rng.normal(0, c.forecast_err_std, H))
        pv = env._arr["pv_kw"][i:i + H] * err
        pr = env._arr["price"][i:i + H] * err
        ci = env._arr["carbon"][i:i + H] * err
        wb = env._arr["t_wb"][i:i + H]
        ud = env._arr["u_demand"][i:i + H]

        backlog0 = sum(b[0] for b in env.backlog)
        # Opportunity cost of stored energy ("water value"): the trailing 24-h mean
        # price. Valuing storage at the *horizon* mean instead would make a 4-h MPC
        # indifferent to arbitrage and it would simply drain the battery.
        w24 = int(24 / DT_H)
        water_value = float(np.mean(env._arr["price"][max(0, i - w24): i + 1]))
        best, best_c = None, np.inf
        tau = c.c_room_kwh_per_k / c.k_cool_kw_per_k
        decay = np.exp(-DT_H / tau)

        for phi0, bcmd0, tsup0 in self.grid:
            soc, troom, back = env.soc, env.t_room, backlog0
            peak = env.peak_kw
            tot = 0.0
            ok = True
            for h in range(H):
                # Rollout MPC: the candidate action is applied at h = 0; the rest of
                # the horizon runs a fixed base policy (serve everything, hold the
                # battery, run the setpoint as warm as the envelope allows). This is
                # far stronger than assuming the candidate action is held constant.
                if h == 0:
                    phi, bcmd, tsup = phi0, bcmd0, tsup0
                else:
                    phi, bcmd = 1.0, 0.0
                    tsup = c.t_sup_max
                u_now = c.alpha_interactive * ud[h]
                back += (1 - c.alpha_interactive) * ud[h] * DT_H
                served = min(phi * back, max(c.u_cap - u_now, 0) * DT_H, back)
                back -= served
                u = np.clip(u_now + served / DT_H, 0, c.u_cap)
                p_it = (c.p_it_idle_kw + (c.p_it_max_kw - c.p_it_idle_kw)
                        * env._spec_curve(u) + c.p_aux_frac * c.p_it_max_kw)
                if c.safety_layer:   # mirror the environment's action projection
                    tsup = min(tsup, (c.t_room_max - troom * decay) / (1 - decay)
                               - p_it / c.k_cool_kw_per_k)
                    tsup = float(np.clip(tsup, c.t_sup_min, c.t_sup_max))
                t_inf = tsup + p_it / c.k_cool_kw_per_k
                troom = t_inf + (troom - t_inf) * decay
                if troom > c.t_room_max + 0.5:
                    ok = False
                    break
                if wb[h] < tsup - c.econ_delta_t:
                    cop = c.cop_free
                else:
                    cop = np.clip(c.cop_nom * (1 + c.cop_gamma * (tsup - c.t_sup_ref))
                                  * (1 - c.cop_delta * (wb[h] - c.t_wb_ref)),
                                  c.cop_min, c.cop_max)
                p_cool = p_it / cop + c.fan_frac * c.p_it_max_kw
                if bcmd >= 0:
                    p_dis = min(bcmd * c.bess_power_kw, c.bess_power_kw,
                                (soc - c.soc_min) * c.bess_energy_kwh * c.eta_dis / DT_H)
                    p_ch = 0.0
                    soc -= p_dis * DT_H / (c.eta_dis * c.bess_energy_kwh)
                else:
                    p_ch = min(-bcmd * c.bess_power_kw, c.bess_power_kw,
                               (c.soc_max - soc) * c.bess_energy_kwh / c.eta_ch / DT_H)
                    p_dis = 0.0
                    soc += p_ch * DT_H * c.eta_ch / c.bess_energy_kwh
                imp = max(p_it + p_cool - pv[h] + p_ch - p_dis, 0.0)
                peak = max(peak, imp)
                tot += (pr[h] * imp * DT_H * env.w[0] / max(c.norm["cost"], 1e-9)
                        + ci[h] * imp * DT_H * env.w[1] / max(c.norm["carbon"], 1e-9))
            if not ok:
                continue
            # terminal backlog cost, the demand charge the horizon would lock in,
            # and the value of the energy left in (or taken out of) the battery --
            # without the last term the controller drains the battery for free.
            tot += env.w[2] * back * 0.5 / max(c.norm["sla"], 1e-9)
            tot += env.w[0] * c.peak_charge * peak / 30.0 / max(c.norm["cost"], 1e-9)
            tot += (env.w[0] * (env.soc - soc) * c.bess_energy_kwh * water_value
                    / max(c.norm["cost"], 1e-9))
            if tot < best_c:
                best_c, best = tot, (phi0, bcmd0, tsup0)

        if best is None:
            return np.array([1.0, 0.0, _map_tsup(env, 20.0)], dtype=np.float32)
        phi, bcmd, tsup = best
        phi = max(phi, self._urgent(env))          # never let a deadline slip
        tsup = self._thermal_guard(env, tsup)
        return np.array([2 * phi - 1, bcmd, _map_tsup(env, tsup)], dtype=np.float32)


REGISTRY = {
    "B0": StatusQuo,
    "B1": GreedySolar,
    "B2": PriceThreshold,
    "B3": CarbonThreshold,
    "B4": RuleBasedCooling,
    "B5": MPC,
    "B6": TOUWarm,
}


def run_controller(ctrl_cls, env, n_episodes: int = 10, seed: int = 0) -> list[dict]:
    """Evaluate a controller over consecutive held-out episodes."""
    out = []
    pol = ctrl_cls(env)
    for e in range(n_episodes):
        env.start_index = e * env.ep_len
        obs, _ = env.reset(seed=seed + e)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(pol.act(obs, env))
            done = term or trunc
        s = env.summary()
        s["episode"] = e
        s["controller"] = pol.name
        out.append(s)
    return out
