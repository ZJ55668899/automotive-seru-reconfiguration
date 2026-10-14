"""Fixture-share sensitivity experiment (Section 5.10).

Paired design: one base instance per seed, re-drawn only in its fixture mask
(15% / 30% / 45% of stations with fixed tooling; masks are nested).  Fixed
stations keep their incumbent seru membership (constraints.fixtures_anchored).

Two measurements per (instance, share):
  1. feasible-neighbourhood shrinkage: a random feasible walk from X_prev,
     counting how many C1/C4-feasible candidates the anchor rule rejects;
  2. search outcome: short surrogate-assisted NSGA-III run; hypervolume of
     the DES-rechecked front (normalised per base instance over the union of
     the three shares) and mean stall ratio.

Usage:  python examples/fixture_share_sensitivity.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from seru_core import (Config, SearchConfig, fixtures_anchored, in_omega_st,
                       initial_config, make_instance,
                       surrogate_assisted_nsga3)

SHARES = (0.15, 0.30, 0.45)
BASE_SEEDS = (101, 202, 303)
N_STATIONS = 21


def hv_normalized(front_objs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Hypervolume of pre-normalised objectives against reference (1.1,)^3."""
    span = np.maximum(hi - lo, 1e-12)
    on = (front_objs - lo) / span
    ref = 1.1
    order = np.argsort(on[:, 2])
    hv, prev_z = 0.0, 0.0
    for idx in order[::-1]:
        z = on[idx, 2]
        band = ref - z
        if band < prev_z:
            break
        sub = on[on[:, 2] <= z]
        s = sub[np.argsort(sub[:, 0])]
        area, best_y = 0.0, ref
        for pt in s:
            if pt[1] < best_y:
                area += (ref - pt[0]) * (best_y - pt[1])
                best_y = pt[1]
        hv += area * (band - prev_z)
        prev_z = band
    return float(hv)


def neighbourhood_acceptance(inst, x_prev, rng, steps: int = 400):
    """Random feasible walk from X_prev.  Returns (n_c1c4_feasible,
    n_anchor_rejected): how many generated candidates satisfied C1/C4, and
    how many of those the fixture-anchor rule then rejects — a direct measure
    of prior-neighbourhood shrinkage."""
    from seru_core import merge, reassign, split
    cfg = x_prev
    feas = rej_anchor = 0
    for _ in range(steps):
        op = int(rng.integers(3))
        if op == 0 and cfg.n_seru > 1:
            a, b = rng.choice(cfg.n_seru, 2, replace=False)
            out = merge(inst, cfg, int(a), int(b))
        elif op == 1:
            out = split(inst, cfg, int(rng.integers(cfg.n_seru)), rng)
        else:
            out = reassign(inst, cfg, int(rng.integers(cfg.n_seru)), rng)
        if out is None or not in_omega_st(inst, out):
            continue
        feas += 1
        if not fixtures_anchored(inst, out, x_prev):
            rej_anchor += 1
            continue                    # anchored walk rejects the candidate
        cfg = out
    return feas, rej_anchor


def main() -> None:
    rows = []
    for base_seed in BASE_SEEDS:
        fronts: dict[float, np.ndarray] = {}
        per_share: dict[float, dict] = {}
        for share in SHARES:
            t0 = time.time()
            inst = make_instance(N_STATIONS, "M", base_seed,
                                 name=f"M21-s{base_seed}-f{share:.2f}",
                                 fixture_share=share)
            labels, team_of = initial_config(inst, inst.rng)
            x_prev = Config.from_labels(labels, team_of)
            rng = np.random.default_rng(7)
            feas_c, rej_a = neighbourhood_acceptance(inst, x_prev, rng)

            sc = SearchConfig(pop_size=20, max_gen=25, warmup_gen=10_000,
                              check_every=5, final_reps=3, g_ctrl=20)
            res = surrogate_assisted_nsga3(inst, x_prev, sim_seed=11,
                                           alg_seed=1, sc=sc)
            fronts[share] = res.objectives
            per_share[share] = dict(
                n_fixture=int(inst.fixtures.sum()),
                neigh_feas=feas_c, neigh_rej=rej_a,
                rho=float(res.rho.mean(1).mean()),
                exec_share=float(res.executable.mean()),
                n_des=res.n_des_calls, wall_s=time.time() - t0,
            )
            print(f"seed {base_seed} share {share:.2f}: "
                  f"fixtures={per_share[share]['n_fixture']}, "
                  f"neigh {rej_a}/{feas_c} anchor-rejected, "
                  f"rho={per_share[share]['rho']:.4f}, "
                  f"exec={per_share[share]['exec_share']:.2f}, "
                  f"des={res.n_des_calls}, {per_share[share]['wall_s']:.0f}s",
                  flush=True)
        # common normalisation per base instance over the three shares
        all_obj = np.vstack(list(fronts.values()))
        lo, hi = all_obj.min(0), all_obj.max(0)
        for share in SHARES:
            hv = hv_normalized(fronts[share], lo, hi)
            d = per_share[share]
            rows.append((base_seed, share, d["n_fixture"],
                         d["neigh_feas"], d["neigh_rej"],
                         round(hv, 4), round(d["rho"], 4),
                         round(d["exec_share"], 4), d["n_des"],
                         round(d["wall_s"], 1)))
            print(f"  -> seed {base_seed} share {share:.2f}: HV={hv:.4f}",
                  flush=True)

    out = pathlib.Path(__file__).with_name("fixture_share_sensitivity.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("base_seed,fixture_share,n_fixture_stations,"
                "neigh_c1c4_feasible,neigh_anchor_rejected,"
                "hv_norm,rho_mean,exec_share,n_des_calls,wall_s\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
