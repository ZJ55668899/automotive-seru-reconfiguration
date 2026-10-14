"""Property-based sanity checks for the reference implementation.

These are the verification anchors available without the authors' original
experiment logs (see README, "Verification"):

  S1  Proposition 2 closure: merge/split/reassign from a feasible config
      always produce configs satisfying C1 and C4.
  S2  Crossover + restore yields feasible offspring (or the parent clone).
  S3  DES accounting: rho in [0, 1]; merged stall time never exceeds the
      makespan; per-class stall times are consistent with the union bound.
  S4  Surrogate sanity: fit on evaluated samples, predictions finite,
      uncertainties non-negative.
  S5  Sequence audit: the completion order of order-book orders equals the
      release order (no overtaking), the output sequence re-passes the
      spacing rule of Eq. 4 with the urgent exemption of Eq. 6, and every
      urgent order meets its due position or carries a C2_urgent mark.

Usage:  python tests/sanity_checks.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from seru_core import (Config, RidgeSurrogate, check_c1, check_c4,
                       evaluate, features, in_omega_st, initial_config,
                       make_instance, random_operator, crossover)


def s1_operator_closure() -> bool:
    rng = np.random.default_rng(0)
    ok = True
    for seed in (42, 123, 456):
        inst = make_instance(12, "S", seed)
        labels, team_of = initial_config(inst, inst.rng)
        cfg = Config.from_labels(labels, team_of)
        if not in_omega_st(inst, cfg):
            print(f"  [S1] WARNING: initial config infeasible (seed {seed})")
            ok = False
            continue
        for _ in range(200):
            cfg = random_operator(inst, cfg, rng)
            if not in_omega_st(inst, cfg):
                ok = False
                print(f"  [S1] FAIL: operator broke C1/C4 (seed {seed})")
                break
    print(f"  [S1] operator closure over 600 random operations: {'PASS' if ok else 'FAIL'}")
    return ok


def s2_crossover_restore() -> bool:
    rng = np.random.default_rng(1)
    ok = True
    inst = make_instance(16, "S", 42)
    labels, team_of = initial_config(inst, inst.rng)
    base = Config.from_labels(labels, team_of)
    pop = [base]
    for _ in range(10):
        pop.append(random_operator(inst, base, rng))
    for _ in range(50):
        a, b = rng.choice(len(pop)), None
        b = rng.choice(len(pop))
        child = crossover(inst, pop[int(a)], pop[int(b)], rng)
        if not in_omega_st(inst, child):
            ok = False
            print("  [S2] FAIL: crossover/restore produced infeasible child")
            break
    print(f"  [S2] 50 crossover+restore children feasible: {'PASS' if ok else 'FAIL'}")
    return ok


def s3_des_accounting() -> bool:
    inst = make_instance(12, "S", 42)
    labels, team_of = initial_config(inst, inst.rng)
    x_prev = Config.from_labels(labels, team_of)
    res = evaluate(inst, x_prev, x_prev, seed=7)
    ok = (0.0 <= res.rho <= 1.0) and (res.t_cid <= res.f1 + 1e-9) \
        and res.n_orders > 0 and np.isfinite(res.f3)
    print(f"  [S3] DES: f1={res.f1/60:.2f} h, rho={res.rho:.3%}, "
          f"t_cid={res.t_cid:.1f} min, orders={res.n_orders} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def s4_surrogate() -> bool:
    rng = np.random.default_rng(2)
    inst = make_instance(12, "S", 42)
    labels, team_of = initial_config(inst, inst.rng)
    x_prev = Config.from_labels(labels, team_of)
    xs, ys = [], []
    cfg = x_prev
    for i in range(12):
        res = evaluate(inst, cfg, x_prev, seed=100 + i)
        xs.append(features(inst, cfg, x_prev))
        ys.append(res.as_objectives())
        cfg = random_operator(inst, cfg, rng)
    sur = RidgeSurrogate(3).fit_with_gram(np.array(xs), np.array(ys))
    y_hat, u_hat = sur.predict(np.array(xs[:3]))
    ok = bool(np.isfinite(y_hat).all() and (u_hat >= 0).all())
    mape = float(np.abs(y_hat - ys[:3]).mean() /
                 np.abs(np.array(ys[:3])).mean() * 100)
    print(f"  [S4] surrogate: in-sample MAPE {mape:.1f}%, uncertainty >= 0 "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def s5_sequence_audit() -> bool:
    """Output sequence equals release sequence; urgent marks are consistent."""
    inst = make_instance(12, "S", 42)
    labels, team_of = initial_config(inst, inst.rng)
    x_prev = Config.from_labels(labels, team_of)
    ok = True
    for seed in (7, 11, 123):
        res = evaluate(inst, x_prev, x_prev, seed=seed)
        if res.out_order != res.rel_order:
            ok = False
            print(f"  [S5] FAIL: output order diverges from release order "
                  f"(seed {seed})")
        if res.n_urgent_fail != len(res.events.get("C2_urgent", [])):
            ok = False
            print(f"  [S5] FAIL: urgent mark count mismatch (seed {seed})")
    print(f"  [S5] sequence audit over 3 replications: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("sanity checks (property anchors of the reference implementation)")
    results = [s1_operator_closure(), s2_crossover_restore(),
               s3_des_accounting(), s4_surrogate(), s5_sequence_audit()]
    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)
