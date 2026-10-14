"""End-to-end demo: one small (S-class) instance through the full pipeline.

Runs in a couple of minutes on a desktop CPU and prints:
  - the current configuration X_prev and its DES evaluation
  - a short surrogate-assisted NSGA-III search
  - the DES-verified final Pareto front with stall ratios rho

Usage:  python examples/run_demo.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from seru_core import (SearchConfig, evaluate, initial_config, make_instance,
                       surrogate_assisted_nsga3)


def main() -> None:
    t0 = time.time()
    inst = make_instance(n_stations=12, size_class="S", seed=42, name="S-demo")
    labels, team_of = initial_config(inst, inst.rng)
    from seru_core import Config
    x_prev = Config.from_labels(labels, team_of)
    print(f"instance {inst.name}: N={inst.n_stations}, teams={inst.n_teams}, "
          f"serus in X_prev={x_prev.n_seru}, lambda={inst.lam_per_min:.3f}/min")

    base = evaluate(inst, x_prev, x_prev, seed=7)
    print(f"X_prev: f1={base.f1/60:.2f} h, f2={base.f2:.1f} kCNY, "
          f"f3={base.f3:.3f}, rho={base.rho:.3%}, orders={base.n_orders}")

    sc = SearchConfig(pop_size=20, max_gen=15, warmup_gen=40,  # demo: no early stop
                      check_every=5, final_reps=3, g_ctrl=20)
    res = surrogate_assisted_nsga3(inst, x_prev, sim_seed=7, alg_seed=1, sc=sc)
    print(f"search: {res.generations} generations, {res.n_des_calls} full DES "
          f"calls, wall clock {res.wall_clock_h*60:.1f} min")
    n_exec = int(res.executable.sum())
    print(f"final front (DES-verified, R=3 replications; "
          f"{n_exec}/{len(res.front)} executable at tau={sc.tau:.0%}):")
    for cfg, obj, rho, ok in zip(res.front, res.objectives, res.rho,
                                 res.executable):
        flag = "executable" if ok else "NOT executable (best-effort)"
        print(f"  K={cfg.n_seru}: f1={obj[0]/60:.2f} h, f2={obj[1]:.1f} kCNY, "
              f"f3={obj[2]:.3f}, rho={rho.mean():.3%}±{rho.std():.2%}  [{flag}]")
    print(f"total wall clock: {(time.time()-t0):.1f} s")


if __name__ == "__main__":
    main()
