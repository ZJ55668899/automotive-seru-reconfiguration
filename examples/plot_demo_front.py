"""Demo with figure output: run the small S-class search and plot the final
Pareto front (makespan vs transition cost, colour = stall ratio rho).

Produces examples/demo_front.png.

Usage:  python examples/plot_demo_front.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from seru_core import (Config, SearchConfig, initial_config, make_instance,
                       surrogate_assisted_nsga3)

OUT = pathlib.Path(__file__).resolve().parent / "demo_front.png"


def main() -> None:
    inst = make_instance(n_stations=12, size_class="S", seed=42, name="S-demo")
    labels, team_of = initial_config(inst, inst.rng)
    x_prev = Config.from_labels(labels, team_of)

    sc = SearchConfig(pop_size=20, max_gen=15, warmup_gen=40,
                      check_every=5, final_reps=3, g_ctrl=20)
    res = surrogate_assisted_nsga3(inst, x_prev, sim_seed=7, alg_seed=1, sc=sc)

    f1 = res.objectives[:, 0] / 60.0          # h
    f2 = res.objectives[:, 1]                 # kCNY
    rho = res.rho.mean(1) * 100.0             # %
    order = np.argsort(f1)
    f1, f2, rho = f1[order], f2[order], rho[order]

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    sc_pts = ax.scatter(f1, f2, c=rho, cmap="RdYlGn_r", s=90,
                        edgecolor="k", linewidth=0.6, zorder=3)
    ax.plot(f1, f2, color="grey", lw=1.0, zorder=2, alpha=0.6)
    i0 = int(np.argmin(f2))
    ax.annotate("current config $X_{prev}$", (f1[i0], f2[i0]),
                textcoords="offset points", xytext=(-10, 14),
                ha="right", fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    cb = fig.colorbar(sc_pts, ax=ax)
    cb.set_label(r"stall ratio $\rho$ (%)")
    ax.set_xlabel("makespan $f_1$ (h)")
    ax.set_ylabel("transition cost $f_2$ (kCNY)")
    ax.set_title("Demo S-instance: final Pareto front (DES-verified)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"saved {OUT} ({len(f1)} front solutions)")


if __name__ == "__main__":
    main()
