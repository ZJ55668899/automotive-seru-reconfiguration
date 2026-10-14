"""Surrogate-assisted NSGA-III search with error control (Sections 4.1/4.4).

Driver of the three modules: candidate generation (operators.py), DES
evaluation (des.py) and surrogate pre-screening (surrogate.py).

Per generation: crossover + operator mutation -> surrogate prediction ->
infill selection of 10% offspring by Eq. (14) (predicted promise AND high
uncertainty) -> full DES for the infill set (common random numbers within
the generation) -> NSGA-III environmental selection.  Error trigger (MAPE
of the last 50 evaluations > eps_max) and periodic full-population DES
checks recalibrate the surrogate; termination is decided on the hypervolume
of the DES-verified archive.  The final Pareto front is re-evaluated with R
independent replications, filtered by the executability test (mean rho <=
tau), and re-screened for non-dominance before being returned.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from deap import base, creator, tools

from .constraints import Config
from .des import DESResult, evaluate
from .operators import crossover, random_operator, warm_start
from .surrogate import RidgeSurrogate, features
from .instances import Instance

# DEAP fitness/individual types for the NSGA-III environmental selection
if "SeruFitnessMin" not in creator.__dict__:
    creator.create("SeruFitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
if "SeruIndividual" not in creator.__dict__:
    creator.create("SeruIndividual", list,
                   fitness=creator.__dict__["SeruFitnessMin"])


@dataclass
class SearchConfig:
    pop_size: int = 100
    infill_share: float = 0.10        # 10% of offspring receive full DES
    eps_max: float = 0.08             # error trigger threshold (MAPE)
    g_ctrl: int = 20                  # periodic full-population DES check
    warmup_gen: int = 40              # termination check starts afterwards
    hv_patience: int = 20             # HV improvement window (generations)
    hv_tol: float = 0.01              # 1% relative HV improvement threshold
    check_every: int = 5              # HV check cadence
    max_gen: int = 500
    max_hours: float = 8.0
    final_reps: int = 20              # R replications for the final front
    ref_p: int = 4                    # Das-Dennis partition (15 directions)
    tau: float = 0.05                 # executability threshold on mean rho


@dataclass
class SearchResult:
    front: list[Config]
    objectives: np.ndarray           # (n_front, 3) DES-verified objectives
    rho: np.ndarray                  # (n_front, R) stall ratios per replication
    executable: np.ndarray           # (n_front,) bool: mean rho <= tau
    n_des_calls: int                 # full DES calls spent during the search
    generations: int
    wall_clock_h: float
    hv_trace: list = field(default_factory=list)


def _non_dominated(objs: np.ndarray) -> np.ndarray:
    """Indices of non-dominated points (minimisation)."""
    n = len(objs)
    keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                keep[i] = False
                break
    return np.where(keep)[0]


def _norm_ranks(objs: np.ndarray) -> np.ndarray:
    """Normalised non-domination ranks in [0, 1] (0 = current front)."""
    n = len(objs)
    ranks = np.zeros(n)
    remaining = np.arange(n)
    r = 0
    while len(remaining):
        idx = _non_dominated(objs[remaining])
        ranks[remaining[idx]] = r
        remaining = np.delete(remaining, idx)
        r += 1
    return ranks / max(1, r - 1) if r > 1 else ranks


def _hypervolume_2d_proxy(front_objs: np.ndarray) -> float:
    """Normalised hypervolume proxy used only for the termination check
    (objectives normalised by the archive's ideal/nadir points)."""
    if len(front_objs) == 0:
        return 0.0
    o = front_objs
    lo, hi = o.min(0), o.max(0)
    span = np.maximum(hi - lo, 1e-12)
    on = (o - lo) / span
    ref = np.ones(3) * 1.1
    # simple Monte-Carlo-free 3D hypervolume by slicing the third objective
    order = np.argsort(on[:, 2])
    hv = 0.0
    prev_z = 0.0
    for idx in order[::-1]:
        z = on[idx, 2]
        band = ref[2] - z
        if band < prev_z:
            break
        sub = on[on[:, 2] <= z]
        # 2D HV of the band slice in (f1, f2)
        s = sub[np.argsort(sub[:, 0])]
        area, best_y = 0.0, ref[1]
        for pt in s:
            if pt[1] < best_y:
                area += (ref[0] - pt[0]) * (best_y - pt[1])
                best_y = pt[1]
        hv += area * (band - prev_z)
        prev_z = band
    return float(hv)


def surrogate_assisted_nsga3(inst: Instance, x_prev: Config,
                             sim_seed: int, alg_seed: int,
                             sc: SearchConfig | None = None) -> SearchResult:
    """The full search loop of Algorithm 2."""
    sc = sc or SearchConfig()
    rng = np.random.default_rng(alg_seed)
    # DEAP's selNSGA3 breaks niche ties through numpy's global RNG; seed it so
    # that runs are reproducible across processes.
    np.random.seed((alg_seed * 2654435761) % (2 ** 32))
    t_start = time.time()

    archive_x: list[np.ndarray] = []      # features of DES-evaluated configs
    archive_y: list[np.ndarray] = []      # their true objectives
    surrogate = RidgeSurrogate(3)
    err_window: list[float] = []          # |pred - true| / true per evaluation
    n_des = 0

    def full_des(cfg: Config) -> DESResult:
        nonlocal n_des
        n_des += 1
        return evaluate(inst, cfg, x_prev, seed=sim_seed)

    # ---- 1. warm start + initial full evaluation ------------------------------
    pop = warm_start(inst, x_prev, rng, sc.pop_size)
    pop_fit: list[np.ndarray] = []
    for cfg in pop:
        res = full_des(cfg)
        pop_fit.append(np.array(res.as_objectives()))
        archive_x.append(features(inst, cfg, x_prev))
        archive_y.append(pop_fit[-1])
    pop_true = [True] * len(pop)      # initial population is fully DES-verified
    surrogate.fit_with_gram(np.array(archive_x), np.array(archive_y))

    ref_points = tools.uniform_reference_points(3, p=sc.ref_p)
    hv_trace: list[float] = []
    best_hv = -np.inf
    gen = 0

    # ---- 2. main loop ------------------------------------------------------------
    while gen < sc.max_gen and (time.time() - t_start) < sc.max_hours * 3600:
        gen += 1
        # crossover + operator mutation (feasibility preserved by Prop. 2,
        # fixed-tooling stations anchored to their incumbent seru)
        offspring: list[Config] = []
        parents = list(range(len(pop)))
        rng.shuffle(parents)
        for a, b in zip(parents[0::2], parents[1::2]):
            offspring.append(crossover(inst, pop[a], pop[b], rng, x_prev))
            offspring.append(crossover(inst, pop[b], pop[a], rng, x_prev))
        offspring = [random_operator(inst, c, rng, x_prev) for c in offspring]
        offspring = offspring[: sc.pop_size]

        # surrogate prediction + infill selection by Eq. (14): high psi =
        # promising predicted objectives AND high uncertainty (exploration)
        x_off = np.array([features(inst, c, x_prev) for c in offspring])
        y_hat, u_hat = surrogate.predict(x_off)
        ranks = _norm_ranks(y_hat)
        u_norm = u_hat.mean(1)
        u_norm = u_norm / (u_norm.max() + 1e-12)
        psi = 0.5 * (1 - ranks) + 0.5 * u_norm
        n_infill = max(1, int(round(sc.infill_share * len(offspring))))
        infill_idx = np.argsort(-psi)[:n_infill]

        periodic = (gen % sc.g_ctrl == 0)
        off_fit: list[np.ndarray] = list(np.array(y_hat))
        off_true = [False] * len(offspring)

        def verify(i: int):
            res = full_des(offspring[i])
            true = np.array(res.as_objectives())
            err = np.abs(true - y_hat[i]) / np.maximum(np.abs(true), 1e-12)
            err_window.append(float(err.mean()))
            archive_x.append(x_off[i])
            archive_y.append(true)
            off_fit[i] = true
            off_true[i] = True

        for i in range(len(offspring)):
            if i in infill_idx or periodic:
                verify(i)
        # periodic full-population check: parents that still carry predicted
        # fitness are re-evaluated as well, so every population member holds
        # DES-verified objectives after a control generation
        if periodic:
            for i, cfg in enumerate(pop):
                if not pop_true[i]:
                    res = full_des(cfg)
                    pop_fit[i] = np.array(res.as_objectives())
                    pop_true[i] = True
                    archive_x.append(features(inst, cfg, x_prev))
                    archive_y.append(pop_fit[i])
        surrogate.fit_with_gram(np.array(archive_x), np.array(archive_y))

        # error trigger: full-population DES for this generation if MAPE too high
        if len(err_window) >= 50 and np.mean(err_window[-50:]) > sc.eps_max:
            for i in range(len(offspring)):
                if not off_true[i]:
                    res = full_des(offspring[i])
                    off_fit[i] = np.array(res.as_objectives())
                    off_true[i] = True
                    archive_x.append(x_off[i])
                    archive_y.append(off_fit[i])
            surrogate.fit_with_gram(np.array(archive_x), np.array(archive_y))
            err_window.clear()

        # NSGA-III environmental selection over parent union offspring
        union_fit = np.array(pop_fit + off_fit)
        union_pop = pop + offspring
        union_true = pop_true + off_true
        inds = []
        for idx, fit in enumerate(union_fit):
            ind = creator.SeruIndividual([idx])
            ind.fitness.values = tuple(float(v) for v in fit)
            inds.append(ind)
        # ideal/nadir from the archive, per Section 4.4
        arch = np.array(archive_y)
        best_pt = arch.min(0).reshape(1, -1)
        worst_pt = arch.max(0).reshape(1, -1)
        chosen = tools.selNSGA3(inds, sc.pop_size, ref_points,
                                best_point=best_pt, worst_point=worst_pt)
        sel_idx = [int(ind[0]) for ind in chosen]
        pop = [union_pop[i] for i in sel_idx]
        pop_fit = [union_fit[i] for i in sel_idx]
        pop_true = [union_true[i] for i in sel_idx]

        # ---- 3. termination check ------------------------------------------------
        # HV is computed on the DES-verified archive only (Section 4.4);
        # hv_trace holds one entry per generation, so the 20-generation window
        # is the entry 20 positions back.
        arch = np.array(archive_y)
        hv = _hypervolume_2d_proxy(arch[_non_dominated(arch)])
        hv_trace.append(hv)
        if gen > sc.warmup_gen and gen % sc.check_every == 0:
            if len(hv_trace) > sc.hv_patience:
                ref_hv = hv_trace[-1 - sc.hv_patience]
                if ref_hv > 0 and (hv - ref_hv) / ref_hv < sc.hv_tol:
                    break
            best_hv = max(best_hv, hv)

    # ---- 4. final front: full DES re-evaluation with R replications --------------
    # R independent random streams per front solution (Section 3.4, Eq. 9);
    # these replications belong to output verification and are not counted in
    # the search budget n_des_calls.
    pop_fit_arr = np.array(pop_fit)
    front_idx = _non_dominated(pop_fit_arr)
    front = [pop[i] for i in front_idx]
    # drop duplicate configurations so each returned front member is distinct
    seen_cfg: set[tuple] = set()
    front = [c for c in front
             if (key := (c.labels.tobytes(), c.team_of.tobytes())) not in seen_cfg
             and not seen_cfg.add(key)]
    final_obj, rho_reps = [], []
    for cfg in front:
        reps = [evaluate(inst, cfg, x_prev, seed=sim_seed + 7919 * (r + 1))
                for r in range(sc.final_reps)]
        final_obj.append(np.mean([r.as_objectives() for r in reps], axis=0))
        rho_reps.append([r.rho for r in reps])
    final_obj = np.array(final_obj)
    rho_reps = np.array(rho_reps)

    # Algorithm 2, final step: keep configurations that pass the executability
    # test (C1/C4 hold by construction via Prop. 2; mean rho over the R
    # replications must not exceed tau), then re-screen for non-dominance on
    # the DES-verified objectives.
    executable = rho_reps.mean(1) <= sc.tau
    if executable.any():
        idx_exec = np.where(executable)[0]
        idx_nd = _non_dominated(final_obj[executable])
        keep = idx_exec[idx_nd]
    else:
        # no configuration meets tau under the recheck: return the rechecked
        # non-dominated set with executable=False throughout, so callers can
        # see the best-effort front and its failed verification
        keep = _non_dominated(final_obj)

    return SearchResult(front=[front[i] for i in keep],
                        objectives=final_obj[keep], rho=rho_reps[keep],
                        executable=executable[keep], n_des_calls=n_des,
                        generations=gen, wall_clock_h=(time.time() - t_start) / 3600,
                        hv_trace=hv_trace)
