"""Candidate-generation operators on the attribute graph (Section 4.2).

The configuration is stored as node attributes (seru label, team label) on a
fixed directed graph, so reconfiguration is pure relabelling.  Each operator
checks C1 and C4 *before* acceptance (execution conditions); Proposition 2
states that starting from Omega_feas^st the operators stay in Omega_feas^st.

merge(k' -> k):     merge seru k' into k; the merged seru keeps one of the two
                    incumbent teams if compatible with all stations, else the
                    operator is rejected (the other team is released).
split(k, S):        split S out of seru k; S receives a compatible unassigned
                    team, the remainder keeps its team.
reassign(k, t):     replace seru k's team by a compatible team t that is
                    unassigned or freed by a paired move.
restore(X):         rebuild feasibility after crossover (see restore()).
"""
from __future__ import annotations

import numpy as np

from .constraints import (Config, check_c1, diameter_ok, in_omega_st,
                          is_connected, seru_stations)
from .instances import Instance, team_compatible


def _team_map(cfg: Config) -> dict[int, int]:
    return {k: int(cfg.team_of[k]) for k in range(cfg.n_seru)}


def merge(inst: Instance, cfg: Config, k_from: int, k_to: int) -> Config | None:
    """Merge seru k_from into k_to (execution conditions of Section 4.2)."""
    if k_from == k_to or k_from >= cfg.n_seru or k_to >= cfg.n_seru:
        return None
    union = np.where(np.isin(cfg.labels, [k_from, k_to]))[0]
    if not (is_connected(inst, union) and diameter_ok(inst, union)):
        return None
    teams = [int(cfg.team_of[k_to]), int(cfg.team_of[k_from])]
    keep = None
    for t in teams:  # one of the two incumbent teams must cover the union
        if team_compatible(inst, t, union):
            keep = t
            break
    if keep is None:
        return None
    labels = cfg.labels.copy()
    labels[labels == k_from] = k_to
    return _relabel(labels, {k_to: keep})


def split(inst: Instance, cfg: Config, k: int, rng: np.random.Generator) -> Config | None:
    """Split seru k into two connected parts; the new seru needs a compatible
    unassigned team."""
    members = seru_stations(cfg, k)
    if len(members) < 2:
        return None
    # split along a random connected boundary: grow part S from a random seed
    seed = int(rng.choice(members))
    s_set = {seed}
    while len(s_set) < len(members) // 2 or rng.random() < 0.5:
        frontier = {v for u in s_set for v in inst.adjacency[u]
                    if v in set(members) and v not in s_set}
        if not frontier or len(s_set) >= len(members) - 1:
            break
        s_set.add(int(rng.choice(sorted(frontier))))
        if len(s_set) >= len(members) - 1:
            break
    rest = sorted(set(int(m) for m in members) - s_set)
    if len(rest) == 0 or len(s_set) == 0:
        return None
    s_arr = np.array(sorted(s_set))
    rest_arr = np.array(rest)
    if not (is_connected(inst, s_arr) and diameter_ok(inst, s_arr)
            and is_connected(inst, rest_arr) and diameter_ok(inst, rest_arr)):
        return None
    used = set(_team_map(cfg).values())
    keep_team = int(cfg.team_of[k])
    if not team_compatible(inst, keep_team, rest_arr):
        return None
    free = [t for t in range(inst.n_teams)
            if t not in used and team_compatible(inst, t, s_arr)]
    if not free:
        return None
    new_label = cfg.n_seru
    labels = cfg.labels.copy()
    labels[s_arr] = new_label
    assign = _team_map(cfg)
    assign[k] = keep_team
    assign[new_label] = int(rng.choice(free))
    return _relabel(labels, assign)


def reassign(inst: Instance, cfg: Config, k: int, rng: np.random.Generator) -> Config | None:
    """Replace seru k's team by another compatible team (unassigned, or freed
    by a paired swap)."""
    members = seru_stations(cfg, k)
    current = int(cfg.team_of[k])
    map_k = _team_map(cfg)
    used = set(map_k.values())
    cands = [t for t in range(inst.n_teams)
             if t != current and team_compatible(inst, t, members)]
    rng.shuffle(cands)
    for t in cands:
        if t not in used:  # unassigned team: direct replacement
            map_k[k] = t
            return _relabel(cfg.labels, map_k)
        # paired move: t currently serves k2; the swap needs current to cover k2
        k2 = next(kk for kk, tt in map_k.items() if tt == t)
        if team_compatible(inst, current, seru_stations(cfg, k2)):
            map_k[k], map_k[k2] = t, current
            return _relabel(cfg.labels, map_k)
    return None


def random_operator(inst: Instance, cfg: Config, rng: np.random.Generator) -> Config:
    """One mutation step: uniformly choose merge / split / reassign; apply only
    when its execution conditions hold, otherwise keep the input (Alg. 1)."""
    ops = ["merge", "split", "reassign"]
    rng.shuffle(ops)
    for op in ops:
        if op == "merge" and cfg.n_seru > 1:
            a, b = rng.choice(cfg.n_seru, 2, replace=False)
            out = merge(inst, cfg, int(a), int(b))
        elif op == "split":
            out = split(inst, cfg, int(rng.integers(cfg.n_seru)), rng)
        elif op == "reassign":
            out = reassign(inst, cfg, int(rng.integers(cfg.n_seru)), rng)
        else:
            out = None
        if out is not None and in_omega_st(inst, out):
            return out
    return cfg


def crossover(inst: Instance, a: Config, b: Config, rng: np.random.Generator) -> Config:
    """Single-point crossover on seru labels followed by restore(); the child
    inherits each seru's team from the parent contributing the larger part."""
    n = inst.n_stations
    cut = int(rng.integers(1, n))
    labels = np.concatenate([a.labels[:cut], b.labels[cut:] + a.n_seru])
    team_map: dict[int, int] = {}
    amap, bmap = _team_map(a), _team_map(b)
    for k in np.unique(labels):
        if k < a.n_seru:
            team_map[int(k)] = amap.get(int(k), 0)
        else:
            team_map[int(k)] = bmap.get(int(k) - a.n_seru, 0)
    child = _relabel(labels, team_map, check=False)
    fixed = restore(inst, child, rng)
    if fixed is None or not in_omega_st(inst, fixed):
        return a.copy()  # fall back to the parent clone
    return fixed


def restore(inst: Instance, cfg: Config, rng: np.random.Generator,
            max_rounds: int = 20) -> Config | None:
    """Restore C1/C4 after crossover (Section 4.2): for each fragmented seru
    keep the largest connected component, merge orphaned components into an
    adjacent seru when the execution conditions hold, otherwise split them off
    as new serus with compatible unassigned teams; then repair C4 by feasible
    reassignments.  Returns None when restoration fails within max_rounds."""
    cfg = cfg.copy()
    for _ in range(max_rounds):
        if check_c1(inst, cfg) and check_c4_flag(inst, cfg):
            return cfg
        changed = False
        for k in list(range(cfg.n_seru)):
            members = seru_stations(cfg, k)
            if len(members) == 0:
                continue
            comps = _components(inst, members)
            if len(comps) <= 1:
                continue
            comps.sort(key=len, reverse=True)
            labels = cfg.labels.copy()
            orphans = [c for c in comps[1:]]
            for comp in orphans:
                merged = False
                # try merging the orphan into an adjacent seru
                neigh_labels = sorted({int(labels[v]) for u in comp for v in inst.adjacency[u]
                                       if labels[v] != k})
                for nb in neigh_labels:
                    trial_labels = labels.copy()
                    trial_labels[list(comp)] = nb
                    union = np.where(trial_labels == nb)[0]
                    if is_connected(inst, union) and diameter_ok(inst, union):
                        tmap = _team_map(cfg)
                        tmap[nb] = int(cfg.team_of[nb])
                        if team_compatible(inst, tmap[nb], union):
                            labels = trial_labels
                            merged = True
                            changed = True
                            break
                if not merged:
                    # split the orphan off as a new seru with a free team
                    new_label = int(labels.max()) + 1
                    labels[list(comp)] = new_label
                    changed = True
            cfg = _relabel(labels, _team_map(cfg), check=False)
            break
        # C4 repair: every seru needs its own compatible team; resolve
        # duplicate teams and uncovered serus by feasible reassignments
        seen_teams: set[int] = set()
        for k in range(cfg.n_seru):
            members = seru_stations(cfg, k)
            t = int(cfg.team_of[k])
            if t in seen_teams or not team_compatible(inst, t, members):
                fixed = reassign(inst, cfg, k, rng)
                if fixed is not None and int(fixed.team_of[k]) != t:
                    cfg = fixed
                    t = int(cfg.team_of[k])
                    changed = True
            seen_teams.add(t)
        if not changed:
            break
    return cfg if (check_c1(inst, cfg) and check_c4_flag(inst, cfg)) else None


# ------------------------------------------------------------------ helpers

def _components(inst: Instance, members: np.ndarray) -> list[list[int]]:
    s = set(int(m) for m in members)
    comps = []
    while s:
        start = s.pop()
        comp = [start]
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in inst.adjacency[u]:
                if v in s and v not in seen:
                    seen.add(v)
                    stack.append(v)
                    comp.append(v)
        for v in comp:
            s.discard(v)
        comps.append(comp)
    return comps


def check_c4_flag(inst: Instance, cfg: Config) -> bool:
    from .constraints import check_c4 as _c4
    return _c4(inst, cfg)


def _relabel(labels: np.ndarray, team_map: dict[int, int], check: bool = True) -> Config:
    """Renumber seru labels to 0..K-1 and build the team vector."""
    labels = np.asarray(labels, int).copy()
    uniq = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
    remapped = np.array([uniq[x] for x in labels], int)
    k_max = len(uniq)
    team_of = np.zeros(k_max, int)
    for old, new in uniq.items():
        team_of[new] = team_map.get(old, team_map.get(int(old), 0))
    return Config(remapped, team_of, k_max)


def warm_start(inst: Instance, base: Config, rng: np.random.Generator,
               pop_size: int) -> list[Config]:
    """Initial population: 1-3 feasible adjustment operations from X_prev
    (Section 4.2, warm start around the current configuration)."""
    pop = [base.copy()]
    while len(pop) < pop_size:
        cfg = base.copy()
        for _ in range(int(rng.integers(1, 4))):
            cfg = random_operator(inst, cfg, rng)
        pop.append(cfg)
    return pop
