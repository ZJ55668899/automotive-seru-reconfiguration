"""Constraint handling for the seru reconfiguration model (Section 3.3).

C1  workstation layout     - checked at candidate generation (here)
C2  vehicle sequencing     - checked in the DES (see des.py)
C3  AGV path capacity      - checked in the DES (see des.py)
C4  team capability match  - checked at candidate generation (here)

Omega_feas^st = {X : C1(X) and C4(X)} is the set of configurations satisfying
the a-priori structural constraints.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instances import D_MAX_M, Instance, team_compatible


@dataclass
class Config:
    """A configuration X = (seru labels per station, team per seru)."""
    labels: np.ndarray      # (N,) int seru label per station
    team_of: np.ndarray     # (K,) team index per seru label
    n_seru: int

    @classmethod
    def from_labels(cls, labels: np.ndarray, team_of: np.ndarray) -> "Config":
        labels = np.asarray(labels, int).copy()
        team_of = np.asarray(team_of, int).copy()
        return cls(labels, team_of, int(labels.max()) + 1)

    def copy(self) -> "Config":
        return Config(self.labels.copy(), self.team_of.copy(), self.n_seru)


# --------------------------------------------------------------------- C1

def seru_stations(cfg: Config, k: int) -> np.ndarray:
    return np.where(cfg.labels == k)[0]


def is_connected(inst: Instance, stations: np.ndarray) -> bool:
    """Connectivity of the induced subgraph H[V_k] (Eq. 3)."""
    s = set(int(i) for i in stations)
    if len(s) <= 1:
        return True
    seen = {next(iter(s))}
    stack = list(seen)
    while stack:
        u = stack.pop()
        for v in inst.adjacency[u]:
            if v in s and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen == s


def diameter_ok(inst: Instance, stations: np.ndarray, d_max: float = D_MAX_M) -> bool:
    """Seru spatial diameter limit (Eq. 2)."""
    if len(stations) <= 1:
        return True
    pts = inst.coords[list(stations)]
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).max()
    return bool(d <= d_max + 1e-9)


def check_c1(inst: Instance, cfg: Config) -> bool:
    return all(
        is_connected(inst, seru_stations(cfg, k)) and diameter_ok(inst, seru_stations(cfg, k))
        for k in range(cfg.n_seru)
    )


# --------------------------------------------------------------------- C4

def check_c4(inst: Instance, cfg: Config) -> bool:
    """One team per seru, injective assignment, skill and crew coverage."""
    teams = [int(t) for t in cfg.team_of[: cfg.n_seru]]
    if len(set(teams)) != len(teams):
        return False
    for k in range(cfg.n_seru):
        if not team_compatible(inst, teams[k], seru_stations(cfg, k)):
            return False
    return True


def in_omega_st(inst: Instance, cfg: Config) -> bool:
    """Membership in Omega_feas^st (both a-priori structural constraints)."""
    return check_c1(inst, cfg) and check_c4(inst, cfg)


def compatibility_matrix(inst: Instance) -> np.ndarray:
    """Xi in {0,1}^{N x T}: 1 iff team t can serve station i (computed once
    per instance, Section 4.2)."""
    xi = np.zeros((inst.n_stations, inst.n_teams), bool)
    for i in range(inst.n_stations):
        for t in range(inst.n_teams):
            xi[i, t] = team_compatible(inst, t, np.array([i]))
    return xi


# ------------------------------------------------------- C2 rule checkers
# The DES evaluates these on the evolving output sequence; the pure rule
# checks live here so that the dispatcher and the auditor share one source.

def spacing_ok(seq_tail: list[int], candidate: int, p: int = 2, q: int = 3) -> bool:
    """Sliding-window spacing rule (Eq. 4): at most p vehicles of the same
    model in any q consecutive outputs."""
    window = (seq_tail + [candidate])[-q:]
    return window.count(candidate) <= p


def setup_within_budget(cum_setup: float, delta: float, budget: float) -> bool:
    """Per-shift changeover-time budget (Eq. 5)."""
    return cum_setup + delta <= budget + 1e-12
