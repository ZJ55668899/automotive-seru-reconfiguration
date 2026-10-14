"""Synthetic instance generation for the automotive final-assembly seru
reconfiguration problem (Section 5.1 and Table 4 of the manuscript).

The full benchmark in the paper augments precedence graphs from the Scholl
assembly-line balancing library.  To keep this reference implementation
self-contained, the generator below produces synthetic U-shaped-line instances
with the same parameterisation (layout spacing, product mix, skill model,
AGV fleet, sequencing rules, cost coefficients).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---- Table 4 (benchmark automotive attributes) -----------------------------
STATION_SPACING_M = 6.0        # main-line station spacing
RETURN_AISLE_M = 20.0          # distance between the two legs of the U
D_MAX_M = 45.0                 # maximal seru spatial diameter
FIXTURE_SHARE = 0.30           # share of stations with fixed tooling
MIX = (0.40, 0.30, 0.20, 0.10)         # product-family mix p_mix
MODEL_COEF = (1.00, 1.18, 1.12, 1.25)  # processing-time coefficient per family
N_MODELS = 4
SPACING_RULE = (2, 3)          # at most 2 of the same model in any 3 outputs
SETUP_RANGE_MIN = (0.4, 1.5)   # model changeover time range (min), 0 if same
SETUP_BUDGET_MIN = 45.0        # per-shift changeover budget B_setup
LOOKAHEAD_W = 5                # dispatch look-ahead window
SEGMENT_CAP_RANGE = (1, 3)     # AGV path-segment capacity
AGV_FLEET = {"S": 12, "M": 24, "L": 40}
AGV_SPEED_MPS = 1.2            # m/s
AGV_LOAD_MIN = 0.5             # load/unload time (min)
SKILL_DIM = 6
TEAM_SKILL_BETA = (5, 2)       # team skill ~ Beta(5,2)
STATION_REQ_BETA = (2, 5)      # station requirement ~ Beta(2,5)
LEARNING_ETA0 = 0.75           # initial efficiency of a relocated team
LEARNING_THETA_MIN = 25.0      # learning time constant (min)
T_SHIFT_MIN = 480.0            # nominal shift length
NOMINAL_UTIL = 0.85            # nominal line utilisation behind lambda
VOLUME_JITTER = 0.30           # shift-to-shift volume fluctuation
COST_ALPHA = 1.2               # kCNY per station with changed seru membership
COST_BETA = 5.0                # kCNY per relocated team
COST_GAMMA = 9.0               # kCNY per hour of transition downtime
T_FLUSH_MIN = 1.5              # WIP transfer time per unit to central buffer
THETA_WAIT_MIN = 2.0           # AGV waiting threshold before stall recording
DEADLOCK_RECOVERY_MEAN_MIN = 3.0
URGENT_SHARE = 0.05            # share of urgent orders
URGENT_LEAD = 8                # urgent order position allowance


@dataclass
class Instance:
    """One benchmark instance (all data needed to encode and evaluate)."""
    name: str
    n_stations: int
    coords: np.ndarray            # (N, 2) metres
    adjacency: list[set[int]]     # layout adjacency graph H
    base_time: np.ndarray         # (N,) seconds, per-station nominal work
    fixtures: np.ndarray          # (N,) bool, fixed-tooling stations
    skill_req: np.ndarray         # (N, d) in [0,1]
    crew_req: np.ndarray          # (N,) int
    team_skill: np.ndarray        # (T, d) in [0,1]
    team_size: np.ndarray         # (T,) int
    k_max: int
    size_class: str
    agv_segments: dict            # (u,v) node pair -> capacity c(l)
    agv_routes: dict              # (seru_i, seru_j) station-pair route -> [segments]
    segment_length: dict          # segment -> metres
    n_agv: int
    lam_per_min: float            # Poisson arrival rate
    rng: np.random.Generator = field(repr=False)

    # -- derived views ------------------------------------------------------
    @property
    def proc_time(self) -> np.ndarray:
        """t_{i,m} in seconds: station base time x model coefficient."""
        return self.base_time[:, None] * np.asarray(MODEL_COEF)[None, :]

    @property
    def n_teams(self) -> int:
        return len(self.team_size)


def _u_line_coords(n: int) -> np.ndarray:
    """U-shaped main line: first half on leg y=0, remainder on leg y=20."""
    h = (n + 1) // 2
    pts = []
    for i in range(n):
        if i < h:
            pts.append((STATION_SPACING_M * i, 0.0))
        else:
            j = i - h
            pts.append((STATION_SPACING_M * (h - 1 - j), RETURN_AISLE_M))
    return np.asarray(pts, float)


def _adjacency(coords: np.ndarray) -> list[set[int]]:
    """H: main-line neighbours plus cross-aisle neighbours (same x column)."""
    n = len(coords)
    adj = [set() for _ in range(n)]
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    xs = {}
    for i, (x, y) in enumerate(coords):
        xs.setdefault(round(x, 6), []).append(i)
    for col in xs.values():
        if len(col) == 2:  # opposite legs in the same column
            a, b = col
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _agv_network(coords: np.ndarray, rng: np.random.Generator):
    """Grid AGV network induced by the layout: segments along both legs and
    one cross-aisle link per column; routes between stations are shortest
    paths (fewest segments) over this grid."""
    n = len(coords)
    segments: dict[tuple[int, int], int] = {}
    length: dict[tuple[int, int], float] = {}

    def add_seg(a: int, b: int):
        cap = int(rng.integers(SEGMENT_CAP_RANGE[0], SEGMENT_CAP_RANGE[1] + 1))
        segments[(a, b)] = cap
        segments[(b, a)] = cap
        d = float(np.hypot(*(coords[a] - coords[b])))
        length[(a, b)] = d
        length[(b, a)] = d

    for i in range(n - 1):
        add_seg(i, i + 1)
    xs: dict[float, list[int]] = {}
    for i, (x, _) in enumerate(coords):
        xs.setdefault(round(x, 6), []).append(i)
    for col in xs.values():
        if len(col) == 2:
            add_seg(*col)

    # all-pairs shortest paths on the segment graph (fewest hops)
    adj = [set() for _ in range(n)]
    for a, b in segments:
        adj[a].add(b)
    routes: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def bfs(s: int, t: int):
        prev = {s: None}
        q = [s]
        for u in q:
            if u == t:
                break
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if t not in prev:
            return []
        path, u = [], t
        while prev[u] is not None:
            path.append((prev[u], u))
            u = prev[u]
        return path[::-1]

    for s in range(n):
        for t in range(n):
            if s != t:
                routes[(s, t)] = bfs(s, t)
    return segments, routes, length


def make_instance(n_stations: int, size_class: str, seed: int,
                  name: str | None = None,
                  fixture_share: float | None = None) -> Instance:
    """Generate one synthetic instance following the Table 4 parameterisation.

    fixture_share overrides the Table 4 default (0.30); the fixture flags pin
    the corresponding stations to their incumbent seru (see
    constraints.fixtures_anchored)."""
    rng = np.random.default_rng(seed)
    coords = _u_line_coords(n_stations)
    adjacency = _adjacency(coords)
    base_time = rng.uniform(30.0, 90.0, n_stations)          # seconds
    share = FIXTURE_SHARE if fixture_share is None else fixture_share
    fixtures = rng.random(n_stations) < share
    skill_req = rng.beta(*STATION_REQ_BETA, (n_stations, SKILL_DIM))
    crew_req = rng.integers(1, 5, n_stations)
    k_max = math.ceil(n_stations / 3) + 2
    n_teams = k_max + 2
    team_skill = rng.beta(*TEAM_SKILL_BETA, (n_teams, SKILL_DIM))
    team_size = rng.integers(2, 6, n_teams)
    segments, routes, seg_len = _agv_network(coords, rng)
    # Arrival rate is calibrated in initial_config() against the bottleneck
    # seru of the current configuration (0.85 nominal utilisation).
    lam_per_min = 0.0
    return Instance(
        name=name or f"{size_class}{n_stations}",
        n_stations=n_stations,
        coords=coords,
        adjacency=adjacency,
        base_time=base_time,
        fixtures=fixtures,
        skill_req=skill_req,
        crew_req=crew_req,
        team_skill=team_skill,
        team_size=team_size,
        k_max=k_max,
        size_class=size_class,
        agv_segments=segments,
        agv_routes=routes,
        segment_length=seg_len,
        n_agv=AGV_FLEET.get(size_class, 24),
        lam_per_min=float(lam_per_min),
        rng=rng,
    )


def initial_config(inst: Instance, rng: np.random.Generator):
    """Current configuration X_prev: contiguous station blocks along the
    process order (5-7 stations each), each with a feasible team."""
    n = inst.n_stations
    labels = np.zeros(n, int)
    k = 0
    i = 0
    while i < n:
        size = int(rng.integers(5, 8))
        labels[i:i + size] = k
        i += size
        k += 1
    n_seru = k
    team_of = np.zeros(n_seru, int)
    used: set[int] = set()
    for s in range(n_seru):
        members = np.where(labels == s)[0]
        feas = [t for t in range(inst.n_teams)
                if t not in used and team_compatible(inst, t, members)]
        if feas:
            team_of[s] = feas[0]
            used.add(feas[0])
        else:
            # The current line configuration is feasible by definition; if the
            # random pool has no compatible team left, synthesise one whose
            # skills cover this seru's requirements with a small margin.
            t = _synthesize_team(inst, members, used)
            team_of[s] = t
            used.add(t)
    # calibrate the arrival rate: 0.85 utilisation of the bottleneck seru
    mix_coef = float(np.dot(MIX, MODEL_COEF))
    seru_times = [float(inst.base_time[labels == s].sum()) * mix_coef
                  for s in range(n_seru)]
    inst.lam_per_min = NOMINAL_UTIL / (max(seru_times) / 60.0)
    return labels, team_of


def _synthesize_team(inst: Instance, members: np.ndarray, used: set[int]) -> int:
    """Append a team to the pool that is compatible with the given stations."""
    t = inst.n_teams
    cover = inst.skill_req[list(members)].max(axis=0)
    inst.team_skill = np.vstack([inst.team_skill,
                                 np.clip(cover + 0.05, 0.0, 1.0)])
    inst.team_size = np.append(inst.team_size,
                               int(inst.crew_req[list(members)].max()))
    return t


def team_compatible(inst: Instance, t: int, stations: np.ndarray) -> bool:
    """C4 matching for team t on the given station set: crew size covers the
    maximal crew requirement and every skill dimension is covered."""
    if len(stations) == 0:
        return True
    if inst.team_size[t] < int(inst.crew_req[list(stations)].max()):
        return False
    return bool(np.all(inst.team_skill[t][None, :] >= inst.skill_req[list(stations)]))
