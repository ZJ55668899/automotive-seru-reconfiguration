"""Discrete-event simulation (DES) evaluator (Section 4.3).

Reproduces a candidate configuration in operation: order arrivals, vehicle
dispatch under the sequencing rules (C2), AGV transport under path-segment
capacity (C3), changeovers, and the transition from the current configuration.

The DES records three system states (normal / constraint stall / recovery).
Stall and recovery intervals are merged on the *system* timeline before
integration, so overlapping events on different resources are never double
counted and rho = T_CID / f_1 stays in [0, 1] (Eq. 13).  Per-constraint event
counts and durations are recorded separately for diagnosis (Figs. 5-6).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import simpy

from .constraints import (Config, check_c1, check_c4, diameter_ok,
                          is_connected, seru_stations, spacing_ok,
                          station_change_mask, unit_match_to_prev)
from .instances import (AGV_LOAD_MIN, AGV_SPEED_MPS, COST_ALPHA, COST_BETA,
                        COST_GAMMA, DEADLOCK_RECOVERY_MEAN_MIN, Instance,
                        LEARNING_ETA0, LEARNING_THETA_MIN, LOOKAHEAD_W, MIX,
                        SETUP_BUDGET_MIN, SETUP_RANGE_MIN, T_FLUSH_MIN,
                        THETA_WAIT_MIN, T_SHIFT_MIN, URGENT_LEAD, URGENT_SHARE,
                        team_compatible)

# after this much continuous dispatch stalling the window is forcibly released
# (documented deadlock-avoidance fallback of the reference implementation)
STALL_ESCAPE_MIN = 10.0

# populated after each evaluate() for debugging purposes
LAST_DEBUG: dict = {}


@dataclass
class DESResult:
    f1: float                      # makespan (min)
    f2: float                      # transition cost (kCNY)
    f3: float                      # realised load CV across serus
    rho: float                     # constraint-stall ratio (Eq. 13)
    t_cid: float                   # merged stall+recovery time (min)
    events: dict = field(default_factory=dict)   # class -> [durations]
    n_orders: int = 0
    stall_by_class: dict = field(default_factory=dict)
    n_urgent_fail: int = 0         # Eq. 6 violations (urgent past due position)
    n_struct_fail: int = 0         # C1/C4 structural violation marks
    out_order: list = field(default_factory=list)   # order ids by completion
    rel_order: list = field(default_factory=list)   # order ids by release

    def as_objectives(self) -> tuple[float, float, float]:
        return self.f1, self.f2, self.f3


class _Union:
    """Merged interval union on the system timeline."""

    def __init__(self):
        self.iv: list[tuple[float, float]] = []

    def add(self, a: float, b: float):
        if b > a:
            self.iv.append((a, b))

    @property
    def total(self) -> float:
        if not self.iv:
            return 0.0
        seg = sorted(self.iv)
        tot, a0, b0 = 0.0, *seg[0]
        for a, b in seg[1:]:
            if a <= b0:
                b0 = max(b0, b)
            else:
                tot += b0 - a0
                a0, b0 = a, b
        return tot + b0 - a0


class AGVManager:
    """AGV fleet and path-segment capacity with wait/reroute/deadlock rules."""

    def __init__(self, env, inst: Instance, stats: "_Stats"):
        self.env = env
        self.inst = inst
        self.stats = stats
        self.fleet = simpy.Resource(env, capacity=inst.n_agv)
        self.seg_res = {seg: simpy.Resource(env, capacity=cap)
                        for seg, cap in inst.agv_segments.items()}
        self.waiting_for: dict[int, tuple] = {}   # task id -> requested segment
        self.holder: dict[tuple, set] = {}        # segment -> task ids inside

    def _alt_route(self, a: int, b: int, banned: set, fallback: list) -> list:
        """Shortest path avoiding currently full segments (rerouting)."""
        adj: dict[int, list[int]] = {}
        for (u, v) in self.seg_res:
            if (u, v) not in banned:
                adj.setdefault(u, []).append(v)
        prev = {a: None}
        q = [a]
        for u in q:
            if u == b:
                break
            for v in adj.get(u, []):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if b not in prev:
            return fallback
        path, u = [], b
        while prev[u] is not None:
            path.append((prev[u], u))
            u = prev[u]
        return path[::-1]

    def _cycle(self, task: int, seg) -> bool:
        """True if the wait edge task -> holders(seg) closes a cycle."""
        seen, stack = set(), list(self.holder.get(seg, ()))
        while stack:
            h = stack.pop()
            if h == task:
                return True
            if h in seen:
                continue
            seen.add(h)
            nxt = self.waiting_for.get(h)
            if nxt is not None:
                stack.extend(self.holder.get(nxt, ()))
        return False

    def transport(self, a: int, b: int, task: int):
        """Move one workpiece from station a to station b (C3 enforcement).

        The AGV traverses the route segment by segment, holding its current
        segment while it requests the next (hold-and-wait), so cycles in the
        wait-for graph can actually form.  A wait longer than theta_wait on a
        segment is recorded as a C3 stall and the remaining route is replanned
        around the currently full segments — the bypass is then actually
        taken.  A cycle triggers deadlock recovery: the AGV releases its held
        segment (retreat to the spur), waits an exponential recovery time, and
        replans.
        """
        with self.fleet.request() as agv:
            yield agv
            yield self.env.timeout(AGV_LOAD_MIN)
            cur, goal = a, b
            route_override: list | None = None
            held: tuple | None = None          # (segment, request) occupied now
            while cur != goal:
                route = (route_override if route_override is not None
                         else self.inst.agv_routes[(cur, goal)])
                route_override = None          # consumed by this iteration
                for si, seg in enumerate(route):
                    res = self.seg_res[seg]
                    req = res.request()
                    self.waiting_for[task] = seg
                    if self._cycle(task, seg):   # hold-and-wait cycle
                        self.waiting_for.pop(task, None)
                        req.cancel()
                        # retreat: release the held segment so the cycle breaks
                        if held is not None:
                            hseg, hreq = held
                            self.seg_res[hseg].release(hreq)
                            self.holder[hseg].discard(task)
                            held = None
                        t0 = self.env.now
                        rec = float(self.stats.rng.exponential(
                            DEADLOCK_RECOVERY_MEAN_MIN))
                        yield self.env.timeout(rec)
                        self.stats.stall("C3_deadlock", t0, self.env.now)
                        route_override = self._alt_route(cur, goal, set(),
                                                         route[si:])
                        break
                    got = yield req | self.env.timeout(THETA_WAIT_MIN)
                    self.waiting_for.pop(task, None)
                    if req not in got:           # wait exceeded theta_wait
                        t0 = self.env.now
                        if req.triggered:
                            res.release(req)     # granted meanwhile: give back
                        else:
                            req.cancel()
                        full = {s for s, r in self.seg_res.items()
                                if r.count >= r.capacity}
                        # compare against the REMAINING route from cur, not
                        # the full original route — otherwise a no-bypass case
                        # would look like a route change
                        alt = self._alt_route(cur, goal, full, route[si:])
                        if alt != route[si:]:
                            # take the bypass from cur onward; the threshold
                            # crossing itself is recorded as a C3 event
                            self.stats.stall("C3_block", t0, t0)
                            route_override = alt
                            break
                        # no better path: keep waiting on the same segment;
                        # the wait edge stays registered so the wait-for graph
                        # still sees this task during the second wait
                        req2 = res.request()
                        self.waiting_for[task] = seg
                        yield req2
                        self.waiting_for.pop(task, None)
                        # one continuous wait: stall runs from the first
                        # threshold crossing to the grant
                        self.stats.stall("C3_block", t0, self.env.now)
                        req = req2
                    # entered seg: release the segment we came from
                    if held is not None:
                        hseg, hreq = held
                        self.seg_res[hseg].release(hreq)
                        self.holder[hseg].discard(task)
                    self.holder.setdefault(seg, set()).add(task)
                    held = (seg, req)
                    yield self.env.timeout(self.inst.segment_length[seg]
                                           / AGV_SPEED_MPS / 60.0)
                    cur = seg[1]
                # route_override set -> replan from cur; otherwise finished
            if held is not None:
                self.seg_res[held[0]].release(held[1])
                self.holder[held[0]].discard(task)
            yield self.env.timeout(AGV_LOAD_MIN)


class _Stats:
    def __init__(self, rng):
        self.rng = rng
        self.union = _Union()
        self.events: dict[str, list[float]] = {}
        self.by_class: dict[str, float] = {}

    def stall(self, cls: str, a: float, b: float):
        dur = max(0.0, b - a)
        self.events.setdefault(cls, []).append(dur)
        if dur > 0:
            self.union.add(a, b)
            self.by_class[cls] = self.by_class.get(cls, 0.0) + dur


def evaluate(inst: Instance, cfg: Config, x_prev: Config, seed: int,
             backlog: int = 8, n_wip: int = 3,
             time_limit_min: float | None = None) -> DESResult:
    """One Monte-Carlo replication of configuration cfg in operation.

    time_limit_min is a safety valve: if given, the simulation stops at that
    simulated time even if orders remain uncompleted (the result then reflects
    a partial run and should be treated as invalid for scoring)."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    stats = _Stats(rng)

    # structural audit: C1/C4 are hard implementation conditions.  A baseline
    # that skips generation-time checks may hand over a violating config; the
    # DES still runs it counterfactually (operating metrics stay comparable),
    # but every violating unit is recorded as a zero-duration structural event
    # and the config must not count as executable downstream.
    for k in range(cfg.n_seru):
        st = seru_stations(cfg, k)
        if not (is_connected(inst, st) and diameter_ok(inst, st)):
            stats.events.setdefault("C1_struct", []).append(0.0)
        if not team_compatible(inst, int(cfg.team_of[k]), st):
            stats.events.setdefault("C4_struct", []).append(0.0)
    _teams = [int(t) for t in cfg.team_of[: cfg.n_seru]]
    if len(set(_teams)) != len(_teams):
        stats.events.setdefault("C4_struct", []).append(0.0)

    # ---- transition at time zero (Eq. 10 ingredients) ------------------------
    # N_aff compares physical unit identity (largest-overlap mapping), so a
    # pure label renumbering of the same partition is zero-cost.
    best_prev_arr = unit_match_to_prev(cfg, x_prev)
    best_prev = {k: int(best_prev_arr[k]) for k in range(cfg.n_seru)}
    n_aff = int(station_change_mask(cfg, x_prev).sum())
    moved = {k for k in range(cfg.n_seru)
             if int(cfg.team_of[k]) != int(x_prev.team_of[best_prev[k]])}
    n_move = len(moved)
    t_down = 0.0   # transition downtime in minutes (Eq. 10, third term)

    # ---- model data -----------------------------------------------------------
    proc = inst.proc_time / 60.0                     # minutes, (N, M)
    n_models = len(MIX)
    setup_delta = rng.uniform(*SETUP_RANGE_MIN, (n_models, n_models))
    np.fill_diagonal(setup_delta, 0.0)

    # seru visit plan along the process order (maximal runs sharing a seru)
    visits: list[tuple[int, list[int]]] = []
    for i in range(inst.n_stations):
        k = int(cfg.labels[i])
        if visits and visits[-1][0] == k:
            visits[-1][1].append(i)
        else:
            visits.append((k, [i]))

    seru_res = {k: simpy.Resource(env, capacity=1) for k in range(cfg.n_seru)}
    agv = AGVManager(env, inst, stats)
    completion: list[float] = []
    busy = np.zeros(cfg.n_seru)

    # ---- order book -------------------------------------------------------------
    orders: list[dict] = []

    def add_order(arrival: float, urgent: bool | None = None) -> int:
        m = int(rng.choice(n_models, p=MIX))
        idx = len(orders)
        if urgent is None:
            urgent = bool(rng.random() < URGENT_SHARE)
        orders.append({"model": m, "arrival": arrival, "urgent": urgent,
                       "due": idx + URGENT_LEAD if urgent else None})
        return idx

    for _ in range(backlog):
        add_order(0.0)
    for _ in range(rng.poisson(inst.lam_per_min * T_SHIFT_MIN)):
        add_order(float(rng.uniform(0, T_SHIFT_MIN)))
    n_orders = len(orders)

    queue: list[int] = []
    deferred: set[int] = set()        # flushed WIP orders enter the queue late
    task_counter = {"n": 0}

    def arrival_proc():
        for i in sorted((i for i in range(n_orders) if i not in deferred),
                        key=lambda j: orders[j]["arrival"]):
            yield env.timeout(max(0.0, orders[i]["arrival"] - env.now))
            queue.append(i)

    env.process(arrival_proc())

    # ---- vehicle flow ------------------------------------------------------------
    def eta_of(k: int, now: float) -> float:
        if k in moved:
            return 1.0 - (1.0 - LEARNING_ETA0) * np.exp(-now / LEARNING_THETA_MIN)
        return 1.0

    def vehicle_proc(model: int, start_visit: int, start_station: int,
                     order_idx: int = -1):
        nonlocal t_down
        pos_v, pos_s = start_visit, start_station
        prev_exit: int | None = None
        while pos_v < len(visits):
            k, stations = visits[pos_v]
            block = [s for s in stations if s >= pos_s]
            if not block:
                pos_v += 1
                pos_s = 0
                continue
            if prev_exit is not None and prev_exit != block[0]:
                task_counter["n"] += 1
                yield env.process(agv.transport(prev_exit, block[0],
                                                task_counter["n"]))
            with seru_res[k].request() as req:
                yield req
                for s in block:
                    eta = eta_of(k, env.now)
                    pt = proc[s, model] / eta
                    yield env.timeout(pt)
                    if k in moved:
                        t_down += pt - proc[s, model]
                    # busy time integrates over the nominal shift horizon
                    # [0, T_shift] only (Eq. 11); work spilling past the shift
                    # end does not enter the load-balance objective
                    done = env.now
                    busy[k] += max(0.0, min(done, T_SHIFT_MIN)
                                   - min(done - pt, T_SHIFT_MIN))
            prev_exit = block[-1]
            pos_v += 1
            pos_s = 0
        completion.append((order_idx, env.now))

    # ---- WIP: flush across changed boundaries, keep in place otherwise ---------
    wip_kept = 0
    for w in range(n_wip):
        st_idx = min(w, inst.n_stations - 1)
        m = int(rng.choice(n_models, p=MIX))
        k_new = int(cfg.labels[st_idx])
        k_old = int(x_prev.labels[st_idx])
        changed = set(seru_stations(cfg, k_new).tolist()) != \
            set(seru_stations(x_prev, k_old).tolist())
        if changed:
            t_down += T_FLUSH_MIN
            idx = add_order(0.0)               # created eagerly at setup
            deferred.add(idx)
            n_orders += 1

            def flush_proc(order_idx=idx):
                yield env.timeout(T_FLUSH_MIN)   # transfer to central buffer
                queue.append(order_idx)          # rejoins the release queue
            env.process(flush_proc())
        else:
            wip_kept += 1
            vidx = next(v for v, (k, ss) in enumerate(visits) if st_idx in ss)
            env.process(vehicle_proc(m, vidx, st_idx, -1))

    # ---- dispatch under C2 ---------------------------------------------------------
    released_seq: list[int] = []     # released order indices in sequence
    state = {"cum_setup": 0.0}

    def window_feasible() -> list[int]:
        """Feasible candidates within the look-ahead window (Eqs. 4-6)."""
        win = queue[:LOOKAHEAD_W]
        nxt_pos = len(released_seq)
        urgent = [i for i in win if orders[i]["urgent"]
                  and orders[i]["due"] is not None
                  and orders[i]["due"] <= nxt_pos + LOOKAHEAD_W]
        if urgent:
            return urgent[:1]                # spacing exemption (Eq. 6)
        tail = [orders[j]["model"] for j in released_seq]
        ok = [i for i in win if spacing_ok(tail, orders[i]["model"])]
        if not ok:
            return []
        within = []
        prev_m = tail[-1] if tail else -1
        for i in ok:
            d = 0.0 if prev_m < 0 else setup_delta[prev_m, orders[i]["model"]]
            if state["cum_setup"] + d <= SETUP_BUDGET_MIN + 1e-12:
                within.append(i)
        return within or ok

    def dispatcher():
        nonlocal t_down
        released = 0
        stall_start: float | None = None
        while released < n_orders:
            if not queue:
                yield env.timeout(0.05)
                continue
            win = window_feasible()
            if not win:
                if stall_start is None:
                    stall_start = env.now
                yield env.timeout(0.5)
                if queue and not window_feasible():
                    if env.now - stall_start >= STALL_ESCAPE_MIN:
                        # fallback: release the window head, record the violation
                        stats.stall("C2_seq", stall_start, env.now)
                        stall_start = None
                        win = queue[:1]          # falls through to release
                    else:
                        continue
                else:
                    if env.now > stall_start:
                        stats.stall("C2_seq", stall_start, env.now)
                    stall_start = None
                    continue
            elif stall_start is not None:
                stats.stall("C2_seq", stall_start, env.now)
                stall_start = None
            # SPT within the feasible window
            i_sel = min(win, key=lambda i: proc[:, orders[i]["model"]].sum())
            queue.remove(i_sel)
            model = orders[i_sel]["model"]
            prev_m = orders[released_seq[-1]]["model"] if released_seq else -1
            d = 0.0 if prev_m < 0 else float(setup_delta[prev_m, model])
            if d > 0:
                yield env.timeout(d)         # changeover time at release
                # only the NEW excess introduced by this changeover counts;
                # an already-exhausted budget must not backfill earlier time
                prev_over = max(0.0, state["cum_setup"] - SETUP_BUDGET_MIN)
                new_over = max(0.0, state["cum_setup"] + d - SETUP_BUDGET_MIN)
                over = min(new_over - prev_over, d)
                if over > 0:
                    # the excess tail of THIS changeover: at most d minutes,
                    # always inside the current changeover interval
                    stats.stall("C2_setup", env.now - over, env.now)
                state["cum_setup"] += d
            released_seq.append(i_sel)
            if (orders[i_sel]["urgent"] and orders[i_sel]["due"] is not None
                    and len(released_seq) > orders[i_sel]["due"]):
                # Eq. 6 audit: urgent order released past its due position is
                # recorded as a C2 violation (zero-length event mark)
                stats.stall("C2_urgent", env.now, env.now)
            released += 1
            env.process(vehicle_proc(model, 0, 0, i_sel))
        while len(completion) < released + wip_kept:
            yield env.timeout(0.5)

    env.process(dispatcher())
    env.run(until=time_limit_min)

    LAST_DEBUG.clear()
    LAST_DEBUG.update({
        "released": len(released_seq), "n_orders": n_orders,
        "completed": len(completion), "wip_kept": wip_kept,
        "queue": len(queue), "sim_now": env.now,
        "fleet_in_use": agv.fleet.count,
        "waiting_for": dict(agv.waiting_for),
    })

    # ---- metrics ---------------------------------------------------------------------
    f1 = max(t for _, t in completion) if completion else T_SHIFT_MIN
    f3 = float(busy.std() / busy.mean()) if busy.mean() > 0 else 0.0
    t_cid = stats.union.total
    f2 = COST_ALPHA * n_aff + COST_BETA * n_move + COST_GAMMA * t_down / 60.0
    # completion order of order-book orders (auditable against the release
    # order; kept-WIP markers (-1) are excluded from the sequence comparison)
    out_order = [i for i, _ in sorted(completion, key=lambda ct: ct[1])
                 if i >= 0]
    return DESResult(f1=f1, f2=f2, f3=f3, rho=t_cid / f1 if f1 > 0 else 0.0,
                     t_cid=t_cid, events=stats.events, n_orders=n_orders,
                     stall_by_class=stats.by_class,
                     n_urgent_fail=len(stats.events.get("C2_urgent", [])),
                     n_struct_fail=len(stats.events.get("C1_struct", []))
                                  + len(stats.events.get("C4_struct", [])),
                     out_order=out_order, rel_order=list(released_seq))
