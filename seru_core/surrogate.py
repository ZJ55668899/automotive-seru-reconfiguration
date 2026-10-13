"""Surrogate models with error control (Section 4.4).

One ridge-regression model per objective, trained on the archive of completed
DES evaluations.  The closed-form leave-one-out residual of ridge regression
provides the predictive uncertainty used by the infill criterion (Eq. 14).
"""
from __future__ import annotations

import numpy as np

from .constraints import Config, seru_stations
from .instances import D_MAX_M, Instance, MODEL_COEF, MIX


def features(inst: Instance, cfg: Config, x_prev: Config) -> np.ndarray:
    """The 13 features extracted from the attribute graph (Section 4.4)."""
    sizes = np.array([len(seru_stations(cfg, k)) for k in range(cfg.n_seru)], float)
    k_cnt = float(cfg.n_seru)
    # inter-seru edge density of the attribute graph
    inter = 0
    for i in range(inst.n_stations):
        for j in inst.adjacency[i]:
            if j > i and cfg.labels[i] != cfg.labels[j]:
                inter += 1
    density = inter / max(1, len(inst.agv_segments))
    # analytical load per seru and its CV
    mix_coef = float(np.dot(MIX, MODEL_COEF))
    loads = np.array([inst.base_time[seru_stations(cfg, k)].sum() * mix_coef
                      for k in range(cfg.n_seru)])
    load_cv = float(loads.std() / loads.mean()) if loads.mean() > 0 else 0.0
    # team-station skill margins (mean and min over assigned teams)
    margins = []
    for k in range(cfg.n_seru):
        t = int(cfg.team_of[k])
        st = seru_stations(cfg, k)
        margins.append((inst.team_skill[t][None, :] - inst.skill_req[st]).min(0).mean())
        margins.append((inst.team_skill[t][None, :] - inst.skill_req[st]).min())
    margins = np.asarray(margins)
    margin_mean = float(margins[0::2].mean())
    margin_min = float(margins[1::2].min())
    # normalised seru diameters (mean and max)
    diams = []
    for k in range(cfg.n_seru):
        pts = inst.coords[seru_stations(cfg, k)]
        if len(pts) <= 1:
            diams.append(0.0)
        else:
            d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)).max()
            diams.append(d / D_MAX_M)
    diams = np.asarray(diams)
    # analytical AGV segment utilisation (mean and max over segments)
    flow = np.zeros(len(inst.agv_segments))
    segs = list(inst.agv_segments)
    seg_idx = {s: i for i, s in enumerate(segs)}
    for i in range(inst.n_stations - 1):
        if cfg.labels[i] == cfg.labels[i + 1]:
            continue
        for seg in inst.agv_routes[(i, i + 1)]:
            if seg in seg_idx:
                flow[seg_idx[seg]] += 1.0
    caps = np.array([inst.agv_segments[s] for s in segs], float)
    util = flow / np.maximum(caps, 1)
    util_mean, util_max = float(util.mean()), float(util.max())
    hamming = float(np.mean(cfg.labels != x_prev.labels))
    return np.array([k_cnt, sizes.mean(), sizes.std(), sizes.max(), density,
                     load_cv, margin_mean, margin_min, diams.mean(), diams.max(),
                     util_mean, util_max, hamming])


class RidgeSurrogate:
    """Ridge regression with closed-form leave-one-out uncertainty."""

    def __init__(self, n_objectives: int = 3):
        self.n_obj = n_objectives
        self.lams = np.ones(n_objectives)
        self.coef: np.ndarray | None = None
        self.x_mean = np.zeros(0)
        self.x_std = np.ones(0)
        self.y_mean = np.zeros(0)
        self.y_std = np.ones(0)
        self.s_loo = np.ones(n_objectives)
        self.n_train = 0

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        self.n_train = len(x)
        self.x_mean, self.x_std = x.mean(0), x.std(0) + 1e-12
        self.y_mean, self.y_std = y.mean(0), y.std(0) + 1e-12
        z = (x - self.x_mean) / self.x_std
        w = (y - self.y_mean) / self.y_std
        p = z.shape[1]
        coef = np.zeros((p, self.n_obj))
        for j in range(self.n_obj):
            lam = self._cv_lambda(z, w[:, j])
            self.lams[j] = lam
            a = z.T @ z + lam * np.eye(p)
            coef[:, j] = np.linalg.solve(a, z.T @ w[:, j])
            # closed-form leave-one-out residuals via the hat matrix
            h = z @ np.linalg.solve(a, z.T)
            resid = w[:, j] - z @ coef[:, j]
            loo = resid / np.maximum(1e-9, 1.0 - np.diag(h))
            self.s_loo[j] = float(loo.std() + 1e-9)
        self.coef = coef

    def _cv_lambda(self, z: np.ndarray, w: np.ndarray) -> float:
        """Five-fold cross-validation over a small lambda grid (Section 4.4)."""
        best, best_err = 1.0, np.inf
        n = len(z)
        p = z.shape[1]
        for lam in (0.01, 0.1, 1.0, 10.0, 100.0):
            errs = []
            for fold in range(5):
                idx = np.arange(n) % 5 == fold
                zt, zv = z[~idx], z[idx]
                wt, wv = w[~idx], w[idx]
                if len(zt) < p + 1 or len(zv) == 0:
                    continue
                a = zt.T @ zt + lam * np.eye(p)
                c = np.linalg.solve(a, zt.T @ wt)
                errs.append(np.abs(zv @ c - wv).mean())
            err = float(np.mean(errs)) if errs else np.inf
            if err < best_err:
                best, best_err = lam, err
        return best

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted objectives and per-point uncertainty (Eq. 14 terms)."""
        x = np.atleast_2d(x)
        z = (x - self.x_mean) / self.x_std
        y = (z @ self.coef) * self.y_std + self.y_mean
        u = np.zeros((len(x), self.n_obj))
        gram_inv = getattr(self, "_gram_inv", None)
        if gram_inv is not None:
            for j in range(self.n_obj):
                q = np.einsum('bi,ij,bj->b', z, gram_inv[j], z)
                u[:, j] = self.s_loo[j] * np.sqrt(np.maximum(0.0, 1.0 + q))
        return y, u

    def fit_with_gram(self, x: np.ndarray, y: np.ndarray):
        """fit() that also stores the inverse Gram matrices for uncertainty."""
        self.fit(x, y)
        z = ((np.asarray(x, float) - self.x_mean) / self.x_std)
        p = z.shape[1]
        self._gram_inv = [np.linalg.inv(z.T @ z + self.lams[j] * np.eye(p))
                          for j in range(self.n_obj)]
        return self
