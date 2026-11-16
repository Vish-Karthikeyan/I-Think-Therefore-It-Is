"""Principal angles between steering subspaces, against random-subspace baselines."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles


def basis(vectors) -> np.ndarray:
    M = np.column_stack([np.asarray(v, dtype=float) for v in vectors])
    q, _ = np.linalg.qr(M)
    return q


def angles(S1, S2) -> np.ndarray:
    return np.degrees(subspace_angles(np.atleast_2d(S1.T).T if S1.ndim == 1 else S1,
                                      np.atleast_2d(S2.T).T if S2.ndim == 1 else S2))


def random_baseline(d: int, r: int, S2: np.ndarray, n: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([angles(np.linalg.qr(rng.normal(size=(d, r)))[0], S2).min() for _ in range(n)])


def pairwise_directions(dirs: dict[str, np.ndarray], n_random: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Angle between every pair of single directions (e.g., the 7 per-dimension directions,
    or pooled vs. sycophancy/authority/status-quo) with the random-pair 1st percentile."""
    names = [k for k, v in dirs.items() if v is not None]
    if not names:
        return pd.DataFrame()
    d = len(dirs[names[0]])
    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(n_random):
        a, b = rng.normal(size=d), rng.normal(size=d)
        rand.append(np.degrees(np.arccos(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b))))
    rand = np.array(rand)
    rows = []
    for a, b in itertools.combinations(names, 2):
        u, v = dirs[a] / np.linalg.norm(dirs[a]), dirs[b] / np.linalg.norm(dirs[b])
        ang = float(np.degrees(np.arccos(np.clip(abs(u @ v), 0, 1))))
        rows.append(dict(a=a, b=b, angle_deg=ang, cosine=float(u @ v),
                         random_p01=float(np.percentile(rand, 1)), p_vs_random=float((rand <= ang).mean())))
    return pd.DataFrame(rows)


def subspace_comparison(set1: list, set2: list, n_random: int = 1000, seed: int = 0) -> dict:
    S1, S2 = basis(set1), basis(set2)
    ang = angles(S1, S2)
    rand = random_baseline(S1.shape[0], S1.shape[1], S2, n_random, seed)
    return dict(angles=ang, min_angle=float(ang.min()), random_min_p01=float(np.percentile(rand, 1)),
                p_vs_random=float((rand <= ang.min()).mean()))
