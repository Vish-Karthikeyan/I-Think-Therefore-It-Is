"""RQ2b (exploratory): behavioral signatures (Section 5.2).

Table: identities (x cue type) as rows, the 8 conditions (source x belief x evidence
direction) as columns; each cell is the slope shift relative to that identity's
no-belief slope, estimated without pooling, with its SE. Each row is summarized by four
preregistered contrasts (contra shift, supported shift, affirm - deny, lab - tank).
Residual structure: weighted low-rank SVD (rank 1-5 by CV), varimax-rotated; bootstrap
stability with orthogonal Procrustes alignment (interpret only if median alignment
correlation >= 0.8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.linalg import orthogonal_procrustes

CONDITIONS = [(s, b, d) for s in ("lab", "tank") for b in (1, -1) for d in (1, -1)]


def _cond_name(c):
    s, b, d = c
    return f"{s}_b{'+' if b > 0 else '-'}_d{'+' if d > 0 else '-'}"


def identity_table(d: pd.DataFrame, row_keys=("group", "cue_type"), min_neutral: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per row: lp_after ~ lp_before + x + sum_c x:I[cond=c] (c over b != 0 conditions)
    -> coefficient = slope shift of condition c vs. the row's own b=0 slope."""
    d = d[(d.e != "null")].copy()
    for c in CONDITIONS:
        s, b, dd = c
        d[_cond_name(c)] = ((d.source == s) & (d.b == b) & (d.d == dd)).astype(float) * d.x
    est, se = {}, {}
    for key, g in d.groupby(list(row_keys), observed=True):
        cols = [_cond_name(c) for c in CONDITIONS if g[_cond_name(c)].abs().sum() > 0]
        if (g.b == 0).sum() < min_neutral or not cols:
            continue
        fit = smf.ols("lp_after ~ lp_before + x + " + " + ".join(f"Q('{c}')" for c in cols), g).fit()
        est[key] = {c: fit.params.get(f"Q('{c}')", np.nan) for c in cols}
        se[key] = {c: fit.bse.get(f"Q('{c}')", np.nan) for c in cols}
    cols = [_cond_name(c) for c in CONDITIONS]
    idx = pd.MultiIndex.from_tuples(list(est), names=list(row_keys)) if est else \
        pd.MultiIndex.from_tuples([], names=list(row_keys))
    E = pd.DataFrame([est[k] for k in est], index=idx).reindex(columns=cols)
    S = pd.DataFrame([se[k] for k in se], index=idx).reindex(columns=cols)
    return E, S


def contrast_matrix() -> pd.DataFrame:
    """Columns: the four contrasts as patterns over the 8 conditions."""
    rows = []
    for c in CONDITIONS:
        s, b, d = c
        rows.append(dict(cond=_cond_name(c), contra=float(b * d < 0), supp=float(b * d > 0),
                         affirm_deny=float(b), lab_tank=1.0 if s == "lab" else -1.0))
    return pd.DataFrame(rows).set_index("cond")


def row_contrasts(E: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """Precision-weighted contrasts with SEs (notes' `contrasts`)."""
    out = []
    for idx in E.index:
        row, se = E.loc[idx], S.loc[idx]
        ok = row.notna() & se.notna() & (se > 0)
        row, se = row[ok], se[ok]
        w = 1 / se ** 2
        conds = {c: dict(zip(("s", "b", "d"), _parse(c))) for c in row.index}

        def wavg(sel):
            sel = [c for c in row.index if sel(conds[c])]
            if not sel:
                return np.nan, np.nan
            ww = w[sel]
            return float((row[sel] * ww).sum() / ww.sum()), float(1 / np.sqrt(ww.sum()))

        contra = wavg(lambda c: c["b"] * c["d"] < 0)
        supp = wavg(lambda c: c["b"] * c["d"] > 0)
        aff, den = wavg(lambda c: c["b"] > 0), wavg(lambda c: c["b"] < 0)
        lab, tank = wavg(lambda c: c["s"] == "lab"), wavg(lambda c: c["s"] == "tank")
        rec = dict(zip(E.index.names, idx if isinstance(idx, tuple) else (idx,)))
        rec.update(contra_shift=contra[0], contra_se=contra[1], supp_shift=supp[0], supp_se=supp[1],
                   affirm_minus_deny=aff[0] - den[0], affirm_minus_deny_se=np.hypot(aff[1], den[1]),
                   lab_minus_tank=lab[0] - tank[0], lab_minus_tank_se=np.hypot(lab[1], tank[1]))
        out.append(rec)
    res = pd.DataFrame(out)
    res["signature"] = res.apply(classify, axis=1)
    return res


def _parse(name: str):
    s, b, d = name.split("_")
    return s, (1 if b[1] == "+" else -1), (1 if d[1] == "+" else -1)


def classify(r, z: float = 1.96) -> str:
    """Signature regions of (contra_shift, supp_shift)."""
    c_neg = r.contra_shift < -z * r.contra_se
    s_pos = r.supp_shift > z * r.supp_se
    s_neg = r.supp_shift < -z * r.supp_se
    s_zero = not s_pos and not s_neg
    if c_neg and s_neg and abs(r.contra_shift - r.supp_shift) < z * np.hypot(r.contra_se, r.supp_se):
        return "general_hedging"
    if c_neg and s_zero:
        return "contradiction_discounting"
    if s_pos and not c_neg:
        return "confirmation_boost"
    if c_neg and s_pos:
        return "discounting+confirmation"
    if c_neg:
        return "discounting(other)"
    return "none"


def residual_table(E: pd.DataFrame, S: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the part of each row explained by the four contrast patterns (WLS per row)."""
    C = contrast_matrix().loc[E.columns].values
    R = E.copy() * np.nan
    for idx in E.index:
        y, se = E.loc[idx].values, S.loc[idx].values
        ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
        if ok.sum() <= C.shape[1]:
            continue
        w = 1 / se[ok]
        coef = np.linalg.lstsq(C[ok] * w[:, None], y[ok] * w, rcond=None)[0]
        R.loc[idx, E.columns[ok]] = y[ok] - C[ok] @ coef
    return R, S


def weighted_lowrank(R: np.ndarray, W: np.ndarray, rank: int, iters: int = 200, seed: int = 0):
    """min sum W (R - U V^T)^2 by alternating weighted least squares (missing: W=0)."""
    rng = np.random.default_rng(seed)
    n, m = R.shape
    R0 = np.nan_to_num(R)
    U = rng.normal(size=(n, rank)) * 0.1
    V = rng.normal(size=(m, rank)) * 0.1
    lam = 1e-6
    for _ in range(iters):
        for i in range(n):
            Wi = W[i]
            A = (V * Wi[:, None]).T @ V + lam * np.eye(rank)
            U[i] = np.linalg.solve(A, (V * Wi[:, None]).T @ R0[i])
        for j in range(m):
            Wj = W[:, j]
            A = (U * Wj[:, None]).T @ U + lam * np.eye(rank)
            V[j] = np.linalg.solve(A, (U * Wj[:, None]).T @ R0[:, j])
    return U, V


def varimax(L: np.ndarray, gamma: float = 1.0, iters: int = 100, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    p, k = L.shape
    Rm = np.eye(k)
    d = 0
    for _ in range(iters):
        Lr = L @ Rm
        u, s, vh = np.linalg.svd(L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag(np.sum(Lr ** 2, axis=0))))
        Rm = u @ vh
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return L @ Rm, Rm


def cv_rank(R: pd.DataFrame, S: pd.DataFrame, ranks=range(1, 6), folds: int = 5, seed: int = 0) -> pd.DataFrame:
    Rv = R.values.astype(float)
    W = np.where(np.isfinite(Rv) & np.isfinite(S.values), 1 / np.square(S.values.astype(float)), 0.0)
    obs = np.argwhere(W > 0)
    rng = np.random.default_rng(seed)
    fold = rng.integers(folds, size=len(obs))
    rows = []
    max_rank = min(R.shape) - 1
    for r in ranks:
        if r > max_rank:
            break
        err = 0.0
        for f in range(folds):
            Wt = W.copy()
            held = obs[fold == f]
            Wt[held[:, 0], held[:, 1]] = 0
            U, V = weighted_lowrank(Rv, Wt, r, iters=60, seed=seed)
            P = U @ V.T
            err += float(np.sum(W[held[:, 0], held[:, 1]] * (np.nan_to_num(Rv)[held[:, 0], held[:, 1]]
                                                               - P[held[:, 0], held[:, 1]]) ** 2))
        rows.append(dict(rank=r, cv_error=err))
    # rank 0 baseline
    base = float(np.sum(W * np.nan_to_num(Rv) ** 2))
    return pd.DataFrame([dict(rank=0, cv_error=base)] + rows)


def residual_components(R: pd.DataFrame, S: pd.DataFrame, rank: int, seed: int = 0):
    Rv = R.values.astype(float)
    W = np.where(np.isfinite(Rv) & np.isfinite(S.values), 1 / np.square(S.values.astype(float)), 0.0)
    U, V = weighted_lowrank(Rv, W, rank, seed=seed)
    Vr, rot = varimax(V)
    Ur = U @ np.linalg.inv(rot).T
    return pd.DataFrame(Ur, index=R.index), pd.DataFrame(Vr, index=R.columns)


def bootstrap(d: pd.DataFrame, rank: int, n_boot: int = 1000, seed: int = 0, row_keys=("group", "cue_type"),
              min_neutral: int = 10):
    """Resample trials within rows, recompute the table, contrasts and residual components;
    Procrustes-align condition loadings to the original."""
    rng = np.random.default_rng(seed)
    E, S = identity_table(d, row_keys, min_neutral)
    R, _ = residual_table(E, S)
    base_U, base_V = residual_components(R, S, rank, seed) if rank > 0 else (None, None)
    contrasts_boot, align = [], []
    groups = list(d.groupby(list(row_keys), observed=True).indices.values())
    for bidx in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        db = d.iloc[idx]
        Eb, Sb = identity_table(db, row_keys, min_neutral)
        cb = row_contrasts(Eb, Sb)
        cb["boot"] = bidx
        contrasts_boot.append(cb)
        if rank > 0:
            Rb, _ = residual_table(Eb.reindex(E.index), Sb.reindex(E.index))
            Rb = Rb.fillna(0.0)
            Sb2 = Sb.reindex(E.index).fillna(1e3)
            _, Vb = residual_components(Rb, Sb2, rank, seed)
            Q, _ = orthogonal_procrustes(Vb.values, base_V.values)
            Va = Vb.values @ Q
            align.append([np.corrcoef(Va[:, k], base_V.values[:, k])[0, 1] for k in range(rank)])
    out = dict(contrasts=pd.concat(contrasts_boot, ignore_index=True))
    if rank > 0:
        A = np.array(align)
        out["alignment_median"] = np.median(A, axis=0)
        out["stable"] = out["alignment_median"] >= 0.8
    return out


def signature_map(contrasts: pd.DataFrame, boot: pd.DataFrame | None = None, path=None, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..cues import GROUP_TO_DIM
    dims = sorted(set(GROUP_TO_DIM.values()))
    cmap = plt.get_cmap("tab10")
    colors = {k: cmap(i) for i, k in enumerate(dims)}
    cts = sorted(contrasts.cue_type.unique()) if "cue_type" in contrasts else [None]
    fig, axes = plt.subplots(1, len(cts), figsize=(6.5 * len(cts), 6), squeeze=False)
    for ax, ct in zip(axes[0], cts):
        c = contrasts if ct is None else contrasts[contrasts.cue_type == ct]
        if boot is not None:
            b = boot if ct is None else boot[boot.cue_type == ct]
            for g, bg in b.groupby("group"):
                ax.scatter(bg.contra_shift, bg.supp_shift, s=2, alpha=0.08, color=colors.get(GROUP_TO_DIM.get(g), "gray"))
        for _, r in c.iterrows():
            col = colors.get(GROUP_TO_DIM.get(r.group), "gray")
            ax.errorbar(r.contra_shift, r.supp_shift, xerr=1.96 * r.contra_se, yerr=1.96 * r.supp_se,
                        fmt="o", color=col, ms=5, lw=0.8)
            ax.annotate(r.group, (r.contra_shift, r.supp_shift), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        lim = np.nanmax(np.abs(np.r_[c.contra_shift, c.supp_shift])) * 1.3 + 0.02
        ax.plot([-lim, 0], [-lim, 0], color="gray", ls=":", lw=0.8)
        ax.set_xlabel("contradicted-trial slope shift"); ax.set_ylabel("supported-trial slope shift")
        ax.set_title(f"{title or ''} {ct or ''}".strip(), fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[k], label=k) for k in dims]
    fig.legend(handles=handles, loc="lower center", ncol=len(dims), fontsize=8)
    fig.suptitle("Identity loadings on behavioral signatures of evidence-weighting distortion", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    if path:
        fig.savefig(path, dpi=140)
    return fig


def run(df: pd.DataFrame, model: str, fmt: str = "chat", rq2_success: dict | None = None,
        n_boot: int = 1000, seed: int = 0, force: bool = False, min_neutral: int = 10) -> dict:
    """Gate: RQ2 must beat both baselines on at least the held-out-topic split."""
    if not force and rq2_success is not None and not rq2_success.get("held_out_topics", False):
        return dict(gated_out=True)
    d = df[(df.model == model) & (df.fmt == fmt) & ~df.excluded & (df.tier == 2)]
    E, S = identity_table(d, min_neutral=min_neutral)
    if E.empty:
        return dict(gated_out=False, empty=True, note=f"no identity row has {min_neutral}+ no-belief trials")
    con = row_contrasts(E, S)
    R, _ = residual_table(E, S)
    cv = cv_rank(R.fillna(np.nan), S)
    best = int(cv.loc[cv.cv_error.idxmin(), "rank"])
    comps = residual_components(R.fillna(0.0), S.where(R.notna(), 1e3), best, seed) if best > 0 else None
    boot = bootstrap(d, best, n_boot=n_boot, seed=seed, min_neutral=min_neutral)
    return dict(table=E, se=S, contrasts=con, cv=cv, rank=best, components=comps, bootstrap=boot)


def replication(results: dict[str, dict]) -> pd.DataFrame:
    """Compare contrasts across models (a pattern seen in one model only is a property of
    that model)."""
    frames = [r["contrasts"].assign(model=m) for m, r in results.items() if "contrasts" in r]
    if len(frames) < 2:
        return pd.DataFrame()
    c = pd.concat(frames)
    wide = c.pivot_table(index=["group", "cue_type"], columns="model", values=["contra_shift", "supp_shift"])
    out = []
    for stat in ("contra_shift", "supp_shift"):
        w = wide[stat].dropna()
        cols = list(w.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                out.append(dict(stat=stat, a=cols[i], b=cols[j], r=w[cols[i]].corr(w[cols[j]]), n=len(w)))
    return pd.DataFrame(out)
