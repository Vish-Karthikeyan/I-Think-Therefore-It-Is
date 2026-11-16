"""Step 7: exploratory additions (approved 2026-11-11, predictions logged before the data existed).

    python analysis/07_exploratory.py

Each prediction is checked with the same primary regression as RQ1 (02_rq1_primary.py).

  P0 (main run)  In chat format, belief-following (Delta_b < 0) grows from OLMo SFT to DPO to
                 Instruct, and is weak or absent in completion format.
  P1             Evidence diagnostic (same tables, no persona or belief): models read rates and
                 compare groups above chance, but are near chance on which hypothesis the data
                 support and on the posterior direction.
  P2             Scale: the no-belief evidence slope (beta) exceeds 0.1 for at least one 13-14B
                 instruct model (OLMo 2 13B Instruct, Phi-4).
  P3             Prompting baseline ("judge only from the data" system prompt; OLMo 7B Instruct,
                 Llama): belief-following shrinks but does not vanish; slopes stay near 0.
  P4             Activations for OLMo SFT and DPO: the belief signal is weak at SFT and appears at
                 DPO. NOT computed here: needs the saved activations (FarmShare) - see README.
  P5             Tulu 3 ladder (Llama 3.1 8B base -> SFT -> DPO -> final): belief-following jumps
                 at DPO, as in OLMo.
  P6             Two reworded prompt templates on a fixed third of trials (OLMo 7B and 13B
                 Instruct, Llama, Phi-4): sign of Delta_b and of the support shift unchanged,
                 magnitudes within the original's 95% CI.

Comparisons of a variant with its original model (P3, P6) use the same trials and one stacked
regression in which every term is allowed to differ by variant; the variant's change in Delta_b
is the coefficient on x:contra:variant, with topic-clustered SEs.

Outputs: analysis/output/exploratory/*.csv and predictions.csv (one verdict line per prediction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import (BASE_TERMS, DATA, ROOT, OLMO7_STAGES, OLMO13_STAGES, TULU_STAGES, analysis_frame, estimate,
                    fit_ols, load_dataset, save)

PRIMARY = __import__("02_rq1_primary")


# ------------------------------------------------------------------ helpers
def stacked_variant_test(df: pd.DataFrame, base: str, variant: str, trial_ids=None) -> dict:
    """Same trials, original vs variant model, every term interacted with a variant dummy."""
    d = df[df.model.isin([base, variant]) & (df.fmt == "chat")]
    if trial_ids is not None:
        d = d[d.trial_id.isin(trial_ids)]
    d = analysis_frame(d, 2)
    d["var"] = (d.model == variant).astype(int)
    terms = [t.strip() for t in BASE_TERMS.split("+")]
    if d.cue_type.nunique() < 2:
        terms = [t for t in terms if "cue_type" not in t]
    f = " + ".join(terms + ["var"] + [f"{t}:var" for t in terms])
    fit = fit_ols(d, f)
    out = {"n": int(fit.nobs)}
    for name, term in (("delta_b_change", "x:contra:var"), ("support_shift_change", "x:support:var"),
                       ("beta_change", "x:var"), ("theta_change", "x:contra:m:var")):
        e = estimate(fit, term)
        out.update({name: e["est"], f"{name}_se": e["se"], f"{name}_p": e["p"]})
    return out


def diag_score(diag: pd.DataFrame) -> pd.DataFrame:
    """P1: rate reading, group comparison, hypothesis support, posterior direction."""
    rows = []
    for (m, f), g in diag.groupby(["model", "fmt"]):
        rate = g[g.kind.isin(["rate_t", "rate_c"]) & g.p.notna()]
        est, truth = 100 * rate.p, rate.truth
        err = (est - truth).abs()
        row = dict(model=m, fmt=f, n_items=g.diag_id.nunique(), rate_mae=err.mean(),
                   rate_within5=(err <= 5).mean(), rate_corr=np.corrcoef(est, truth)[0, 1])
        for kind in ("compare", "support"):
            c = g[(g.kind == kind) & g.p_option_A.notna()]
            p_right = np.where(c.truth == 0, c.p_option_A, c.p_option_B)
            k, n = int((p_right > 0.5).sum()), len(c)
            row.update({f"{kind}_acc": k / n, f"{kind}_p_right": p_right.mean(),
                        f"{kind}_p_A": c.p_option_A.mean(),               # position bias toward "A"
                        f"{kind}_binom_p": stats.binomtest(k, n, 0.5, alternative="greater").pvalue})
        post = g[(g.kind == "post") & g.p.notna()]
        lp = np.log(post.p.clip(0.005, 0.995) / (1 - post.p.clip(0.005, 0.995)))
        row["post_slope_vs_x"] = np.polyfit(post.x, lp, 1)[0]
        hit = (np.sign(post.p - 0.5) == np.sign(post.x))
        row["post_dir_acc"] = hit.mean()
        row["post_dir_binom_p"] = stats.binomtest(int(hit.sum()), len(post), 0.5, alternative="greater").pvalue
        row["mean_coverage"] = g.coverage.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(cond: bool | None) -> str:
    return "supported" if cond else ("not supported" if cond is not None else "not tested")


def verdict_parts(parts: list[bool]) -> str:
    """A prediction with several parts: supported only if every part holds."""
    return "supported" if all(parts) else ("not supported" if not any(parts) else "partly supported")


def main() -> None:
    df = load_dataset("main")
    main_df, ext = df[df.study == "main"], df[df.study == "exploratory"]
    both = df  # variants are compared with their original model (main run)
    pm = pd.concat([PRIMARY.per_model(main_df).assign(study="main"),
                    PRIMARY.per_model(ext).assign(study="exploratory")], ignore_index=True)
    save(pm, "exploratory", "per_model_all")
    preds = []

    # ---- P0: OLMo 7B ladder, chat vs completion
    p0 = pd.concat([PRIMARY.stage_contrasts(pm, OLMO7_STAGES, f).assign(ladder="olmo7") for f in ("chat", "completion")])
    save(p0, "exploratory", "P0_olmo7_ladder")
    ch = pm[(pm.fmt == "chat")].set_index("model").delta_b
    co = pm[(pm.fmt == "completion") & pm.model.isin(OLMO7_STAGES)]
    grows = ch.get("olmo_sft", np.nan) > ch.get("olmo_dpo", np.nan) > ch.get("olmo_instruct", np.nan)
    weak_completion = bool((~co.holm_delta_b.astype(bool)).all())
    preds.append(dict(prediction="P0", verdict=verdict(bool(grows and weak_completion)),
                      evidence=f"chat Delta_b SFT {ch.get('olmo_sft'):.3f}, DPO {ch.get('olmo_dpo'):.3f}, "
                               f"Instruct {ch.get('olmo_instruct'):.3f}; completion significant after Holm: "
                               f"{int(co.holm_delta_b.astype(bool).sum())} of {len(co)}"))

    # ---- P1: evidence diagnostic
    diag = pd.read_csv(DATA / "evidence_diagnostic.csv.gz")
    p1 = diag_score(diag)
    save(p1, "exploratory", "P1_evidence_diagnostic")
    # Three parts, each judged on the chat-format models (majority rule):
    #   (a) rates read: at least 80% of read percentages within 5 points of the truth
    #   (b) groups compared above chance: one-sided binomial p < .05
    #   (c) support and posterior direction near chance: accuracy below 0.6 on both
    chat = p1[p1.fmt == "chat"]
    a_ok = chat.rate_within5 >= 0.8
    b_ok = (chat.compare_acc > 0.5) & (chat.compare_binom_p < 0.05)
    c_ok = (chat.support_acc < 0.6) & (chat.post_dir_acc < 0.6)
    parts = [bool(a_ok.mean() > 0.5), bool(b_ok.mean() > 0.5), bool(c_ok.mean() > 0.5)]
    preds.append(dict(prediction="P1", verdict=verdict_parts(parts),
                      evidence=f"chat models (n={len(chat)}): (a) rates read within 5 points in {int(a_ok.sum())}, "
                               f"median MAE {chat.rate_mae.median():.1f} points; (b) group comparison above chance in "
                               f"{int(b_ok.sum())}, accuracy {chat.compare_acc.min():.2f}-{chat.compare_acc.max():.2f}; "
                               f"(c) support and posterior near chance in {int(c_ok.sum())} (support median "
                               f"{chat.support_acc.median():.2f}, posterior direction median {chat.post_dir_acc.median():.2f}); "
                               f"exception: Phi-4 (compare {chat.set_index('model').compare_acc.get('phi4', np.nan):.2f})"))

    # ---- P2: scale
    big = pm[pm.model.isin(["olmo13_instruct", "phi4"]) & (pm.fmt == "chat")]
    save(big, "exploratory", "P2_scale")
    preds.append(dict(prediction="P2", verdict=verdict(bool((big.beta_neutral > 0.1).any()) if len(big) else None),
                      evidence="; ".join(f"{r.model} beta {r.beta_neutral:.3f}" for r in big.itertuples())))
    save(pd.concat([PRIMARY.stage_contrasts(pm, OLMO13_STAGES, f) for f in ("chat", "completion")]),
         "exploratory", "P2_olmo13_ladder")

    # ---- P3: debiasing system prompt
    rows = []
    for base in ("olmo_instruct", "llama"):
        r = stacked_variant_test(both, base, f"{base}_sp")
        b0 = pm[(pm.model == base) & (pm.fmt == "chat")].iloc[0]
        b1 = pm[(pm.model == f"{base}_sp") & (pm.fmt == "chat")].iloc[0]
        rows.append(dict(model=base, delta_b_original=b0.delta_b, delta_b_sp=b1.delta_b,
                         delta_b_sp_p=b1.delta_b_p, beta_original=b0.beta_neutral, beta_sp=b1.beta_neutral, **r))
    p3 = pd.DataFrame(rows)
    save(p3, "exploratory", "P3_system_prompt")
    # Parts: (a) Delta_b shrinks significantly (stacked test p < .05, change > 0) in both models;
    # (b) it does not vanish (still significant with the prompt); (c) beta stays near 0 (|beta| < 0.1)
    parts = [bool(((p3.delta_b_change > 0) & (p3.delta_b_change_p < 0.05)).all()),
             bool((p3.delta_b_sp_p < 0.05).all()), bool((p3.beta_sp.abs() < 0.1).all())]
    preds.append(dict(prediction="P3", verdict=verdict_parts(parts),
                      evidence="; ".join(f"{r.model}: Delta_b {r.delta_b_original:.3f} -> {r.delta_b_sp:.3f} "
                                         f"(change {r.delta_b_change:+.3f}, p={r.delta_b_change_p:.3g}), beta "
                                         f"{r.beta_original:.3f} -> {r.beta_sp:.3f}" for r in p3.itertuples())))

    # ---- P4: activations (computed on FarmShare by 10_p4_stage_probes.py; results/p4/)
    # Parts, on the later layers (18-26, where the signal is strongest): (a) the contradiction probe is
    # weaker at SFT than at DPO (AUC at least 0.03 lower); (b) DPO's bias direction is closer to
    # Instruct's than SFT's is. Caveat: those activations were recorded before the answer prefix
    # (see the RQ3 implementation note), like the original RQ3.
    p4f = ROOT / "results" / "p4" / "p4_stage_probes.csv"
    if p4f.exists():
        p4 = pd.read_csv(p4f)
        late = p4[p4.layer >= 18].groupby("stage")[["probe_acc", "probe_auc", "majority",
                                                     "cos_hat_k8_to_instruct", "cos_raw_to_instruct"]].mean()
        save(late.reset_index(), "exploratory", "P4_stage_probes_late_layers")
        save(p4, "exploratory", "P4_stage_probes_all_layers", show=False)
        a = late.loc["olmo_dpo", "probe_auc"] - late.loc["olmo_sft", "probe_auc"] >= 0.03
        b = late.loc["olmo_dpo", "cos_raw_to_instruct"] > late.loc["olmo_sft", "cos_raw_to_instruct"]
        preds.append(dict(prediction="P4", verdict=verdict_parts([bool(a), bool(b)]),
                          evidence=f"layers 18-26: probe AUC SFT {late.loc['olmo_sft', 'probe_auc']:.3f}, DPO "
                                   f"{late.loc['olmo_dpo', 'probe_auc']:.3f}, Instruct {late.loc['olmo_instruct', 'probe_auc']:.3f}; "
                                   f"cosine to Instruct's direction SFT {late.loc['olmo_sft', 'cos_raw_to_instruct']:+.2f}, "
                                   f"DPO {late.loc['olmo_dpo', 'cos_raw_to_instruct']:+.2f} (Instruct, labels only vs its saved "
                                   f"direction, {late.loc['olmo_instruct', 'cos_raw_to_instruct']:+.2f}): the signal is present from SFT on"))
    else:
        preds.append(dict(prediction="P4", verdict="not tested here",
                          evidence="run analysis/10_p4_stage_probes.py where the activations are"))

    # ---- P5: Tulu 3 ladder
    p5 = pd.concat([PRIMARY.stage_contrasts(pm, TULU_STAGES, f) for f in ("chat", "completion")])
    save(p5, "exploratory", "P5_tulu_ladder")
    # "Jumps at DPO": the SFT -> DPO step in Delta_b is negative and significant (same trials,
    # stacked test). The DPO -> final step is reported alongside (OLMo's two steps were similar).
    tc = pm[(pm.fmt == "chat")].set_index("model")
    step1 = stacked_variant_test(both, "tulu_sft", "tulu_dpo")
    step2 = stacked_variant_test(both, "tulu_dpo", "tulu_final")
    o1 = stacked_variant_test(both, "olmo_sft", "olmo_dpo")
    o2 = stacked_variant_test(both, "olmo_dpo", "olmo_instruct")
    steps = pd.DataFrame([dict(ladder=l, step=s, **r) for l, s, r in
                          (("tulu", "sft->dpo", step1), ("tulu", "dpo->final", step2),
                           ("olmo7", "sft->dpo", o1), ("olmo7", "dpo->instruct", o2))])
    save(steps, "exploratory", "P5_P0_stage_steps")
    jump = bool(step1["delta_b_change"] < 0 and step1["delta_b_change_p"] < 0.05)
    preds.append(dict(prediction="P5", verdict=verdict(jump),
                      evidence=f"chat Delta_b SFT {tc.delta_b['tulu_sft']:.3f}, DPO {tc.delta_b['tulu_dpo']:.3f}, "
                               f"final {tc.delta_b['tulu_final']:.3f}; SFT->DPO change {step1['delta_b_change']:+.3f} "
                               f"(p={step1['delta_b_change_p']:.2g}); DPO->final change {step2['delta_b_change']:+.3f} "
                               f"(p={step2['delta_b_change_p']:.2g}): the drop continues after DPO"))

    # ---- P6: reworded prompts, same third of trials, compared with the original model
    rows = []
    for base in ("olmo_instruct", "llama", "olmo13_instruct", "phi4"):
        for v in (1, 2):
            var = f"{base}_pv{v}"
            ids = set(both[(both.model == var)].trial_id)
            if not ids:
                continue
            orig = PRIMARY.per_model(both[(both.model == base) & both.trial_id.isin(ids)])
            new = PRIMARY.per_model(both[both.model == var])
            o, n = orig[orig.fmt == "chat"].iloc[0], new.iloc[0]
            if "delta_b" not in n or pd.isna(n.get("delta_b")):
                rows.append(dict(model=base, variant=var, note=n.get("note")))
                continue
            st = stacked_variant_test(both, base, var, ids)
            rows.append(dict(model=base, variant=var, n_original=o.n, n_variant=n.n,
                             delta_b_original=o.delta_b, delta_b_lo=o.delta_b_lo, delta_b_hi=o.delta_b_hi,
                             delta_b_variant=n.delta_b,
                             support_original=o.support_shift, support_variant=n.support_shift,
                             same_sign_delta_b=np.sign(o.delta_b) == np.sign(n.delta_b),
                             same_sign_support=np.sign(o.support_shift) == np.sign(n.support_shift),
                             within_ci=o.delta_b_lo <= n.delta_b <= o.delta_b_hi, **st))
    p6 = pd.DataFrame(rows)
    save(p6, "exploratory", "P6_prompt_wording")
    ok = p6.dropna(subset=["same_sign_delta_b"]) if "same_sign_delta_b" in p6 else p6
    preds.append(dict(prediction="P6", verdict=verdict_parts([bool((ok.same_sign_delta_b & ok.same_sign_support).all()),
                                                              bool(ok.within_ci.all())]),
                      evidence=f"{int((ok.same_sign_delta_b & ok.same_sign_support).sum())}/{len(ok)} variants keep both signs; "
                               f"{int(ok.within_ci.sum())}/{len(ok)} Delta_b within the original's 95% CI"))

    save(pd.DataFrame(preds), "exploratory", "predictions")


if __name__ == "__main__":
    main()
