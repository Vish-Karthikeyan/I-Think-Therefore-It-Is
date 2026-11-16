"""Preregistration document generated from the config (Section 4.7 checklist). Fill the
bracketed fields (exact revisions, run dates, pilot-derived sample sizes) before posting
to OSF / AsPredicted. Pilot data are used only to set sample sizes and x ranges."""
from __future__ import annotations

from .cues import DIMENSIONS
from .models import get_model
from .pipeline import run_dir


def _k(cfg, model):
    from .pipeline import sample_k_for
    return sample_k_for(cfg, model)


def write_prereg(cfg: dict) -> str:
    rd = run_dir(cfg)
    d = cfg.get("design") or {}
    models = cfg.get("models") or []
    lines = [
        "# IDEA preregistration (draft generated from config)",
        "",
        f"Config: `{cfg['name']}`; run folder `{rd.name}`; seed {cfg.get('seed', 0)}.",
        "",
        "## Co-primary estimands (Tier 2; Holm-corrected at alpha = 0.05)",
        "- Delta_b = beta(kappa=1) - beta(b=0): coefficient on x:contra.",
        "- theta_cong: coefficient on x:contra:m, m = b * g(c, H1).",
        "- Model: lp_after ~ lp_before + x + x:contra + x:support + x:contra:m + x:support:m + x:dpos"
        " + x:cue_type + x:source + (1|item) + (0+x|topic) + (0+x:contra:m|dimension).",
        "- 'No' for RQ1: both co-primary 95% intervals include zero.",
        "",
        "## Secondary",
        "- delta_j for all 22 groups vs. Unspecified (Eq. 6); within-dimension equality Wald tests; Tier 1 delta_j.",
        "- Contrasts by cue type, survey congruence m_svy, behavioral stance g_beh (robustness), tier amplification.",
        "- Supporting-trial slopes reported separately.",
        "- Secondary specifications reported regardless: spline in x (df=4), two-component mixture (update vs. ignore),"
        " latent-prior measurement model (replicate as instrument).",
        "- Every estimand per model, then jointly with model as a factor (Wald test of x:contra:model and"
        " x:contra:m:model); OLMo stage contrasts in completion format.",
        "",
        "## Models (record exact version strings and run dates)",
    ]
    for m in models:
        s = get_model(m)
        lines.append(f"- {m}: {s.hf_id or s.openrouter_id} revision [{s.revision} -> fill sha]; formats {list(s.formats)};"
                     f" readout {s.readout}")
    lines += [
        f"- Reduced design (explicit labels only): {cfg.get('reduced_models') or 'none'}",
        f"- Sampling readout (API models without logprobs): K = {_k(cfg, 'claude')} samples at temperature 1 for Claude.",
        "- Readout calibration subset: open models also read out by K=10 sampling at temperature 1"
        f" (fraction {d.get('calibration_frac', 0)}).",
        "",
        "## Identities and cues",
        "- 22 groups: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in DIMENSIONS.items()) + ".",
        "- Explicit labels and 4 vignettes per group; Unspecified neutral cues of matched length.",
        "- Vignette pass threshold: >= 0.7 probability on the target group in a fresh-context identification call;"
        " leakage: any other dimension with top-option probability >= 0.5 is recorded (covariate) or rewritten.",
        "",
        "## Topics and pairing",
        f"- Topic screen: keep a Tier 2 topic if >= 2 dimensions have some group with |g| >= 0.5; at least 3 topics"
        f" per dimension. Tier 1: all topics kept (deviation approved 2026-11-11: the |g| < 0.25 rule removed every"
        f" Tier 1 topic in the pilot); the rule is applied as a robustness subset of (topic, cue) pairs with"
        f" |g| < 0.25, averaged over pilot models (IDEA_TIER1_RULE=1).",
        f"- Pairing rule: share {d.get('p_match', 0.75)} of trials on topics that engage the identity's dimension"
        " (from the screen); the rest on non-engaged topics.",
        "- g definition: logit p_cue - logit p_unspecified (matched cue type), per model, per H1 direction.",
        "- v_k(t): highest minus lowest group support on the matched Pew item, percentage points; missing flagged.",
        "",
        "## Evidence tables (deviation from the working notes, approved 2026-11-10 after pilot 1)",
        f"- Hypotheses: q in {list(d.get('q_values', []))} (H1: rate q under the policy and 1-q without it; H0 the"
        f" reverse). Group sizes {list(d.get('n_range', []))} per arm; success rates within"
        f" {list(d.get('rate_range', []))}. The notes specified q 0.52-0.58, 50-400 units per arm and rates 30-70%.",
        "- Every table's evidence direction (sign of x) matches its visible rate comparison; null tables (x = 0) show"
        " two identical groups. x is computed exactly from Eq. 1 and spans about +/-2.9; half of the non-null tables"
        " are heuristic-conflicting (raw counts mislead), balanced within each q.",
        "- Reasons, from pilot 1: (a) with rates far from q vs 1-q the exact log-LR pointed against the visible rate"
        " comparison in 46% of tables; (b) with large groups and q near 0.5 even |x| = 3 showed as rate gaps of 1-3"
        " points and GPT-4o's evidence slope was -0.34. With the settings above the median visible gap is about 10"
        " points; in a 20-trial check GPT-4o (slope 0.31) and Claude Sonnet 5 (slope 0.67) used the evidence. Claude"
        " Haiku 5.5 was checked the same way and rejected: it answered 50 in 83% of samples (slope 0.00).",
        "",
        "## Answer formats and parsing (deviation, approved 2026-11-10)",
        "- Open weights: probability-weighted readout of the next token (Section 4.5); integer-token coverage < 0.9"
        " -> re-prompted once with the answer prefix 'Probability (0-100): '.",
        "- GPT-4o: structured output that must be {\"probability\": integer 0-100}; the readout is the"
        " probability-weighted mean over the integers among the top-20 log-probabilities at the number token,"
        " renormalized (their captured share is recorded). In free text GPT began reasoning on 43% of answers.",
        f"- Claude (Sonnet 5, extended thinking off; pilot only): a required function call with one integer 0-100, K ="
        f" {_k(cfg, 'claude')} samples at temperature 1, averaged after mapping k -> (k+0.5)/101. In free text Claude"
        " began reasoning on about two thirds of answers; Sonnet 5 also rejects an answer prefill.",
        "- Stance questions (g) for GPT and Claude stay in free text (first-token readout for GPT; sampled replies for"
        " Claude that count only if they start with an integer 0-100). Stance and trials therefore use different"
        " formats for the API models; reported as a limitation.",
        "- Free-text API fallback (not used for trials): one retry appending 'Reply with the number only. Do not"
        " explain.' (chosen on 12 failing pilot prompts: 9/12 usable vs 0/12 for an answer prefix).",
        "",
        "## Exclusions",
        "- Open weights: integer-token coverage < 0.9 after the answer-prefix re-prompt (chat format); < 0.5 for the"
        " plain-text completion format (deviation approved 2026-11-10 after the pilot showed completion-format coverage"
        " near 0.85-0.9 for OLMo; robustness check at 0.8).",
        "- GPT-4o and Claude trials: excluded only if the constrained reply is not a valid integer (GPT) or fewer than"
        " half of the K samples are valid (Claude). The 0.9 coverage rule does not apply to GPT's structured readout:"
        " its top-20 list captures only part of a spread distribution (54-61% in the pilot check).",
        "- Failed vignettes are replaced by a passing vignette of the same group.",
        "",
        "## RQ2",
        "- Targets: unpooled per-cell slopes, cells = identity group x cue type x source x belief x evidence direction"
        " (deviation, approved 2026-11-10: the notes' cells also split by topic and H1 wording, which leaves a median"
        " of 1 trial per cell at the main-run size). Splits: held-out topics in 4 folds (the same cells estimated on"
        " training topics and on unseen topics; median 28 and 9 trials per cell) and held-out dimension (7 folds;"
        " median 37). g and v_k enter as within-cell averages.",
        "- rank(W) = 2; ridge by inner CV on training data. Baselines: (b, d) training means; model without W.",
        "- Success: full model beats both baselines (RMSE and Spearman) on a split. 'No' otherwise.",
        "",
        "## RQ2b (exploratory)",
        "- Gate: RQ2 succeeds on held-out topics. Four contrasts (contra shift, supported shift, affirm - deny,"
        " lab - tank), precision-weighted. Residual weighted SVD, rank 1-5 by CV, varimax; 1000 bootstraps with"
        " orthogonal Procrustes; interpret a residual pattern only if median alignment >= 0.8 and it replicates in"
        " both final open-weight models.",
        "",
        "## RQ3",
        "- Caved: omega < 0.5; resisted: omega >= 0.8 (contradicted trials, |x| > 0.5), matched on item, |x| bin,"
        " topic, cue, cue type. Topic projection k in {4, 8, 16}. Layer and psi chosen on validation topics;"
        " evaluated once on held-out topics.",
        "- Success: contradicted slope moves toward the unsteered neutral slope AND neutral and Tier 1 slopes change"
        " by < 0.05, in every dimension, and the spread of delta_j within dimensions shrinks. Controls: random"
        " direction of equal norm; topic-only component. Prompt-contrast direction as a label-free check.",
        "- 'No' for RQ3: the cleaned direction does no better than a random one, or fixes the slope only by"
        " flattening it everywhere.",
        "- RQ3b (exploratory): signature-specific directions built on training topics, evaluated on held-out topics.",
        "",
        "## Sample size",
        f"- About {d.get('n_belief', 600) // 2 - 9} contradicted, {d.get('n_belief', 600) // 2 - 9} supported and"
        f" {d.get('n_neutral', 150)} belief-neutral trials per group and cue type in Tier 2; Tier 1 at"
        f" {d.get('tier1_scale', 0.4)}x; identical for every main-run model.",
        "- Set by simulation from the pilot (each model's residual noise, slope, rho and spread of m; Delta_b at"
        " the pilot estimate; theta_cong = -0.05, -0.10, -0.15): at 150 contradicted trials per group and cue type,"
        " GPT-4o has power of about 1.00 for both co-primary tests and delta_j 95% half-widths of about 0.11."
        " The simulation omits item effects, so these powers are optimistic.",
        "- Claude Sonnet 5 is not in the main run (budget). Its pilot (1,932 trials) is reported as exploratory"
        " and never pooled with main-run data; at 150 per cell its simulated power was 0.83-0.93, and delta_j"
        " half-widths stayed near 0.26 even at the notes' full size.",
        "",
    ]
    text = "\n".join(lines)
    (rd / "PREREGISTRATION.md").write_text(text)
    return text
