# Visualizing the belief subspace

## How prior interpretability work visualizes concept subspaces

The linear-representation line of work treats a human-interpretable concept as a direction (or a
low-dimensional subspace) in a model's hidden-state space, and visualizes it by **projecting activations onto
that direction and plotting the result**. The recurring moves are: (1) a *2-D projection onto the concept's own
directions* with points colored by the concept's value, so a binary concept shows up as two separated clouds
along one axis (the canonical "steering vector / concept direction" scatter, and the activation-patching and
difference-in-means pictures that accompany it); (2) *PCA / low-rank scatters* of the activations, where the
leading components recover the concept axis when the concept dominates the local geometry; (3) a *1-D
histogram / density* of the projection onto a learned discriminant (the probing / logistic-readout view, which
asks how linearly separable the concept is); and (4) *layerwise or logit-lens trajectories* that track how a
direction's strength, sign, or geometry changes with depth, revealing where and how the representation forms.
These are the standard ways (difference-in-means directions, linear probes, PCA, logit lens / tuned lens) that
concept subspaces are made legible.

## Our figures and what each shows about the belief subspace

The corrected RQ3 gives us, at each model's erase layer, an answer-position subspace `ans_U2` (2-D) / `ans_U4`
(4-D) that encodes the user's stated belief. `scripts/capture_subspace_viz.py` projects all 1500 counterfactual
trials onto it.

**Two labels for the same belief, which must not be confused.** `b` is the belief relative to the hypothesis the
model scores (+1 believes H1, −1 believes H0; the model reports P(H1)). `belief_dir = b · h_dir` is the concrete
claim asserted (policy helps / hurts). H1 is worded in both directions, so the two are crossed across items (within
one item they coincide).

**What the projections show** (`analysis/output/rq3_v2/subspace_separation.csv`):

| model | d(`b`) in `ans_U2` | cross-fit (leave topic out) | d(claim) in `ans_U2` | d(claim) in first 3 dims of `ans_U4` |
|---|---|---|---|---|
| OLMo 2 7B Instruct (L24) | 5.27 | 5.14 | 0.13 | 2.09 |
| Llama 3.1 8B Instruct (L16) | 9.36 | 9.31 | 0.22 | 1.01 |
| Tülu 3 final (L24) | 8.62 | 8.52 | 0.60 | 1.45 |

The 2-D core, which the interchange test shows is sufficient (74–90% recovery), encodes the belief *relative to the
scored hypothesis*, nearly perfectly, and the concrete claim hardly at all. The claim content appears more weakly in
the extra `ans_U4` dimensions. The `b` split holds within every identity dimension and with no identity cue
(smallest within-dimension d: 3.9 / 8.5 / 6.7).

*Correction to an earlier version of this note:* it colored trials by `belief_dir`, saw little separation in
`ans_U2`, and concluded that `ans_U2` carries only a "belief is present" magnitude with polarity in `ans_U4`. That was
wrong: the polarity is in `ans_U2`, coded relative to the hypothesis; what `ans_U2` lacks is the concrete claim.

`analysis/14_subspace_viz.py` draws:

- **F12 — `ans_U2` plane (KEY FIGURE).** Top row colored by `b`: two disjoint clouds. Bottom row, the same points
  colored by the concrete claim: mixed. Move (1), the canonical concept-direction scatter.
- **F13 — same plane colored by identity dimension (marker = `b`):** the split holds in every dimension.
- **F14 — layerwise trajectory** (from `geometry.csv`): cosine between the believe-H1 and believe-H0 mean directions,
  near +1 early (a belief was stated) and negative by the erase layer (which belief). Move (4).
- **F15 — density** of each trial's score on the `b` axis within `ans_U2`. Move (3).

## Dose–response (analysis/13)

Natural trial-to-trial magnitude on the `b` axis does **not** predict how much contradicting evidence is
discounted (pooled r = 0.03 / 0.01 / 0.05; within-belief r small and sign-inconsistent). Signing matters: the
response must be signed by `b` (H1 space). An earlier version signed by `belief_dir`, used the concrete-claim axis
from `scripts/capture_doses.py` (which correlates 0.54–0.64 with the model's own answer), and got a spurious r up to
0.62. `doses.parquet` is kept for transparency but is not the primary dose.
