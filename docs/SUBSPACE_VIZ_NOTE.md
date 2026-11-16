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
(4-D) that encodes the user's **stated belief**. `scripts/capture_subspace_viz.py` projects all 1500
counterfactual trials onto it. A useful structural fact fell out of the projections: the top-2 subspace
`ans_U2` carries mostly the common-mode *"a belief is present" magnitude* (believe-H1 and believe-H0 barely
separate on it, Cohen's *d* ≈ 0.1–0.6), whereas the *polarity* — **which** belief — lives in the added
`ans_U4` dimension (the 3rd basis vector alone gives *d* ≈ −2.2 / +1.7 / +1.6 for OLMo / Llama / Tülu). So the
figures use the belief axis derived within `ans_U4`. `analysis/14_subspace_viz.py` draws:

- **F12 — belief-subspace scatter, colored by stated belief (KEY FIGURE).** This is move (1): each trial
  plotted as (belief axis, leading orthogonal direction), where the belief axis is the difference-of-class-means
  direction within `ans_U4` and the y-axis is the leading residual (orthogonal) direction. One panel per model,
  with class centroids. believe-H1 and believe-H0 form cleanly separated clouds along the belief axis — the
  direct "you can see the belief direction" plot (Cohen's *d* = 2.09 / 1.01 / 1.45 in the titles).
- **F13 — same scatter, colored by identity dimension (marker = belief).** The belief split is preserved
  within every identity dimension (political, religion, gender, …): points of all dimensions intermix along the
  orthogonal direction, yet the believe-H1 / believe-H0 split along the belief axis holds everywhere. The
  belief axis is identity-/topic-agnostic, not an artifact of one persona or topic.
- **F14 — layerwise geometry trajectory.** Move (4): the cosine between the believe-H1 and believe-H0 mean
  directions vs layer (from `geometry.csv`). It is ~+1 in early layers (a shared "a belief is present" axis)
  and goes negative by the erase layer (the two beliefs point apart) — the presence→polarity transition, with
  the erase layer marked. This is the layerwise counterpart of the F12 result: `ans_U2` ≈ the early "presence"
  geometry, the extra `ans_U4` dimension ≈ the late "polarity" geometry.
- **F15 — belief-discriminant density.** Move (3): the 1-D histogram of each trial's score on the belief axis,
  per model; the two belief distributions separate — the classifier/probe view of the same subspace.

F12 best mirrors the canonical concept-direction scatter; F13 and F15 are its identity-agnosticism and
separability complements; F14 is the layer-trajectory view of how the subspace splits presence from polarity.
