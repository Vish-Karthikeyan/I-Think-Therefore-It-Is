# Steering vectors and belief subspaces (RQ3)

Activation-space directions and subspaces estimated from the open-weight models' hidden states in
the IDEA trials. They let you remove (or, in reverse, strengthen) the effect of a user's stated
belief on how the model weighs evidence.

> **Dual use.** The same vectors applied with the opposite sign *strengthen* belief-following: they
> make a model defer more to what the user already believes and weigh contrary evidence less. They
> are released openly so the result can be checked and built on; do not use them to make deployed
> systems more sycophantic. See the paper's ethics statement.

## Files

| Folder | Model (Hugging Face id @ pinned commit) | Files |
|---|---|---|
| `olmo_instruct/` | `allenai/OLMo-2-1124-7B-Instruct` @ `470b1fba1ae01581f270116362ee4aa1b97f4c84` | `belief_subspaces_v2.npz`, `directions.npz`, `directions.json`, `summary.json` |
| `llama/` | `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659` | `belief_subspaces_v2.npz`, `directions.npz`, `directions.json`, `summary.json` |
| `tulu_final/` | `allenai/Llama-3.1-Tulu-3-8B` @ `666943798adbde0b1aff34626007e26986a3c107` | `belief_subspaces_v2.npz` |

All three models have hidden size 4096. Every array is float32 (except `random` in
`directions.npz`, float64). Keys are `L<layer>/<name>`.

**Layers.** `L<l>` is the output of decoder block `l`, 0-indexed: the module
`model.model.layers[l]` in Hugging Face Transformers, i.e. `output_hidden_states[l + 1]`. All
interventions below are forward hooks on that module's output (the residual stream after block `l`).

### `belief_subspaces_v2.npz` (corrected RQ3; the analysis reported in the paper)

Layers 4, 8, 12, 16, 20, 24. Estimated from 1,500 counterfactual triples on training topics: the
same trial rendered three ways, with the user believing H1, believing H0, or stating no belief.

**Answer position** = the last token of the chat prompt rendered with the chat template, the
generation prompt and the answer prefix `Probability (0-100): ` (for Llama-based models the prefix
without its trailing space, because their number tokens carry the space). This is where the model
reads out its probability, so it is where the answer-position subspaces act.

| Key | Shape | What it is |
|---|---|---|
| `ans_U2` | (2, 4096) | Orthonormal basis of the plane spanned by the two mean belief displacements at the answer position, (believes H1 − no belief) and (believes H0 − no belief). |
| `ans_U4` | (4, 4096) | Top-4 right singular vectors of all counterfactual displacements (belief − no belief) at the answer position. Orthonormal rows. |
| `ans_t2`, `ans_t4` | (2,), (4,) | Erasure targets: the mean no-belief coordinates, `U @ mean(h_no_belief)`. |
| `span_U1` | (1, 4096) | Unit mean difference (believes H1 − believes H0) of the activations averaged over the tokens of the belief sentence. |
| `span_U2` | (2, 4096) | Top-2 right singular vectors of those belief-sentence differences. |
| `span_t1`, `span_t2` | (1,), (2,) | Targets at the belief tokens: the coordinates of the midpoint between the two beliefs (there is no "no belief" sentence to target). |
| `rand_U2`, `rand_U4` | (2, 4096), (4, 4096) | Random orthonormal subspaces of the same sizes (controls); `rand_t2`, `rand_t4` their no-belief targets. |
| `cr_hat_k8` | (4096,) | Caved-minus-resisted direction at the answer position: mean activation of contradicted trials where the model ignored the evidence minus those where it updated, matched within strata (item, evidence bin, topic, cue), with the top-8 topic principal components of the no-belief activations projected out. OLMo and Llama only (not estimated for Tülu). |
| `cr_scale` | () | Standard deviation of the training activations projected on `cr_hat_k8`/‖`cr_hat_k8`‖: the unit for steering strength. |

### `directions.npz`, `directions.json`, `summary.json` (original, preregistered RQ3)

Layers 8, 10, …, 26. Estimated at the last token of the plain chat prompt (without the answer
prefix). The corrected analysis above superseded these because most answers are produced after
the answer prefix; they are kept because the preregistered test used them.

| Key in `directions.npz` | Shape | What it is |
|---|---|---|
| `raw` | (4096,) | Pooled caved-minus-resisted difference of means (matched strata). |
| `per_dim_raw/<dimension>` | (4096,) | The same within one identity dimension (only dimensions with matched strata). |
| `prompt` | (4096,) | Stated-belief minus no-belief activations on contradicted trials, matched on (item, cue). |
| `hat_k{4,8,16}` | (4096,) | `raw` with the top-k topic principal components projected out (`hat_k8` is the main direction). |
| `topic_only_k{k}` | (4096,) | The part of `raw` inside the top-k topic subspace (control). |
| `per_dim_hat_k{k}/<dimension>`, `prompt_hat_k{k}` | (4096,) | Per-dimension and prompt-contrast directions, topic-projected. |
| `random` | (4096,) | Random direction with the norm of `raw` (control). |

`directions.json`: per layer, `n_strata` and `n_strata_dim` (matched strata behind each
direction), `topic_frac_k{k}` (share of `raw`'s squared norm inside the topic subspace) and `scale`
(per direction, the standard deviation of the activations projected on the unit direction).
`summary.json`: the setting chosen on validation topics (`layer`, `multiple`, `mode`: steer the
last position or all positions), the probe layer, its held-out R², and the masking index.

## Loading and applying

```python
import numpy as np
S = np.load("steering_vectors/llama/belief_subspaces_v2.npz")
U, t = S["L16/ans_U4"], S["L16/ans_t4"]          # (4, 4096), (4,)
```

**Erasure (target setting).** Set the coordinates of the hidden state `h` inside the subspace to
the no-belief target, leaving the orthogonal complement unchanged:

    h' = h − Uᵀ(U h − t)

Applied at the answer position (last token) for `ans_*`, or at every token of the belief sentence
for `span_*`.

**Steering along a direction.** With `u = d/‖d‖` and strength `ψ = multiple × scale`
(`cr_scale`, or `directions.json` → `L<l>/scale/<name>`):

    h' = h − ψ u

at the last position or at every position. Subtracting the caved-minus-resisted direction reduces
belief-following; adding it (ψ < 0), or moving the subspace coordinates away from `t`, increases it.

A self-contained forward hook (Hugging Face Transformers, no other dependency):

```python
import torch

def erase_hook(U, t, last_only=True):
    U = torch.as_tensor(U, dtype=torch.float32)
    t = torch.as_tensor(t, dtype=torch.float32)
    def hook(module, inputs, out):
        h = out[0] if isinstance(out, tuple) else out
        Ud, td = U.to(h.device), t.to(h.device)
        hf = h.float()
        sel = hf[:, -1:, :] if last_only else hf
        new = sel - (sel @ Ud.T - td) @ Ud                 # h - U^T (U h - t)
        if last_only:
            hf = torch.cat([hf[:, :-1, :], new], dim=1)
        else:
            hf = new
        h = hf.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
#                                              revision="0e9e39f249a16976918f6564b8830bc894c89659", ...)
handle = model.model.layers[16].register_forward_hook(erase_hook(U, t))
# ... run prompts rendered with the answer prefix; read the next-token distribution over "0".."100"
handle.remove()
```

With the `idea` package (`pip install -e ".[hf]"`), the HF backend applies the same operations and
reads out the answers exactly as the study did:

```python
from idea.elicit import make_backend
from idea.models import MODELS
b = make_backend(MODELS["llama"])                                   # pinned revision
b.set_steering([dict(kind="erase", layer=16, U=U, target=t, pos="last")])     # erasure
# or a direction: b.set_steering([(16, S["L16/cr_hat_k8"], 2.0 * float(S["L16/cr_scale"]), "all")])
```

`analysis/rq3.py` re-runs the paper's RQ3 analyses with these files (GPU required).

## License and attribution

The steering vectors and belief subspaces are released under the
[PolyForm Noncommercial License 1.0.0](../LICENSE.md), subject also to the license of the model
each was derived from:

- `olmo_instruct/`: derived from OLMo 2 7B Instruct (Ai2), Apache License 2.0.
- `llama/` and `tulu_final/`: derived from Llama 3.1 8B Instruct and from Tülu 3 8B (Ai2, built on
  Llama 3.1 8B). **Built with Llama.** Llama 3.1 is licensed under the Llama 3.1 Community License,
  Copyright © Meta Platforms, Inc. All Rights Reserved. Use is subject to that license
  ([`third_party/LLAMA_3.1_LICENSE`](../third_party/LLAMA_3.1_LICENSE)) and Meta's Acceptable Use
  Policy; Tülu 3 is additionally subject to Ai2's terms for Tülu 3.
