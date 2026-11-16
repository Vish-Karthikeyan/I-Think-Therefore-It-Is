belief_subspaces_v2.npz (corrected RQ3, answer position; one file per model: olmo_instruct, llama, tulu_final)

Keys are "L<layer>/<name>", layers 4, 8, 12, 16, 20, 24:
  ans_U2, ans_U4    orthonormal basis (2 or 4 rows, each of the hidden size) of the belief subspace at the answer position,
                    from counterfactual inputs that differ only in the stated belief
  ans_t2, ans_t4    the no-belief coordinates used as the erasure target: h <- h - U^T (U h - t)
  span_U1, span_U2  the same at the belief-sentence tokens; span_t1, span_t2 their targets
  rand_U2, rand_U4  random subspaces of the same size (controls); rand_t2, rand_t4 their targets
  cr_hat_k8         counterfactual belief direction with the top-8 topic components removed; cr_scale its norm scale

Applying the erasure in reverse (moving the coordinates away from the no-belief target) strengthens
belief-following. See the secondary paper, Section "RQ3: masking and steering", and src/idea/mech/rq3v2.py.
