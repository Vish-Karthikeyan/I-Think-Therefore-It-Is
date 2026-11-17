"""Model registry: the study's models with pinned Hugging Face revisions and OpenRouter ids
(Section 4.3). The HF backend also records the resolved commit in every result."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str | None = None          # Hugging Face repo for open weights
    revision: str = "main"
    openrouter_id: str | None = None  # API model id
    backend: str = "hf"               # "hf" | "openrouter" | "mock"
    formats: tuple[str, ...] = ("chat",)
    readout: str = "dist"             # "dist" (full next-token / logprobs) | "sample"
    stage: str | None = None          # OLMo stage label
    open_weights: bool = True
    extra_body: dict = field(default_factory=dict)
    # API re-prompt when an answer is unusable: "suffix" appends "Reply with the number only..."
    # to the question; "prefill" starts the model's reply with the answer prefix (Anthropic models
    # continue a prefilled assistant turn; OpenAI models do not).
    reprompt: str = "suffix"
    # How API models give number answers: "text" (free text, first token read), "json"
    # (OpenAI structured output {"probability": int}; logprobs read at the number token), or
    # "tool" (a required function call with one integer argument; sampled K times).
    answer_format: str = "text"
    # Exploratory additions (2026-11-11). Defaults leave every prompt unchanged.
    system_prompt: str | None = None  # prepended as a system message to chat trial prompts
    prompt_variant: int = 0           # reworded trial templates (prompts.VARIANTS); 0 = original
    belief_strength: int = 0          # hedge on the belief sentence (prompts.BELIEF_HEDGES); 0 = original


MODELS: dict[str, ModelSpec] = {
    "olmo_base": ModelSpec("olmo_base", "allenai/OLMo-2-1124-7B", revision="7df9a82518afdecae4e8c026b27adccc8c1f0032", formats=("completion",), stage="base"),
    "olmo_sft": ModelSpec("olmo_sft", "allenai/OLMo-2-1124-7B-SFT", revision="1de02c0175118a9de5854aec80a1f970e701e928", formats=("chat", "completion"), stage="sft"),
    "olmo_dpo": ModelSpec("olmo_dpo", "allenai/OLMo-2-1124-7B-DPO", revision="e34ea60adff2e575f4fe7569eaffd1b28509b6fd", formats=("chat", "completion"), stage="dpo"),
    "olmo_instruct": ModelSpec("olmo_instruct", "allenai/OLMo-2-1124-7B-Instruct", revision="470b1fba1ae01581f270116362ee4aa1b97f4c84",
                               formats=("chat", "completion"), stage="instruct"),
    "llama": ModelSpec("llama", "meta-llama/Llama-3.1-8B-Instruct", revision="0e9e39f249a16976918f6564b8830bc894c89659", formats=("chat",),
                       openrouter_id="meta-llama/llama-3.1-8b-instruct"),
    # Non-reasoning GPT chat model that returns top-20 logprobs through OpenRouter.
    "gpt": ModelSpec("gpt", openrouter_id="openai/gpt-4o-2024-11-20", backend="openrouter",
                     readout="dist", open_weights=False,
                     extra_body={"provider": {"order": ["OpenAI"], "allow_fallbacks": False,
                                              "require_parameters": True}},
                     answer_format="json"),
    # Claude with extended thinking disabled; no logprobs, so the K=10 sampling readout.
    # (Sonnet 5.5 and Opus 5.5 make reasoning mandatory on OpenRouter as of 2026-11-09, so the
    # current Sonnet that allows thinking off is Sonnet 5.)
    # Claude Sonnet 5 (2026-11-10). Haiku 5.5 was tested and rejected: it answered "50" in 83% of
    # samples regardless of the evidence (slope 0.00 vs Sonnet 5's 0.67 on the same 20 tables).
    "claude": ModelSpec("claude", openrouter_id="anthropic/claude-sonnet-5", backend="openrouter",
                        readout="sample", open_weights=False,
                        extra_body={"reasoning": {"enabled": False},
                                    "provider": {"order": ["Anthropic"], "allow_fallbacks": False}},
                        answer_format="tool"),
    # Gemini 2.5 Flash (added 2026-11-15, exploratory; via OpenRouter, which exposes no Gemini logprobs, so
    # the sampling readout with a structured {"probability": k} answer; thinking switched off).
    "gemini": ModelSpec("gemini", openrouter_id="google/gemini-2.5-flash", backend="openrouter",
                        readout="sample", open_weights=False,
                        extra_body={"reasoning": {"enabled": False}}, answer_format="json"),
    # (Sonnet 5 rejects assistant prefill: "This model does not support assistant message
    # prefill", tested 2026-11-10, so Claude uses the suffix re-prompt.)
    # Small real model for testing the open-weight code paths (capture, probes, steering, audit) on a
    # laptop; same OLMo 2 tokenizer as the 7B checkpoints. Not part of the study.
    "olmo1b_test": ModelSpec("olmo1b_test", "allenai/OLMo-2-0425-1B-Instruct", formats=("chat",)),
    # Simulated model with planted effects, for testing the pipeline end to end.
    "mock": ModelSpec("mock", backend="mock", formats=("chat", "completion")),
    "mock_sample": ModelSpec("mock_sample", backend="mock", readout="sample", open_weights=False),
}

# ---------------------------------------------------------------------------
# Exploratory additions (2026-11-11); not preregistered.
# ---------------------------------------------------------------------------
DEBIAS_SYSTEM_PROMPT = ("Judge the hypotheses only from the study design and results. The person's own "
                        "views are not evidence and should not affect your answer.")


def _alias(base: str, suffix: str, **kw) -> ModelSpec:
    from dataclasses import replace
    return replace(MODELS[base], key=f"{base}_{suffix}", formats=("chat",), **kw)


MODELS.update({
    # Scale: OLMo 2 13B stages (same recipe as 7B) and Phi-4 14B (single-token numbers 0-100;
    # Qwen 2.5 was dropped because it splits numbers into digits).
    "olmo13_base": ModelSpec("olmo13_base", "allenai/OLMo-2-1124-13B", revision="3fefddc1bf18a30e1d9b91000271630718f2aa8b", formats=("completion",), stage="base"),
    "olmo13_sft": ModelSpec("olmo13_sft", "allenai/OLMo-2-1124-13B-SFT", revision="b7ec47ed94f3f1d244d9d3cc53eb5d269f8eceec", formats=("chat", "completion"), stage="sft"),
    "olmo13_dpo": ModelSpec("olmo13_dpo", "allenai/OLMo-2-1124-13B-DPO", revision="97a7f33fc8b419808db4ff92795244021e5cf5c2", formats=("chat", "completion"), stage="dpo"),
    "olmo13_instruct": ModelSpec("olmo13_instruct", "allenai/OLMo-2-1124-13B-Instruct", revision="3a5c85baefbb1896a54d56fe2e76c0395627ddf4", formats=("chat", "completion"), stage="instruct"),
    "phi4": ModelSpec("phi4", "microsoft/phi-4", revision="2db69c1c3e91a05d2c64a3185acfbaf36f744e25", formats=("chat",)),
    # Tulu 3: the OLMo 2 post-training recipe applied to Llama-3.1-8B (base -> SFT -> DPO -> RLVR).
    "tulu_base": ModelSpec("tulu_base", "meta-llama/Llama-3.1-8B", revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b", formats=("completion",), stage="base"),
    "tulu_sft": ModelSpec("tulu_sft", "allenai/Llama-3.1-Tulu-3-8B-SFT", revision="f2a0b46b0cfda21003c6141b1ff837b7e165524d", formats=("chat", "completion"), stage="sft"),
    "tulu_dpo": ModelSpec("tulu_dpo", "allenai/Llama-3.1-Tulu-3-8B-DPO", revision="a7beb67e33ffd01cc87ac3b46cadc1000985b8db", formats=("chat", "completion"), stage="dpo"),
    "tulu_final": ModelSpec("tulu_final", "allenai/Llama-3.1-Tulu-3-8B", revision="666943798adbde0b1aff34626007e26986a3c107", formats=("chat", "completion"), stage="instruct"),
})
MODELS.update({
    # Prompting baseline for RQ3: same weights, a debiasing system prompt.
    "olmo_instruct_sp": _alias("olmo_instruct", "sp", system_prompt=DEBIAS_SYSTEM_PROMPT),
    "llama_sp": _alias("llama", "sp", system_prompt=DEBIAS_SYSTEM_PROMPT),
    # Prompt-wording robustness: reworded templates on a fixed third of trials.
    **{f"{b}_pv{v}": _alias(b, f"pv{v}", prompt_variant=v)
       for b in ("olmo_instruct", "llama", "olmo13_instruct", "phi4") for v in (1, 2)},
})
OLMO13_STAGES = ["olmo13_base", "olmo13_sft", "olmo13_dpo", "olmo13_instruct"]
TULU_STAGES = ["tulu_base", "tulu_sft", "tulu_dpo", "tulu_final"]

OLMO_STAGES = ["olmo_base", "olmo_sft", "olmo_dpo", "olmo_instruct"]
RQ3_MODELS = ["olmo_instruct", "llama"]


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {sorted(MODELS)}") from None
