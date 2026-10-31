"""Engine tests on a tiny random model with PersonaPlex's stream layout (no weights needed).

    .venv/bin/python tests/test_engine.py
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosstalk.engine import SINE_TOKENS, Generator, Voice  # noqa: E402
from crosstalk.model.lm import Lm, LmConfig, personaplex_7b_config  # noqa: E402
from crosstalk.model.transformer import TransformerConfig  # noqa: E402


def tiny_config(context: int = 3000) -> LmConfig:
    full = personaplex_7b_config()
    return LmConfig(
        transformer=TransformerConfig(
            d_model=64, num_heads=4, num_layers=2, dim_feedforward=256,
            context=context, max_period=10000, positional_embedding="rope",
        ),
        depformer=TransformerConfig(
            d_model=32, num_heads=2, num_layers=2, dim_feedforward=128,
            context=8, max_period=10000, positional_embedding="none",
        ),
        dep_q=full.dep_q, n_q=full.n_q, text_card=full.text_card,
        audio_card=full.audio_card, delays=full.delays,
    )


def make(context: int = 3000) -> tuple[Lm, Voice]:
    mx.random.seed(0)
    model = Lm(tiny_config(context))
    mx.eval(model.parameters())
    rng = np.random.default_rng(0)
    voice = Voice(
        "test",
        mx.array(rng.normal(0, 0.05, (12, 1, 1, 64)).astype(np.float32)),
        rng.integers(0, 2048, (1, 17, 4)),
    )
    return model, voice


def next_logits(gen: Generator) -> tuple[np.ndarray, tuple]:
    state = (gen.cache.copy(), gen.provided.copy(), gen.offset)
    p = gen._prepare(user=SINE_TOKENS)
    _, logits = gen.model.forward_codes(mx.array(p.input[None, :, None].astype(np.int32)))
    return np.array(logits)[0, -1], state


def test_prefill_matches_step():
    model, voice = make()
    prompt = list(range(100, 140))
    results = {}
    for mode in ("step", "prefill"):
        gen = Generator(model)
        gen.system_prompts(voice, prompt, mode=mode)
        results[mode] = next_logits(gen)
    (l_step, s_step), (l_pre, s_pre) = results["step"], results["prefill"]
    assert all(np.array_equal(a, b) for a, b in zip(s_step[:2], s_pre[:2])) and s_step[2] == s_pre[2]
    assert np.abs(l_step - l_pre).max() < 1e-3, np.abs(l_step - l_pre).max()
    print("prefill == step: OK (max logit diff %.2e)" % np.abs(l_step - l_pre).max())


def test_prompt_without_voice():
    model, _ = make()
    gen = Generator(model)
    gen.system_prompts(None, [5, 6, 7])
    assert gen.offset == 6 + 3 + 6
    print("prompt without voice: OK")


def test_live_steps_and_ring_wrap():
    # A small context forces the attention ring buffer to wrap around during the call.
    model, voice = make(context=64)
    gen = Generator(model)
    gen.system_prompts(voice, [1, 2, 3])
    outs = [gen.step(SINE_TOKENS) for _ in range(120)]
    assert all(o is not None and o.shape == (9,) for o in outs)
    assert all((o[1:] >= 0).all() and (o[1:] < 2048).all() for o in outs)
    # The first frame's delayed agent codebooks were forced to silence by the prompt.
    assert model.transformer_cache[0].offset == 12 + 6 + 3 + 6 + 120
    print("120 live steps with ring wrap-around: OK")


def test_first_live_frame_forcing():
    model, voice = make()
    gen = Generator(model)
    gen.system_prompts(voice, [1, 2, 3])
    p = gen._prepare(user=SINE_TOKENS)
    # Agent codebooks with delay 1 were written by the last silence frame, so they are
    # forced in the first live frame; the delay-0 one and the text are sampled.
    assert not p.provided[0] and not p.provided[1] and p.provided[2:9].all()
    print("first live frame teacher forcing: OK")


if __name__ == "__main__":
    test_prefill_matches_step()
    test_prompt_without_voice()
    test_live_steps_and_ring_wrap()
    test_first_live_frame_forcing()
