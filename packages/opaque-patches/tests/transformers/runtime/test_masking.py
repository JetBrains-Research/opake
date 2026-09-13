import inspect

import pytest

pytest.importorskip("transformers")
import torch

from opaque.api.patches.transformers.runtime.masking import (
    vmap_create_causal_mask,
    vmap_create_compact_sdpa_sliding_window_causal_mask,
    vmap_create_recurrent_attention_mask,
    vmap_create_sliding_window_causal_mask,
    vmap_update_linear_attention_mask,
)
from opaque.patches import apply_runtime_patches


def test_vmap_causal_mask():
    apply_runtime_patches(vmap_masking=True)

    # create_causal_mask should be patched now
    import transformers.masking_utils as masking_utils

    create_causal_mask = masking_utils.create_causal_mask

    input_embeds = torch.randn(1, 4, 16)

    class DummyConfig:
        _attn_implementation = "eager"

    config = DummyConfig()

    # Test that it handles normal execution
    mask = create_causal_mask(
        config=config,
        input_embeds=input_embeds,
        attention_mask=None,
        cache_position=torch.arange(4),
        past_key_values=None,
    )
    assert mask.shape == (1, 1, 4, 4)

    # Test under vmap with a fake batched tensor (we can simulate by just passing higher dimension)
    input_embeds_vmap = torch.randn(1, 4, 16)
    attention_mask = torch.ones(4)  # 1D mask to simulate vmap masking

    mask_vmap = create_causal_mask(
        config=config,
        input_embeds=input_embeds_vmap,
        attention_mask=attention_mask,
        cache_position=torch.arange(4),
        past_key_values=None,
    )
    assert mask_vmap.shape == (1, 1, 4, 4)


def test_vmap_causal_mask_follows_autocast(monkeypatch):
    """Mask dtype mirrors autocast so SDPA's attn_mask matches a bf16 query."""
    from opaque.api.patches.transformers.runtime import masking

    apply_runtime_patches(vmap_masking=True)
    import transformers.masking_utils as masking_utils

    create_causal_mask = masking_utils.create_causal_mask
    input_embeds = torch.randn(1, 4, 16, dtype=torch.float32)

    class DummyConfig:
        _attn_implementation = "eager"

    mask = create_causal_mask(
        config=DummyConfig(),
        input_embeds=input_embeds,
        attention_mask=None,
        cache_position=torch.arange(4),
        past_key_values=None,
    )
    assert mask.dtype == torch.float32

    # Simulate CUDA autocast through the helper used by vmap_create_causal_mask;
    # ``torch.is_autocast_enabled("cuda")`` only flips inside a real CUDA ctx.
    monkeypatch.setattr(
        masking,
        "_active_mask_dtype",
        lambda _ie: torch.bfloat16,
    )
    mask_bf16 = create_causal_mask(
        config=DummyConfig(),
        input_embeds=input_embeds,
        attention_mask=None,
        cache_position=torch.arange(4),
        past_key_values=None,
    )
    assert mask_bf16.dtype == torch.bfloat16


def test_masking_runtime_patch_idempotent_for_ignore_causal_mask_sdpa():
    apply_runtime_patches(vmap_masking=True)

    import transformers.masking_utils as masking_utils

    patched_fn = masking_utils._ignore_causal_mask_sdpa
    original_fn = getattr(patched_fn, "_original", None)
    assert original_fn is not None
    assert original_fn is not patched_fn

    # Re-applying runtime patches must preserve the same original binding.
    apply_runtime_patches(vmap_masking=True)
    patched_fn_2 = masking_utils._ignore_causal_mask_sdpa
    original_fn_2 = getattr(patched_fn_2, "_original", None)
    assert patched_fn_2 is patched_fn
    assert original_fn_2 is original_fn


def test_vmap_causal_mask_preserves_all_valid_sdpa_fast_path():
    class DummyConfig:
        _attn_implementation = "sdpa"

    input_embeds = torch.randn(1, 8, 16)
    created_masks = []
    parameter = torch.ones(1, requires_grad=True)

    def create_mask(param, attention_mask):
        created_masks.append(
            vmap_create_causal_mask(
                config=DummyConfig(),
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                cache_position=torch.arange(8),
                past_key_values=None,
            )
        )
        return param.square().sum() + attention_mask.sum() * 0

    torch.vmap(torch.func.grad(create_mask), in_dims=(None, 0))(
        parameter, torch.ones(3, 8, dtype=torch.bool)
    )
    assert created_masks == [None]

    created_masks.clear()
    padded = torch.ones(3, 8, dtype=torch.bool)
    padded[1, -1] = False
    torch.vmap(torch.func.grad(create_mask), in_dims=(None, 0))(parameter, padded)
    assert len(created_masks) == 1
    assert created_masks[0] is not None


@pytest.mark.parametrize(
    ("input_shape", "mask", "expected"),
    [
        ((3, 4), [0, 1, 1, 0, 1], [1, 0, 1]),
        ((1, 3, 4), [[0, 1, 1, 0, 1]], [[1, 0, 1]]),
    ],
)
def test_recurrent_attention_mask_trims_batchless_and_batched_padding(
    input_shape, mask, expected
):
    result = vmap_create_recurrent_attention_mask(
        config=object(),
        inputs_embeds=torch.randn(input_shape),
        attention_mask=torch.tensor(mask),
    )
    assert torch.equal(result, torch.tensor(expected))
    assert result.is_contiguous()


def test_recurrent_attention_mask_preserves_per_example_padding_under_vmap():
    inputs_embeds = torch.randn(3, 4)
    masks = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 0, 1]])

    result = torch.vmap(
        lambda mask: vmap_create_recurrent_attention_mask(
            config=object(),
            inputs_embeds=inputs_embeds,
            attention_mask=mask,
        )
    )(masks)

    assert torch.equal(result, masks[:, -inputs_embeds.shape[0] :])


def test_legacy_linear_attention_mask_preserves_prefill_and_skips_cache():
    padded = torch.tensor([[0, 1, 1]])
    prefill = vmap_update_linear_attention_mask(
        object(), padded, cache_position=torch.arange(3)
    )
    cached = vmap_update_linear_attention_mask(
        object(), torch.ones(1, 4), cache_position=torch.tensor([3])
    )

    assert prefill is padded
    assert cached is None


def test_legacy_linear_attention_mask_is_vmap_safe():
    masks = torch.tensor([[1, 1, 1], [0, 1, 1], [1, 0, 1]])
    result = torch.vmap(
        lambda mask: vmap_update_linear_attention_mask(
            object(), mask, cache_position=torch.arange(3)
        )
    )(masks)
    assert torch.equal(result, masks)


class TestSlidingWindowCausalMask:
    """vmap_create_sliding_window_causal_mask enforces the look-back limit."""

    class _EagerConfig:
        _attn_implementation = "eager"
        sliding_window = 2

    class _EagerConfigNoWindow:
        _attn_implementation = "eager"

    def _make_mask(self, seq_len, sliding_window=2, past_seen_tokens=0):
        config = type(
            "Cfg",
            (),
            {"_attn_implementation": "eager", "sliding_window": sliding_window},
        )()
        input_embeds = torch.randn(1, seq_len, 8)
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len)
        return vmap_create_sliding_window_causal_mask(
            config,
            inputs_embeds=input_embeds,
            attention_mask=None,
            past_key_values=None,
            cache_position=cache_position,
        )

    def test_shape(self):
        mask = self._make_mask(seq_len=4, sliding_window=2)
        assert mask.shape == (1, 1, 4, 4)

    def test_within_window_causal_positions_are_zero(self):
        # seq_len=4, sliding_window=2; positions 0..3
        mask = self._make_mask(seq_len=4, sliding_window=2)
        # q=0, k=0: causal (k==q) and in-window (0 >= 0-2+1=-1) → 0.0
        assert mask[0, 0, 0, 0] == 0.0
        # q=1, k=1: in-window → 0.0
        assert mask[0, 0, 1, 1] == 0.0
        # q=2, k=1: causal (k<q), in-window (1 >= 2-2+1=1) → 0.0
        assert mask[0, 0, 2, 1] == 0.0
        # q=3, k=2: causal, in-window (2 >= 3-2+1=2) → 0.0
        assert mask[0, 0, 3, 2] == 0.0

    def test_outside_window_positions_are_neg_inf(self):
        # seq_len=4, sliding_window=2
        mask = self._make_mask(seq_len=4, sliding_window=2)
        neg_inf = torch.finfo(mask.dtype).min
        # q=2, k=0: causal, but out-of-window (0 < 2-2+1=1) → -inf
        assert mask[0, 0, 2, 0] == neg_inf
        # q=3, k=0: out-of-window (0 < 3-2+1=2) → -inf
        assert mask[0, 0, 3, 0] == neg_inf
        # q=3, k=1: out-of-window (1 < 3-2+1=2) → -inf
        assert mask[0, 0, 3, 1] == neg_inf

    def test_anti_causal_positions_remain_neg_inf(self):
        # Future tokens must always be -inf regardless of window.
        mask = self._make_mask(seq_len=4, sliding_window=100)
        neg_inf = torch.finfo(mask.dtype).min
        # q=0, k=1 (future)
        assert mask[0, 0, 0, 1] == neg_inf
        assert mask[0, 0, 0, 3] == neg_inf
        assert mask[0, 0, 1, 2] == neg_inf

    def test_window_larger_than_seq_reduces_to_causal(self):
        # When sliding_window >= seq_len, the result equals plain causal mask.
        seq_len = 4
        causal_cfg = type("Cfg", (), {"_attn_implementation": "eager"})()
        input_embeds = torch.randn(1, seq_len, 8)
        cache_position = torch.arange(seq_len)
        causal_mask = vmap_create_causal_mask(
            causal_cfg, inputs_embeds=input_embeds, cache_position=cache_position
        )
        sliding_mask = self._make_mask(seq_len=seq_len, sliding_window=seq_len + 10)
        assert torch.equal(causal_mask, sliding_mask)

    def test_none_passthrough_when_no_sliding_window_attr(self):
        # Without config.sliding_window the function must return the causal mask
        # unchanged (not None just because the attribute is absent).
        config = type("Cfg", (), {"_attn_implementation": "eager"})()
        input_embeds = torch.randn(1, 4, 8)
        cache_position = torch.arange(4)
        mask = vmap_create_sliding_window_causal_mask(
            config, inputs_embeds=input_embeds, cache_position=cache_position
        )
        # Should be a valid mask (not None) equal to the plain causal mask.
        causal_mask = vmap_create_causal_mask(
            config, inputs_embeds=input_embeds, cache_position=cache_position
        )
        assert mask is not None
        assert torch.equal(mask, causal_mask)

    def test_cached_kv_window_uses_correct_absolute_positions(self):
        # With past_seen_tokens=3, seq_len=2, sliding_window=2:
        # vmap_create_causal_mask lays out the key dim as:
        #   cols 0-1: current tokens at abs positions 3, 4  (cache_position)
        #   cols 2-4: past cached tokens at abs positions 0, 1, 2
        # Query at abs pos 4 (q_idx=1), window=2 → lower bound = 3.
        # → col 0 (abs 3) ✓, col 1 (abs 4) ✓ (future — blocked by causal),
        #   col 2 (abs 0) ✗, col 3 (abs 1) ✗, col 4 (abs 2) ✗.
        seq_len = 2
        past_seen_tokens = 3
        sliding_window = 2
        config = type(
            "Cfg",
            (),
            {"_attn_implementation": "eager", "sliding_window": sliding_window},
        )()
        input_embeds = torch.randn(1, seq_len, 8)
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len)

        # Provide a minimal DynamicCache-like object so past_seen_tokens is read.
        class _FakeCache:
            def get_seq_length(self):
                return past_seen_tokens

        mask = vmap_create_sliding_window_causal_mask(
            config,
            inputs_embeds=input_embeds,
            past_key_values=_FakeCache(),
            cache_position=cache_position,
        )
        neg_inf = torch.finfo(mask.dtype).min
        target_length = past_seen_tokens + seq_len  # 5

        assert mask.shape == (1, 1, seq_len, target_length)

        # q=1 (abs pos 4): can see col 0 (abs pos 3, in window), not cols 2-4 (abs 0-2)
        assert mask[0, 0, 1, 0] == 0.0, "col 0 (abs 3) should be in window"
        assert mask[0, 0, 1, 2] == neg_inf, "col 2 (abs 0) should be out of window"
        assert mask[0, 0, 1, 3] == neg_inf, "col 3 (abs 1) should be out of window"
        assert mask[0, 0, 1, 4] == neg_inf, "col 4 (abs 2) should be out of window"


def test_ignore_causal_mask_shim_accepts_the_upstream_signature():
    """The shim must be callable exactly as transformers calls the real thing.

    ``masking_utils`` calls ``_ignore_causal_mask_sdpa`` positionally, and the
    upstream signature has grown across the supported range — 5.x inserted
    ``q_offset``. A shim that names the parameters raises ``TypeError`` at the
    call site, which surfaces only once a model actually builds a mask. Bind the
    shim against upstream's own signature so the mismatch is caught here.
    """
    masking_utils = pytest.importorskip("transformers.masking_utils")

    from opaque.api.patches.transformers.runtime.masking import (
        _vmap_safe_ignore_causal_mask_sdpa,
    )

    upstream = getattr(
        _vmap_safe_ignore_causal_mask_sdpa,
        "_original",
        masking_utils._ignore_causal_mask_sdpa,
    )
    signature = inspect.signature(upstream)
    bound = signature.bind(None, *range(len(signature.parameters) - 1))
    inspect.signature(_vmap_safe_ignore_causal_mask_sdpa).bind(*bound.args)


def test_ignore_causal_mask_shim_short_circuits_on_a_padding_mask():
    """A present mask forces mask creation without consulting the original."""
    from opaque.api.patches.transformers.runtime.masking import (
        _vmap_safe_ignore_causal_mask_sdpa,
    )

    calls: list[object] = []
    previous = getattr(_vmap_safe_ignore_causal_mask_sdpa, "_original", None)
    _vmap_safe_ignore_causal_mask_sdpa._original = lambda *a, **k: calls.append(a)
    try:
        assert _vmap_safe_ignore_causal_mask_sdpa(object(), 1, 2, 3, 4, None) is False
        assert _vmap_safe_ignore_causal_mask_sdpa(padding_mask=object()) is False
        assert calls == []
    finally:
        if previous is None:
            del _vmap_safe_ignore_causal_mask_sdpa._original
        else:
            _vmap_safe_ignore_causal_mask_sdpa._original = previous


class TestSlidingWindowWithoutPaddingMask:
    """A binding window must be materialized even with no padding mask.

    SDPA's ``is_causal`` shortcut expresses causality only, so returning ``None``
    there silently widens attention from the window to the whole causal prefix.
    """

    @staticmethod
    def _make_mask(attn_impl, sliding_window, seq_len=8, batch_size=1):
        config = type(
            "Cfg",
            (),
            {"_attn_implementation": attn_impl, "sliding_window": sliding_window},
        )()
        return vmap_create_sliding_window_causal_mask(
            config,
            inputs_embeds=torch.randn(batch_size, seq_len, 8),
            attention_mask=None,
            past_key_values=None,
        )

    def test_binding_window_is_materialized_for_sdpa(self):
        mask = self._make_mask("sdpa", sliding_window=2, seq_len=8)
        assert mask is not None, "binding window must not fall back to is_causal"
        assert mask.dtype == torch.bool
        assert mask[0, 0, 3, 2], "in-window key must be visible"
        assert not mask[0, 0, 3, 1], "out-of-window key must be blocked"

    def test_binding_sdpa_mask_broadcasts_without_batch_copies(self):
        seq_len = 8
        mask = self._make_mask("sdpa", sliding_window=2, seq_len=seq_len, batch_size=4)
        assert mask.shape == (4, 1, seq_len, seq_len)
        assert mask.untyped_storage().nbytes() == seq_len * seq_len

    def test_compact_sdpa_path_skips_the_binding_window_mask(self):
        config = type(
            "Cfg",
            (),
            {
                "_attn_implementation": "sdpa",
                "sliding_window": 2,
                "attention_dropout": 0.0,
            },
        )()
        mask = vmap_create_compact_sdpa_sliding_window_causal_mask(
            config,
            inputs_embeds=torch.randn(4, 8, 8),
            attention_mask=None,
            past_key_values=None,
        )
        assert mask is None

    def test_compact_sdpa_path_uses_the_evaluation_fast_path(self):
        config = type(
            "Cfg",
            (),
            {
                "_attn_implementation": "sdpa",
                "sliding_window": 2,
                "attention_dropout": 0.1,
            },
        )()
        inputs_embeds = torch.randn(1, 8, 8)
        with torch.no_grad():
            mask = vmap_create_compact_sdpa_sliding_window_causal_mask(
                config,
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=None,
            )
        assert mask is None

    def test_compact_sdpa_path_retains_dropout_training_fallback(self):
        config = type(
            "Cfg",
            (),
            {
                "_attn_implementation": "sdpa",
                "sliding_window": 2,
                "attention_dropout": 0.1,
            },
        )()
        mask = vmap_create_compact_sdpa_sliding_window_causal_mask(
            config,
            inputs_embeds=torch.randn(1, 8, 8),
            attention_mask=None,
            past_key_values=None,
        )
        assert mask is not None

    def test_compact_sdpa_path_retains_padding_mask_fallback(self):
        config = type(
            "Cfg",
            (),
            {"_attn_implementation": "sdpa", "sliding_window": 2},
        )()
        mask = vmap_create_compact_sdpa_sliding_window_causal_mask(
            config,
            inputs_embeds=torch.randn(1, 8, 8),
            attention_mask=torch.ones(1, 8),
            past_key_values=None,
        )
        assert mask is not None
        assert mask.dtype == torch.float32

    def test_non_binding_window_keeps_the_is_causal_fast_path(self):
        assert self._make_mask("sdpa", sliding_window=64, seq_len=8) is None

    def test_window_equal_to_sequence_keeps_the_is_causal_fast_path(self):
        assert self._make_mask("sdpa", sliding_window=8, seq_len=8) is None

    def test_flash_backend_keeps_the_is_causal_fast_path(self):
        # Flash kernels receive ``sliding_window`` and mask in-kernel.
        assert self._make_mask("flash_attention_2", sliding_window=2, seq_len=8) is None

    def test_allow_is_causal_skip_is_honoured_by_the_causal_builder(self):
        config = type("Cfg", (), {"_attn_implementation": "sdpa"})()
        kwargs = {"inputs_embeds": torch.randn(1, 8, 8), "attention_mask": None}
        assert vmap_create_causal_mask(config, **kwargs) is None
        forced = vmap_create_causal_mask(config, allow_is_causal_skip=False, **kwargs)
        assert forced is not None
        assert forced.shape == (1, 1, 8, 8)
