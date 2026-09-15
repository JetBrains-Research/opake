# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap-compatible causal-mask implementations for patched transformers."""

import torch

from opaque.api.patches.transformers.components.attention import vmap_repeat_kv
from opaque.exceptions import InputTypeError

# Process-level packed-sequences policy consulted by ``vmap_create_causal_mask``.
# ``None`` keeps the data-driven probe of the physical batch; see
# :func:`set_packed_sequences`.
_PACKED_SEQUENCES: bool | None = None


def set_packed_sequences(flag: bool | None) -> None:
    """Declare whether every collated row is fully valid (no padding).

    The causal-mask builder may skip materialising the mask and let SDPA run
    its ``is_causal`` fast path when no row is padded. By default it decides
    that by probing the physical (micro)batch, which makes the attention
    kernel a function of the batch composition rather than of the example
    alone. Under DP training the choice should come from a public property of
    the data instead:

    - ``True``: the collator guarantees packed, all-valid rows; the fast path
      is allowed without inspecting the batch.
    - ``False``: never probe the batch; materialise the mask whenever an
      attention mask is given, so every example's attention runs the same
      kernel regardless of its microbatch mates.
    - ``None`` (default): probe the batch as before.

    The policy is a process-wide module setting, not thread-local: a caller
    that sets it for one run (``DPTrainer`` does so for the duration of
    ``train()``) should read :func:`packed_sequences` first and put the
    previous value back afterwards.
    """
    global _PACKED_SEQUENCES
    if flag is not None and not isinstance(flag, bool):
        InputTypeError.raise_("packed_sequences must be True, False or None")
    _PACKED_SEQUENCES = flag


def packed_sequences() -> bool | None:
    """Current packed-sequences policy (see :func:`set_packed_sequences`)."""
    return _PACKED_SEQUENCES


def _active_mask_dtype(input_embeds: torch.Tensor) -> torch.dtype:
    """Dtype the causal mask must use under autocast.

    Follow autocast when active on the input's device, otherwise honour the
    input dtype. Without this the attention block's q_proj casts query to bf16
    while the mask stays fp32, and SDPA raises ``invalid dtype for bias -
    should match query's dtype``. Reading the dtype off the input's device type
    keeps this correct on CUDA, MPS and CPU alike.
    """
    device_type = input_embeds.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return input_embeds.dtype


def _safe_seq_length(past_key_values) -> int:
    """``get_seq_length`` that tolerates hybrid / linear-attention caches.

    Some caches (e.g. qwen3_next's GatedDeltaNet) raise when queried globally;
    for mask building an unknown length is equivalent to no cached tokens.
    """
    if past_key_values is None:
        return 0
    get = getattr(past_key_values, "get_seq_length", None)
    if get is not None:
        try:
            return get()
        except Exception:
            return 0
    return getattr(past_key_values, "seen_tokens", 0)


def _backend_enforces_sliding_window(attn_impl: str | None) -> bool:
    """True when the backend applies the look-back limit in-kernel.

    Flash-attention receives ``sliding_window`` directly; every mask-consuming
    backend (eager, SDPA, flex) needs it materialized in the additive mask.
    """
    return attn_impl is not None and "flash" in attn_impl


def _query_length(input_embeds: torch.Tensor) -> int:
    """Query length of an embedding tensor, batched (3D) or batchless (2D)."""
    return (
        input_embeds.shape[0]
        if input_embeds.ndim == 2  # noqa: PLR2004 - batchless embeddings are 2D
        else input_embeds.shape[1]
    )


def _sdpa_sliding_window_mask(
    input_embeds: torch.Tensor, sliding_window: int
) -> torch.Tensor:
    """Build one broadcastable Boolean SDPA band without dense temporaries."""
    batch_size = (
        1
        if input_embeds.ndim == 2  # noqa: PLR2004 - batchless embeddings are 2D
        else input_embeds.shape[0]
    )
    seq_len = _query_length(input_embeds)
    mask = torch.ones(
        (1, 1, seq_len, seq_len), dtype=torch.bool, device=input_embeds.device
    )
    mask.tril_()
    mask.triu_(diagonal=1 - sliding_window)
    return mask.expand(batch_size, -1, -1, -1)


def _can_use_compact_sdpa_sliding_window(
    config,
    input_embeds: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    past_key_values,
    cache_position: torch.Tensor | None,
    allow_is_causal_skip: bool,
    or_mask_function,
    and_mask_function,
    block_sequence_ids: torch.Tensor | None,
) -> bool:
    """Whether a patched SDPA forward can apply the window in query chunks."""
    sliding_window = getattr(config, "sliding_window", None)
    return (
        allow_is_causal_skip
        and sliding_window is not None
        and input_embeds is not None
        and attention_mask is None
        and getattr(config, "_attn_implementation", None) == "sdpa"
        and _safe_seq_length(past_key_values) == 0
        and cache_position is None
        and or_mask_function is None
        and and_mask_function is None
        and block_sequence_ids is None
        and (
            not torch.is_grad_enabled()
            or getattr(config, "attention_dropout", None) == 0.0
        )
        and _query_length(input_embeds) > sliding_window
    )


def vmap_create_compact_sdpa_sliding_window_causal_mask(
    config,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    position_ids: torch.Tensor | None = None,
    or_mask_function=None,
    and_mask_function=None,
    *,
    cache_position: torch.Tensor | None = None,
    input_embeds: torch.Tensor | None = None,
    allow_is_causal_skip: bool = True,
    block_sequence_ids: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor | None:
    """Leave no-padding SDPA prefill windowing to its compact attention path.

    This helper is installed only in model modules whose patched SDPA
    implementation consumes ``sliding_window``. Other callers retain
    :func:`vmap_create_sliding_window_causal_mask` and its dense fallback.
    """
    input_embeds = inputs_embeds if inputs_embeds is not None else input_embeds
    if _can_use_compact_sdpa_sliding_window(
        config,
        input_embeds,
        attention_mask,
        past_key_values,
        cache_position,
        allow_is_causal_skip,
        or_mask_function,
        and_mask_function,
        block_sequence_ids,
    ):
        return None
    return vmap_create_sliding_window_causal_mask(
        config,
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
        cache_position=cache_position,
        allow_is_causal_skip=allow_is_causal_skip,
        block_sequence_ids=block_sequence_ids,
        **kwargs,
    )


def vmap_create_causal_mask(
    config,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    position_ids: torch.Tensor | None = None,
    or_mask_function=None,
    and_mask_function=None,
    *,
    cache_position: torch.Tensor | None = None,
    input_embeds: torch.Tensor | None = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> torch.Tensor | None:
    """vmap-compatible create_causal_mask.

    Handles arbitrary batch dimensions from vmap.

    Original: inputs_embeds (batch, seq, hidden) -> mask (batch, 1, seq, seq)
    Under vmap with with_batch_dim: inputs_embeds (1, seq, hidden) -> mask (1, 1, seq, seq)
    Under vmap without with_batch_dim: inputs_embeds (seq, hidden) -> mask (1, 1, seq, seq)

    Signature spans v4 and v5: v4 passes ``input_embeds`` + ``cache_position``;
    v5 renames to ``inputs_embeds``, drops ``cache_position``, and may add
    ``block_sequence_ids``. All callers use keywords, so the flexible signature
    handles both.
    """
    input_embeds = inputs_embeds if inputs_embeds is not None else input_embeds
    # When no padding mask is provided AND the attention backend handles
    # causality internally (SDPA uses is_causal=True, flash uses masking
    # kernels), return None to avoid materializing the full mask tensor.
    # Eager attention needs an explicit causal mask — never skip for eager.
    # Note: past_key_values may be a DynamicCache even in training (HF's
    # @check_model_inputs resolves use_cache=None to config.use_cache=True),
    # so we check for actual cached data rather than just None.
    attn_impl = getattr(config, "_attn_implementation", None)
    all_valid_attention = attention_mask is None
    packed = _PACKED_SEQUENCES
    if not all_valid_attention and packed is not None:
        # Public policy: never inspect the batch content (see
        # ``set_packed_sequences``).
        all_valid_attention = packed
    elif (
        not all_valid_attention
        and attention_mask.ndim <= 2  # noqa: PLR2004 - padding masks are 1D/2D
        and not torch.compiler.is_compiling()
    ):
        try:
            functorch = torch._C._functorch
            physical_mask = attention_mask
            while functorch.is_functorch_wrapped_tensor(physical_mask):
                physical_mask = functorch.get_unwrapped(physical_mask)
            all_valid_attention = bool(physical_mask.all())
        except (AttributeError, RuntimeError):
            all_valid_attention = False

    if (
        allow_is_causal_skip
        and all_valid_attention
        and attn_impl != "eager"
        and _safe_seq_length(past_key_values) <= 0
    ):
        return None

    # Detect batchless input (under vmap without with_batch_dim) vs batched
    if input_embeds.ndim == 2:  # noqa: PLR2004 - batchless embeddings are 2D
        # Under vmap without batch dim: (seq_len, hidden_dim)
        batch_size = 1
        seq_len = input_embeds.shape[0]
    else:
        # Normal execution or under vmap with batch dim: (batch, seq_len, hidden_dim)
        batch_size = input_embeds.shape[0]
        seq_len = input_embeds.shape[1]

    # Determine target_length (total sequence length including cache)
    past_seen_tokens = _safe_seq_length(past_key_values)
    target_length = past_seen_tokens + seq_len

    # v5 drops cache_position; synthesize contiguous positions so the logic below
    # is version-agnostic. v4 supplies it and skips this branch.
    if cache_position is None:
        cache_position = torch.arange(
            past_seen_tokens, target_length, device=input_embeds.device
        )

    # Mask dtype follows autocast so SDPA's attn_mask matches the bf16 query.
    mask_dtype = _active_mask_dtype(input_embeds)
    # Create causal mask
    causal_mask = torch.full(
        (batch_size, 1, seq_len, target_length),
        torch.finfo(mask_dtype).min,
        dtype=mask_dtype,
        device=input_embeds.device,
    )

    # Fill the causal part (allow attention to past and current tokens)
    if seq_len > 1:
        # For each query position, allow attention to keys up to and including that position
        # cache_position may be batched under vmap - flatten it first to detect batching
        # Normal: cache_position.shape == (seq_len,) → flat has shape (seq_len,)
        # Batched: cache_position.shape == (batch, seq_len) → flat has shape (batch*seq_len,)
        cache_pos_flat = cache_position.view(-1)
        if cache_pos_flat.shape[0] == seq_len:
            # Normal case: cache_position has shape (seq_len,)
            # Use actual position values for the mask
            mask_cond = cache_pos_flat.view(seq_len, 1) >= cache_pos_flat.view(
                1, seq_len
            )
            # Expand to full target_length if we have past KV cache
            if target_length > seq_len:
                full_mask_cond = torch.zeros(
                    (seq_len, target_length),
                    dtype=torch.bool,
                    device=input_embeds.device,
                )
                full_mask_cond[:, :seq_len] = mask_cond
                # Can attend to all past cached tokens
                full_mask_cond[:, seq_len:target_length] = True
                mask_cond = full_mask_cond
        else:
            # Under vmap with batch dimension: cache_position shape (batch*seq_len,)
            # Just create a standard causal mask
            mask_cond = torch.tril(
                torch.ones(
                    (seq_len, seq_len), dtype=torch.bool, device=input_embeds.device
                )
            )
            if target_length > seq_len:
                full_mask_cond = torch.zeros(
                    (seq_len, target_length),
                    dtype=torch.bool,
                    device=input_embeds.device,
                )
                full_mask_cond[:, :seq_len] = mask_cond
                full_mask_cond[:, seq_len:target_length] = True
                mask_cond = full_mask_cond

        causal_mask[..., :seq_len, :target_length] = torch.where(
            mask_cond,
            torch.tensor(0.0, dtype=mask_dtype, device=input_embeds.device),
            causal_mask[..., :seq_len, :target_length],
        )
    else:
        # Single token: can attend to all cached tokens
        causal_mask[..., :, :target_length] = 0.0

    # Apply attention_mask if provided (padding mask)
    if attention_mask is not None:
        # attention_mask: (batch, seq) with 1 for valid, 0 for padding
        if attention_mask.ndim == 1:
            # Under vmap: (seq,) -> (1, seq)
            attention_mask = attention_mask.unsqueeze(0)

        # Expand to 4D: (batch, 1, 1, target_length)
        attention_mask = attention_mask[:, None, None, :]

        # Combine: set padding positions to -inf
        causal_mask = causal_mask.masked_fill(
            attention_mask == 0, torch.finfo(mask_dtype).min
        )

    return causal_mask


def vmap_create_sliding_window_causal_mask(
    config,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    position_ids: torch.Tensor | None = None,
    or_mask_function=None,
    and_mask_function=None,
    *,
    cache_position: torch.Tensor | None = None,
    input_embeds: torch.Tensor | None = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> torch.Tensor | None:
    """vmap-compatible ``create_sliding_window_causal_mask``.

    The stock implementation uses BlockMask / Flex Attention helpers that rely
    on data-dependent control flow incompatible with vmap.  This replacement
    builds a mask that enforces both the causal constraint (no future
    attention) and the sliding-window look-back limit
    (``config.sliding_window``). SDPA receives its native Boolean mask format;
    eager attention receives an additive mask.

    For each query at absolute position ``q_abs`` (given by ``cache_position``),
    key positions ``k_abs < q_abs - sliding_window + 1`` are set to ``-inf``.
    The causal upper-triangle is already blocked by the underlying
    ``vmap_create_causal_mask``; this function only adds the look-back limit.

    The mask is materialized once the key length reaches the window: SDPA's
    ``is_causal`` shortcut expresses causality but not the look-back limit.

    Signature is version-agnostic: v4 uses ``input_embeds`` + ``cache_position``,
    v5 renames to ``inputs_embeds`` and drops ``cache_position``.
    """
    input_embeds = inputs_embeds if inputs_embeds is not None else input_embeds
    sliding_window = getattr(config, "sliding_window", None)
    attn_impl = getattr(config, "_attn_implementation", None)
    past_seen_tokens = _safe_seq_length(past_key_values)

    # This generic fallback is used by model families without a compact
    # sliding-window SDPA forward. Its Boolean band avoids additive and batch
    # copies, but remains quadratic; patched families instead return ``None``
    # through vmap_create_compact_sdpa_sliding_window_causal_mask above.
    if (
        sliding_window is not None
        and input_embeds is not None
        and attention_mask is None
        and attn_impl == "sdpa"
        and past_seen_tokens == 0
        and cache_position is None
    ):
        target_length = _query_length(input_embeds)
        if allow_is_causal_skip and target_length <= sliding_window:
            return None
        return _sdpa_sliding_window_mask(input_embeds, sliding_window)

    if (
        allow_is_causal_skip
        and sliding_window is not None
        and input_embeds is not None
        and not _backend_enforces_sliding_window(attn_impl)
    ):
        target_length = past_seen_tokens + _query_length(input_embeds)
        allow_is_causal_skip = target_length <= sliding_window

    causal_mask = vmap_create_causal_mask(
        config,
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
        cache_position=cache_position,
        allow_is_causal_skip=allow_is_causal_skip,
    )

    if causal_mask is None:
        return None

    if sliding_window is None:
        return causal_mask

    # causal_mask shape: (batch_size, 1, seq_len, target_length)
    seq_len = causal_mask.shape[-2]
    device = input_embeds.device
    mask_dtype = causal_mask.dtype

    # Re-derive cache_position the same way vmap_create_causal_mask does.
    if cache_position is None:
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=device
        )

    # Absolute positions for every key slot in column order.
    # vmap_create_causal_mask lays the key dimension out as:
    #   cols 0..seq_len-1        → current tokens at cache_position[0..seq_len-1]
    #   cols seq_len..target_length-1 → past cached tokens at absolute positions 0..past_seen_tokens-1
    # (When past_seen_tokens == 0 the second slice is empty and
    #  key_abs_positions == cache_position == torch.arange(seq_len).)
    cache_pos_flat = cache_position.view(-1)
    if cache_pos_flat.shape[0] == seq_len:
        # Normal or vmap-without-batch-dim: cache_position is (seq_len,).
        query_abs_positions = cache_pos_flat  # (seq_len,)
        key_abs_positions = torch.cat(
            [cache_pos_flat, torch.arange(past_seen_tokens, device=device)]
        )  # (target_length,)
    else:
        # vmap-with-batch-dim: cache_position is (batch * seq_len,).
        # Fall back to contiguous positions starting at past_seen_tokens.
        query_abs_positions = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=device
        )
        key_abs_positions = torch.cat(
            [query_abs_positions, torch.arange(past_seen_tokens, device=device)]
        )  # (target_length,)

    # window_in_mask[q, k] = True  iff  k_abs >= q_abs - sliding_window + 1
    # shape: (seq_len, target_length)
    window_in_mask = key_abs_positions.unsqueeze(0) >= (
        query_abs_positions.unsqueeze(1) - sliding_window + 1
    )

    # Broadcast to (1, 1, seq_len, target_length) and block out-of-window slots.
    causal_mask = causal_mask.masked_fill(
        ~window_in_mask.unsqueeze(0).unsqueeze(0),
        torch.finfo(mask_dtype).min,
    )

    return causal_mask


def _vmap_safe_ignore_causal_mask_sdpa(*args, **kwargs) -> bool:
    """vmap-safe ``_ignore_causal_mask_sdpa``.

    The original calls ``padding_mask.all()`` — data-dependent control flow
    that breaks under vmap.  When ``padding_mask is None`` all remaining
    checks use Python ints so we delegate to the original.  When a mask is
    present we return ``False`` (force mask creation — needed for padding anyway).

    Arguments are forwarded unread.  Transformers 5.x inserted ``q_offset``
    ahead of ``kv_offset`` and calls this hook positionally, so a shim that
    names the parameters raises ``TypeError`` at every mask build.
    """
    padding_mask = args[0] if args else kwargs.get("padding_mask")
    if padding_mask is not None:
        return False
    return _vmap_safe_ignore_causal_mask_sdpa._original(*args, **kwargs)


def apply_masking_patches(*, vmap_masking: bool = True) -> None:
    """Apply patches to shared utilities used by all models.

    Patches:
    - transformers.masking_utils.create_causal_mask
    - transformers.masking_utils.create_sliding_window_causal_mask (Gemma2/Gemma3)
    - transformers.masking_utils._ignore_causal_mask_sdpa (vmap-safe)
    - transformers.integrations.sdpa_attention.repeat_kv

    These are required by all models (standard models, Gemma2, Gemma3, ...).
    """
    if vmap_masking is False:
        return

    # Patch shared masking_utils
    try:
        import transformers.masking_utils as masking_utils

        if hasattr(masking_utils, "create_causal_mask"):
            masking_utils.create_causal_mask = vmap_create_causal_mask

        # Models with a ``causal_mask_mapping`` (Gemma2, Gemma3) also call the
        # sliding variant; rebind it to a vmap-safe shim that delegates to the
        # standard causal-mask builder.
        if hasattr(masking_utils, "create_sliding_window_causal_mask"):
            masking_utils.create_sliding_window_causal_mask = (
                vmap_create_sliding_window_causal_mask
            )

        # Patch _ignore_causal_mask_sdpa for sliding-window models (Gemma2, Phi-3, Mistral).
        # The original calls padding_mask.all() which is data-dependent control flow
        # incompatible with vmap.
        if hasattr(masking_utils, "_ignore_causal_mask_sdpa"):
            current_fn = masking_utils._ignore_causal_mask_sdpa
            # Idempotency guard: avoid wrapping our own wrapper on repeated calls.
            if current_fn is not _vmap_safe_ignore_causal_mask_sdpa:
                _vmap_safe_ignore_causal_mask_sdpa._original = current_fn
                masking_utils._ignore_causal_mask_sdpa = (
                    _vmap_safe_ignore_causal_mask_sdpa
                )
    except ImportError:
        pass

    # Patch shared sdpa_attention (used by SDPA implementation)
    try:
        import transformers.integrations.sdpa_attention as sdpa_attention

        if hasattr(sdpa_attention, "repeat_kv"):
            sdpa_attention.repeat_kv = vmap_repeat_kv
    except ImportError:
        pass
