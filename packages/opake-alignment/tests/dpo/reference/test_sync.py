"""Unit tests for :func:`ema_update_reference` (TR-DPO core, §7.8).

Covers:
- Hand-computed EMA on a nested dict pytree (``{"a": tensor, "b": {"c": tensor}}``).
- Boundary cases: ``alpha=0`` (ref unchanged), ``alpha=1`` (copy policy).
- Non-mutation guarantee: original ``ref_params`` and ``policy_params`` are
  not modified by the call.
- Structure preservation: the returned pytree has exactly the same keys and
  nesting as the inputs.
"""

from __future__ import annotations

import torch

from opake.api.alignment.dpo.reference._sync import ema_update_reference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pytree() -> tuple[dict, dict]:
    """Return a pair of nested-dict pytrees for testing."""
    ref = {
        "a": torch.tensor([1.0, 2.0]),
        "b": {"c": torch.tensor([3.0, 4.0])},
    }
    policy = {
        "a": torch.tensor([5.0, 6.0]),
        "b": {"c": torch.tensor([7.0, 8.0])},
    }
    return ref, policy


# ---------------------------------------------------------------------------
# Hand-computed reference case
# ---------------------------------------------------------------------------


def test_ema_update_hand_computed() -> None:
    """EMA values match (1-alpha)*ref + alpha*policy for each leaf."""
    alpha = 0.3
    ref, policy = _make_pytree()

    result = ema_update_reference(ref, policy, alpha)

    # "a" leaf: (1 - 0.3) * [1, 2] + 0.3 * [5, 6] = [0.7 + 1.5, 1.4 + 1.8] = [2.2, 3.2]
    expected_a = torch.tensor([0.7 * 1.0 + 0.3 * 5.0, 0.7 * 2.0 + 0.3 * 6.0])
    # "b"/"c" leaf: (1 - 0.3) * [3, 4] + 0.3 * [7, 8] = [2.1 + 2.1, 2.8 + 2.4] = [4.2, 5.2]
    expected_bc = torch.tensor([0.7 * 3.0 + 0.3 * 7.0, 0.7 * 4.0 + 0.3 * 8.0])

    assert torch.allclose(result["a"], expected_a), (
        f"'a' mismatch: {result['a']} != {expected_a}"
    )
    assert torch.allclose(result["b"]["c"], expected_bc), (
        f"'b/c' mismatch: {result['b']['c']} != {expected_bc}"
    )


def test_ema_update_intermediate_alpha() -> None:
    """Spot-check a different alpha to rule out hard-coded coefficients."""
    alpha = 0.7
    ref = {"a": torch.tensor([10.0]), "b": {"c": torch.tensor([20.0])}}
    policy = {"a": torch.tensor([0.0]), "b": {"c": torch.tensor([0.0])}}

    result = ema_update_reference(ref, policy, alpha)

    # With policy=0: result = (1-alpha) * ref
    assert torch.allclose(result["a"], torch.tensor([3.0]))  # 0.3 * 10
    assert torch.allclose(result["b"]["c"], torch.tensor([6.0]))  # 0.3 * 20


# ---------------------------------------------------------------------------
# Boundary: alpha=0 keeps ref
# ---------------------------------------------------------------------------


def test_ema_update_alpha_zero_equals_ref() -> None:
    """alpha=0 must return values equal to ref_params (ref unchanged)."""
    ref, policy = _make_pytree()
    result = ema_update_reference(ref, policy, alpha=0.0)

    assert torch.allclose(result["a"], ref["a"]), "alpha=0: 'a' should equal ref"
    assert torch.allclose(result["b"]["c"], ref["b"]["c"]), (
        "alpha=0: 'b/c' should equal ref"
    )


# ---------------------------------------------------------------------------
# Boundary: alpha=1 copies policy
# ---------------------------------------------------------------------------


def test_ema_update_alpha_one_equals_policy() -> None:
    """alpha=1 must return values equal to policy_params."""
    ref, policy = _make_pytree()
    result = ema_update_reference(ref, policy, alpha=1.0)

    assert torch.allclose(result["a"], policy["a"]), "alpha=1: 'a' should equal policy"
    assert torch.allclose(result["b"]["c"], policy["b"]["c"]), (
        "alpha=1: 'b/c' should equal policy"
    )


# ---------------------------------------------------------------------------
# Non-mutation guarantee
# ---------------------------------------------------------------------------


def test_ema_update_does_not_mutate_ref() -> None:
    """ref_params must not be mutated by the call."""
    ref, policy = _make_pytree()

    # Clone originals for comparison.
    ref_a_orig = ref["a"].clone()
    ref_bc_orig = ref["b"]["c"].clone()

    _ = ema_update_reference(ref, policy, alpha=0.5)

    assert torch.allclose(ref["a"], ref_a_orig), "ref['a'] was mutated"
    assert torch.allclose(ref["b"]["c"], ref_bc_orig), "ref['b']['c'] was mutated"


def test_ema_update_does_not_mutate_policy() -> None:
    """policy_params must not be mutated by the call."""
    ref, policy = _make_pytree()

    policy_a_orig = policy["a"].clone()
    policy_bc_orig = policy["b"]["c"].clone()

    _ = ema_update_reference(ref, policy, alpha=0.5)

    assert torch.allclose(policy["a"], policy_a_orig), "policy['a'] was mutated"
    assert torch.allclose(policy["b"]["c"], policy_bc_orig), (
        "policy['b']['c'] was mutated"
    )


def test_ema_update_result_is_new_object() -> None:
    """The returned leaves must be distinct tensor objects from both inputs."""
    ref, policy = _make_pytree()
    result = ema_update_reference(ref, policy, alpha=0.5)

    assert result["a"] is not ref["a"]
    assert result["a"] is not policy["a"]
    assert result["b"]["c"] is not ref["b"]["c"]
    assert result["b"]["c"] is not policy["b"]["c"]


# ---------------------------------------------------------------------------
# Structure preservation
# ---------------------------------------------------------------------------


def test_ema_update_preserves_structure() -> None:
    """Returned pytree has the same keys and nesting as the inputs."""
    ref, policy = _make_pytree()
    result = ema_update_reference(ref, policy, alpha=0.4)

    assert isinstance(result, dict), "Top level should be a dict"
    assert set(result.keys()) == {"a", "b"}, (
        f"Unexpected top-level keys: {result.keys()}"
    )
    assert isinstance(result["b"], dict), "'b' should be a nested dict"
    assert set(result["b"].keys()) == {"c"}, (
        f"Unexpected nested keys: {result['b'].keys()}"
    )
    assert isinstance(result["a"], torch.Tensor), "'a' should be a Tensor"
    assert isinstance(result["b"]["c"], torch.Tensor), "'b/c' should be a Tensor"


def test_ema_update_preserves_tensor_shape_and_dtype() -> None:
    """Leaf tensors in the result have the same shape and dtype as inputs."""
    ref = {"w": torch.ones(3, 4, dtype=torch.float32)}
    policy = {"w": torch.zeros(3, 4, dtype=torch.float32)}
    result = ema_update_reference(ref, policy, alpha=0.5)

    assert result["w"].shape == (3, 4)
    assert result["w"].dtype == torch.float32


# ---------------------------------------------------------------------------
# vmap composition (TR-DPO loop contract)
# ---------------------------------------------------------------------------


def test_ema_update_composes_with_vmap_grad_outside_loop() -> None:
    """The TR-DPO outer-update + vmap inner-loss pattern produces finite grads.

    Mirrors the production loop:

        ref = ema_update_reference(ref, policy, alpha)   # outside vmap
        for batch in loader:
            grads = vmap(grad(loss(policy, *batch, ref=ref)))(batch)
            ...

    The closure reads ``policy_params`` (the vmapped argument) and the
    EMA'd ``ref_params`` (captured from the outer scope). The contract is
    that ``ref_params`` from a pure ``ema_update_reference`` call carries
    no autograd graph back into the policy update — so ``vmap(grad(...))``
    must produce finite gradients with the captured ref treated as a
    constant.
    """
    from torch.func import grad, vmap

    torch.manual_seed(0)
    # Tiny per-parameter-dict pytree: two leaves so structure preservation
    # under vmap is exercised.
    ref = {"w": torch.randn(4, 3), "b": torch.randn(4)}
    policy = {"w": torch.randn(4, 3), "b": torch.randn(4)}

    # One EMA update outside any vmap context.
    new_ref = ema_update_reference(ref, policy, alpha=0.1)
    # The new ref must not carry a grad_fn — ema_update_reference is pure
    # and intended to be safe to use as a captured constant inside vmap.
    assert new_ref["w"].grad_fn is None
    assert new_ref["b"].grad_fn is None

    # Per-example synthetic batch: 5 examples, each a (3,)-vector input + scalar
    # target. Loss: squared error from (policy @ x + bias) − target, with the
    # EMA'd ref subtracted from policy to produce a TR-DPO-style "deviation"
    # term. Whatever the closure does, the key invariant is that grad flows
    # only through policy and produces finite gradients under vmap.
    B, H, K = 5, 3, 4
    xs = torch.randn(B, H)
    ys = torch.randn(B, K)

    def per_example_loss(p, x, y):
        # `new_ref` captured from outer scope; treated as constant.
        deviation = p["w"] - new_ref["w"]  # (K, H)
        pred = p["w"] @ x + p["b"]  # (K,)
        return ((pred - y) ** 2).sum() + 0.01 * (deviation**2).sum()

    g = vmap(grad(per_example_loss), in_dims=(None, 0, 0))(policy, xs, ys)
    assert isinstance(g, dict), "grad over a dict pytree must return a dict"
    assert set(g.keys()) == {"w", "b"}
    # in_dims=None for the policy means vmap broadcasts it across examples;
    # each per-example grad contributes a separate (leaf_shape) tensor,
    # stacked along the batch dim — exactly what DP-SGD per-example
    # clipping operates on.
    assert g["w"].shape == (B, *policy["w"].shape), (
        f"expected (B={B}, *policy['w'].shape={tuple(policy['w'].shape)}), "
        f"got {tuple(g['w'].shape)}"
    )
    assert g["b"].shape == (B, *policy["b"].shape)
    assert torch.isfinite(g["w"]).all()
    assert torch.isfinite(g["b"]).all()


def test_ema_update_alternated_with_vmap_no_graph_accumulation() -> None:
    """Alternating EMA updates + vmap(grad(...)) doesn't accumulate autograd graph.

    Simulates several TR-DPO outer-step iterations: EMA-update the reference,
    take a vmap-grad-based "policy update," EMA-update again with the new
    policy, etc. After several iterations, the reference pytree must still
    carry no grad_fn — if EMA accidentally retained a graph through the
    captured-policy path, this test would surface it via a non-None
    ``grad_fn`` on the EMA result.
    """
    from torch.func import grad, vmap

    torch.manual_seed(1)
    ref = {"w": torch.randn(3, 2)}
    policy = {"w": torch.randn(3, 2)}

    B, H, K = 4, 2, 3
    xs = torch.randn(B, H)
    ys = torch.randn(B, K)

    def per_example_loss(p, x, y, ref_w):
        return ((p["w"] @ x - y) ** 2).sum() + 0.01 * ((p["w"] - ref_w) ** 2).sum()

    for _step in range(3):
        ref = ema_update_reference(ref, policy, alpha=0.2)
        assert ref["w"].grad_fn is None, (
            f"EMA reference must not carry grad_fn after step {_step}"
        )
        g = vmap(grad(per_example_loss), in_dims=(None, 0, 0, None))(
            policy, xs, ys, ref["w"]
        )
        assert torch.isfinite(g["w"]).all()
        # Apply a vanilla gradient step on the policy and continue.
        policy = {"w": policy["w"] - 0.01 * g["w"]}
