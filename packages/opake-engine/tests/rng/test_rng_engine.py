"""Unit tests for generic RNG engine (JAX-style semantics)."""

import pytest
import torch

from opake.random import fold_in, generator_from_key, key, split


def test_key_requires_int_seed():
    with pytest.raises(TypeError, match="seed must be int"):
        key("42")


def test_split_is_deterministic_for_same_key():
    k1 = key(123)
    k2 = key(123)
    c11, c12 = split(k1, 2)
    c21, c22 = split(k2, 2)
    assert c11.seed == c21.seed
    assert c12.seed == c22.seed


def test_split_children_are_distinct():
    k = key(123)
    c1, c2, c3 = split(k, 3)
    assert len({c1.seed, c2.seed, c3.seed}) == 3


def test_fold_in_domain_separates():
    k = key(123)
    a = fold_in(k, "noise")
    b = fold_in(k, "sampling")
    assert a.seed != b.seed


def test_generator_from_key_is_reproducible():
    k = key(999)
    g1 = generator_from_key(k)
    g2 = generator_from_key(k)
    x1 = torch.randn(16, generator=g1)
    x2 = torch.randn(16, generator=g2)
    assert torch.allclose(x1, x2)


def test_fold_in_variadic_equals_sequential():
    """fold_in(k, a, b) must equal fold_in(fold_in(k, a), b)."""
    k = key(42)
    chained = fold_in(fold_in(k, 7), 3)
    variadic = fold_in(k, 7, 3)
    assert chained.seed == variadic.seed


def test_fold_in_rejects_bool():
    k = key(42)
    with pytest.raises(TypeError, match="int or str"):
        fold_in(k, True)


def test_fold_in_rejects_non_int_str_data():
    k = key(42)
    with pytest.raises(TypeError, match="int or str"):
        fold_in(k, 1.5)
    with pytest.raises(TypeError, match="int or str"):
        fold_in(k, [1])
    with pytest.raises(TypeError, match="rng_key must be RngKey"):
        fold_in(42, 0)  # type: ignore[arg-type]


def test_split_rejects_non_int_num():
    k = key(42)
    with pytest.raises(TypeError, match="num must be int"):
        split(k, True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="num must be int"):
        split(k, 2.0)  # type: ignore[arg-type]


def test_key_rejects_bool():
    with pytest.raises(TypeError, match="seed must be int"):
        key(True)


def test_integer_and_string_tags_occupy_disjoint_spaces():
    base = key(1234)
    for value in (0, 1, 2, 17):
        assert fold_in(base, value).seed != fold_in(base, str(value)).seed


def test_integer_and_string_tags_are_structurally_disjoint():
    base = key(42)
    integer_payload = 1
    string_payload = "\x01" + "\x00" * 15

    assert fold_in(base, integer_payload).seed != fold_in(base, string_payload).seed


def test_fold_in_variadic_three_values():
    """fold_in(k, a, b, c) == fold_in(fold_in(fold_in(k, a), b), c)."""
    k = key(0)
    chained = fold_in(fold_in(fold_in(k, 1), 2), 3)
    variadic = fold_in(k, 1, 2, 3)
    assert chained.seed == variadic.seed


def test_fold_in_no_data_raises():
    """fold_in(k) with no data arguments must raise ValueError."""
    k = key(42)
    with pytest.raises(ValueError, match="at least one data argument"):
        fold_in(k)
