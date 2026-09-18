"""MinSepSampler: cohort stream, policy compilation, serialization."""

from itertools import islice

import pytest

from opaque.federated import (
    MinSepSampler,
    Population,
)
from opaque.federated import (
    population as make_population,
)
from opaque.serialization import from_state_dict, state_dict


@pytest.fixture
def population():
    return make_population("/hive")


def test_yields_cohort_specs(population):
    sampler = MinSepSampler(population, batch_size=8, bands=4)
    specs = list(islice(iter(sampler), 3))
    assert [s.round for s in specs] == [0, 1, 2]
    assert all(s.size == 8 for s in specs)
    assert all(s.separation == 3 for s in specs)  # bands - 1
    # loader-only fields are unstamped on raw sampler output
    assert all(
        s.rounds is None and s.population is None and s.origin is None for s in specs
    )


def test_bands_one_means_no_separation(population):
    sampler = MinSepSampler(population, batch_size=2, bands=1)
    (spec,) = islice(iter(sampler), 1)
    assert spec.separation == 0


def test_unbounded_and_consumed(population):
    sampler = MinSepSampler(population, batch_size=2, bands=2)
    assert sampler.consumed == 0
    list(islice(iter(sampler), 100))
    assert sampler.consumed == 100
    list(islice(iter(sampler), 5))  # keeps going — no n_steps
    assert sampler.consumed == 105


def test_validation(population):
    with pytest.raises(TypeError, match=r"opaque\.federated\.Population"):
        MinSepSampler("/hive", batch_size=2, bands=2)
    with pytest.raises(ValueError, match="batch_size"):
        MinSepSampler(population, batch_size=0, bands=2)
    with pytest.raises(ValueError, match="bands"):
        MinSepSampler(population, batch_size=2, bands=0)


def test_accounting_attributes(population):
    """The attributes non-amplified BandMF accounting reads."""
    sampler = MinSepSampler(population, batch_size=8, bands=4)
    assert sampler.bands == 4
    assert sampler.batch_size == 8
    assert sampler.assign_delta == 3


def test_serialization_roundtrip(population):
    sampler = MinSepSampler(population, batch_size=8, bands=4)
    list(islice(iter(sampler), 7))
    sd = state_dict(sampler)

    template = MinSepSampler(population, batch_size=8, bands=4)
    restored = from_state_dict(template, sd)
    assert restored.batch_size == 8
    assert restored.bands == 4
    assert restored.consumed == 7
    (spec,) = islice(iter(restored), 1)
    assert spec.round == 7


def test_serialization_rejects_population_mismatch(population):
    sampler = MinSepSampler(population, batch_size=2, bands=2)
    sd = state_dict(sampler)
    other = MinSepSampler(Population(name="/other"), batch_size=2, bands=2)
    with pytest.raises(ValueError, match="population"):
        from_state_dict(other, sd)


def test_serialization_rejects_population_version_mismatch():
    population = make_population("/hive", version="1.*")
    sampler = MinSepSampler(population, batch_size=2, bands=2)
    snapshot = state_dict(sampler)
    assert snapshot["population_version"] == "1.*"

    other = MinSepSampler(
        make_population("/hive", version="2.*"), batch_size=2, bands=2
    )
    with pytest.raises(ValueError, match="population"):
        from_state_dict(other, snapshot)
