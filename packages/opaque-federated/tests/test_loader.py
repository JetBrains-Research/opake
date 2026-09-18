"""DataLoader: round horizon, cohort stamping, origin identity, serialization."""

import pytest

from opaque.api.federated.data.types import (
    Cohort as InternalCohort,
)
from opaque.api.federated.data.types import (
    Population as InternalPopulation,
)
from opaque.federated import (
    Cohort,
    DataLoader,
    MinSepSampler,
    Population,
)
from opaque.federated import (
    population as make_population,
)
from opaque.federated.data import (
    DataLoader as DataLoaderFromData,
)
from opaque.federated.data import (
    population as population_from_data,
)
from opaque.federated.data.types import (
    Cohort as FacadeCohort,
)
from opaque.federated.data.types import (
    Population as FacadePopulation,
)
from opaque.serialization import from_state_dict, state_dict


@pytest.fixture
def population():
    return make_population("/hive")


def _loader(population, rounds=5, batch_size=2, bands=2):
    sampler = MinSepSampler(population, batch_size=batch_size, bands=bands)
    return DataLoader(population, batch_sampler=sampler, rounds=rounds)


def test_population_factory_and_type_facades():
    value = make_population("/hive", version="1.*")
    assert value == population_from_data("/hive", version="1.*")
    assert value == Population(name="/hive", version="1.*")
    assert value.version == "1.*"
    assert Population is FacadePopulation is InternalPopulation
    assert Cohort is FacadeCohort is InternalCohort
    assert DataLoaderFromData is DataLoader


@pytest.mark.parametrize("name", ["hive", "", 1])
def test_population_factory_rejects_invalid_name(name):
    with pytest.raises((TypeError, ValueError), match="population name"):
        make_population(name)


def test_population_factory_rejects_empty_version():
    with pytest.raises(ValueError, match="version"):
        make_population("/hive", version="")


def test_yields_exactly_rounds_cohorts(population):
    loader = _loader(population, rounds=5)
    cohorts = list(loader)
    assert len(cohorts) == 5
    assert [c.round for c in cohorts] == [0, 1, 2, 3, 4]


def test_stamps_loader_fields(population):
    loader = _loader(population, rounds=3, batch_size=4, bands=3)
    cohorts = list(loader)
    assert all(c.rounds == 3 for c in cohorts)
    assert all(c.population == population for c in cohorts)
    assert all(c.origin is not None for c in cohorts)
    assert len({id(c.origin) for c in cohorts}) == 1  # one origin per loader
    assert all(c.size == 4 and c.separation == 2 for c in cohorts)


def test_distinct_loaders_have_distinct_origins(population):
    a = next(iter(_loader(population)))
    b = next(iter(_loader(population)))
    assert a.origin is not b.origin


def test_len_counts_down(population):
    loader = _loader(population, rounds=4)
    assert len(loader) == 4
    it = iter(loader)
    next(it)
    next(it)
    assert len(loader) == 2


def test_population_mismatch_raises(population):
    sampler = MinSepSampler(population, batch_size=2, bands=2)
    other = Population(name="/other")
    with pytest.raises(ValueError, match="does not match loader population"):
        DataLoader(other, batch_sampler=sampler, rounds=3)


def test_rounds_validation(population):
    sampler = MinSepSampler(population, batch_size=2, bands=2)
    with pytest.raises(ValueError, match="rounds"):
        DataLoader(population, batch_sampler=sampler, rounds=0)


def test_serialization_roundtrip(population):
    loader = _loader(population, rounds=6, batch_size=3, bands=2)
    it = iter(loader)
    next(it)
    next(it)
    sd = state_dict(loader)

    restored = from_state_dict(_loader(population, rounds=6, batch_size=3, bands=2), sd)
    assert restored.rounds == 6
    assert restored.consumed == 2
    remaining = list(restored)
    assert [c.round for c in remaining] == [2, 3, 4, 5]


def test_serialization_retains_population_version():
    original_population = make_population("/hive", version="1.*")
    loader = _loader(original_population)
    snapshot = state_dict(loader)
    assert snapshot["population_version"] == "1.*"

    restored = from_state_dict(_loader(original_population), snapshot)
    assert restored.population.version == "1.*"

    other_version = make_population("/hive", version="2.*")
    with pytest.raises(ValueError, match="population"):
        from_state_dict(_loader(other_version), snapshot)
