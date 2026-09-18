"""Round loader — the federated ``DataLoader``."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from opaque.api.federated.data.types import Cohort, Population
from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from opaque.api.federated.sampling import MinSepSampler


class DataLoader:
    """Iterate a population for ``rounds`` cohorts.

    The federated twin of ``torch.utils.data.DataLoader``: it draws per-round
    cohort specs from ``batch_sampler`` and stamps each with the loader's
    identity — the round index, the total ``rounds`` horizon, the
    ``population``, and an ``origin`` token. The consumer
    (:func:`opaque.api.federated.clipping.clipped_grad`'s ``grad_fn``) uses
    those stamps to verify every cohort belongs to the same loader and arrives
    in order.

    Unlike a central loader it yields **symbolic** batches: the per-client
    contributions depend on the model parameters, so a cohort is resolved into
    data only by executing its round.

    Args:
        population: The Opaque :class:`Population` to iterate.
        batch_sampler: The cohort sampler (e.g. :class:`MinSepSampler`); its
            population must match ``population``.
        rounds: The round horizon — how many cohorts to yield.
    """

    def __init__(
        self,
        population: Population,
        *,
        batch_sampler: MinSepSampler,
        rounds: int,
    ):
        if rounds < 1:
            raise ConfigurationError(*(f"rounds must be >= 1, got {rounds}",))
        sampler_population = getattr(batch_sampler, "population", None)
        if sampler_population is not None and sampler_population != population:
            raise ConfigurationError(
                *(
                    f"batch_sampler population {sampler_population.name!r} does "
                    f"not match loader population {population.name!r}",
                )
            )
        self.population = population
        self.batch_sampler = batch_sampler
        self.rounds = rounds
        self._consumed = 0
        self._origin = object()

    @property
    def consumed(self) -> int:
        """Number of cohorts yielded so far."""
        return self._consumed

    def __len__(self) -> int:
        return self.rounds - self._consumed

    def __iter__(self) -> Iterator[Cohort]:
        sampler = iter(self.batch_sampler)
        while self._consumed < self.rounds:
            spec = next(sampler)
            cohort = dataclasses.replace(
                spec,
                round=self._consumed,
                rounds=self.rounds,
                population=self.population,
                origin=self._origin,
            )
            self._consumed += 1
            yield cohort

    def __repr__(self) -> str:
        return (
            f"DataLoader(population={self.population.name!r}, "
            f"batch_sampler={self.batch_sampler!r}, rounds={self.rounds})"
        )


def _state_dict_loader(loader: DataLoader) -> dict[str, Any]:
    from opaque.serialization import state_dict

    return {
        "population_name": loader.population.name,
        "population_version": loader.population.version,
        "rounds": int(loader.rounds),
        "consumed": int(loader.consumed),
        "batch_sampler": state_dict(loader.batch_sampler),
    }


def _from_state_dict_loader(template: DataLoader, sd: Mapping[str, Any]) -> DataLoader:
    from opaque.serialization import from_state_dict

    saved = Population(
        name=str(sd["population_name"]),
        version=str(sd.get("population_version", "*")),
    )
    have = template.population
    if saved != have:
        raise ConfigurationError(
            *(
                f"DataLoader.from_state_dict: template population {have!r} does "
                f"not match snapshot {saved!r}.",
            )
        )
    loader = DataLoader(
        template.population,
        batch_sampler=from_state_dict(template.batch_sampler, sd["batch_sampler"]),
        rounds=int(sd["rounds"]),
    )
    loader._consumed = int(sd["consumed"])
    return loader


def _register_loader_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(DataLoader, _state_dict_loader, _from_state_dict_loader)


_register_loader_serializer()
