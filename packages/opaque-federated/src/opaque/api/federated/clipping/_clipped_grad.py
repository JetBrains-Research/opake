"""Federated per-client clipped gradients — the twin of central ``clipped_grad``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opaque.api.engine.clipping.types import FixedClipState
from opaque.api.engine.types import ClippedPytree, clipped
from opaque.exceptions import ConfigurationError, InputTypeError

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.federated.data.types import Cohort


def clipped_grad(
    session: Any,
    strategy: Any,
    *,
    normalize_by: float | None = None,
    timeout: float | None = None,
) -> tuple[Callable, FixedClipState]:
    """Drive one federated round per call, returning its clipped gradient.

    The federated twin of ``opaque.dpsgd.clipping.clipped_grad``: the batch
    axis is a cohort of clients and IFED executes each round. ``session``
    already fixes the population, the cardinality and the assignment
    separation, and ``strategy`` — a
    :func:`~opaque.api.federated.clipping.clipped_sum` — fixes the per-client
    clipping threshold, so this call only threads parameters in and gradients
    out::

        strategy = fed.clipped_sum(clipping_norm=1.0)
        plan = ifed.build_train(net=net, source=Iris, loss=ifed.Loss.mse,
                                batch_size=None, strategy=strategy)
        with ifed.session(plan, store) as run:
            params = plan.init_state.params
            grad_fn, clip_state = fed.clipped_grad(run, strategy)
            for cohort in loader:
                grads, clip_state = grad_fn(params, cohort, state=clip_state)

    ``grads`` is a ``ClippedPytree`` with ``max_norm = clipping_norm /
    normalize_by``, so the noise → optimizer → accountant chain downstream is
    the central DP-SGD one unchanged.

    Args:
        session: An open ``ifed.session(...)`` over a plan built with
            ``strategy``.
        strategy: The :func:`clipped_sum` strategy that plan was built with;
            its ``clipping_norm`` is the sensitivity the result advertises.
        normalize_by: Gradient normalization constant. Defaults to the cohort
            size, giving averaged gradients of sensitivity ``C / k``.
        timeout: Seconds one round may take, for a remote session that
            supports it. Reaching it cancels the round.

    Returns:
        ``(grad_fn, clip_state)`` where ``grad_fn(params, cohort, *, state)``
        returns ``(ClippedPytree, state)``, and ``clip_state`` is the same
        fieldless :class:`FixedClipState` marker central ``clipped_grad``
        threads.
    """
    clipping_norm = getattr(strategy, "clipping_norm", None)
    if clipping_norm is None:
        raise InputTypeError(
            *(
                "strategy must be an opaque.federated.clipped_sum(...) — a plan "
                "built with any other strategy releases something other than "
                "the per-client-clipped sum this returns a max_norm for",
            )
        )
    from ifed import MetricsBundle, ServerState

    bound: dict[str, Any] = {}
    expected_round = 0

    def grad_fn(
        params: dict, cohort: Cohort, *, state: FixedClipState
    ) -> tuple[ClippedPytree, FixedClipState]:
        nonlocal expected_round
        if cohort.origin is None or cohort.population is None:
            raise ConfigurationError(
                *(
                    "cohorts must come from opaque.federated.DataLoader — a raw "
                    "Cohort carries no population or origin to check the round "
                    "against",
                )
            )
        if not bound:
            bound["origin"] = cohort.origin
            bound["size"] = cohort.size
            bound["separation"] = cohort.separation
            bound["normalize_by"] = (
                float(normalize_by) if normalize_by is not None else float(cohort.size)
            )
        elif cohort.origin is not bound["origin"]:
            raise ConfigurationError(
                *("cohort comes from a different DataLoader than round 0's",)
            )
        elif cohort.size != bound["size"] or cohort.separation != bound["separation"]:
            raise ConfigurationError(
                *(
                    "cohort size/separation changed mid-run; a task fixes both "
                    "for its lifetime, and the accounting is computed from them",
                )
            )
        if cohort.round != expected_round:
            raise ConfigurationError(
                *(
                    f"out-of-order cohort: expected round {expected_round}, got "
                    f"{cohort.round}",
                )
            )
        expected_round += 1

        seed = ServerState(
            params=params,
            round=cohort.round,
            metrics=MetricsBundle(scalars={}, histograms={}),
        )
        # no cardinality= override: it would change the divisor the sensitivity is stated for
        out = (
            session.step(seed)
            if timeout is None
            else session.step(seed, timeout=timeout)
        )
        divisor = bound["normalize_by"]
        grads = {name: value / divisor for name, value in out.params.items()}
        return clipped(grads, max_norm=clipping_norm / divisor), state

    return grad_fn, FixedClipState()


__all__ = ["clipped_grad"]
