"""Per-client clipping as an IFED strategy, shipped cloudpickled to the executor.

The phases below run in IFED's executor, which has ``torch`` but no ``opaque``:
this module references nothing from ``opaque.*``, and the SDK names it uses
travel inside the plan bundle by value.
"""

# NO `from __future__ import annotations` here: the executor resolves the
# phases' hints at runtime (it reads the agent-state class off aggregate's
# annotation), so they must be concrete objects and not strings evaluated
# against globals the executor does not have.
import torch
from ifed import AgentState, AggrState, FedSgd, InterState, MetricsBundle, ServerState

_EPS = 1e-12


def _clip(
    params: dict[str, torch.Tensor] | None, clipping_norm: float
) -> dict[str, torch.Tensor] | None:
    """Scale one client's whole gradient pytree to L2 norm ``clipping_norm``."""
    if params is None:
        return None
    if not params:
        return {}
    # float32 accumulation: a half-precision gradient's own squares can overflow
    squares = torch.stack(
        [value.detach().float().pow(2).sum() for value in params.values()]
    )
    norm = float(torch.sqrt(squares.sum()))
    # a nan norm leaves scale at 1.0, so the round drops the client instead of hiding it
    scale = min(1.0, clipping_norm / (norm + _EPS))
    return {name: value * scale for name, value in params.items()}


class _ClippedSum(FedSgd):
    """``FedSgd`` whose round releases the clipped sum instead of a step.

    Two phases change:

    - ``aggregate`` clips every client's gradient to ``clipping_norm`` before
      the round combines them, and undoes the equal-weight average so what it
      returns is the sum. Clipping per client before summation is what bounds
      the sum's sensitivity to any one client by ``clipping_norm`` — the
      contract a central-DP noise mechanism needs, with the *client* as the
      privacy unit.
    - ``finalize`` carries that sum out untouched rather than taking a server
      SGD step: the step belongs to Opaque's own noise → optimizer chain.

    ``weighted=False`` with ``max_skipped=0.0`` is what keeps the divisor
    data-independent: every client counts once, and a round with even one
    unusable client fails instead of quietly averaging over the survivors.
    Its metrics bundle is emptied on the way out, so a round's only release is
    the clipped sum.
    """

    def __init__(self, clipping_norm: float):
        super().__init__(weighted=False, max_skipped=0.0)
        self.clipping_norm = clipping_norm

    def aggregate(self, agent_states: list[AgentState]) -> AggrState:
        clipped = [
            state._replace(params=_clip(state.params, self.clipping_norm))
            for state in agent_states
        ]
        aggregated = FedSgd.aggregate(self, clipped)
        # max_skipped=0.0 means every client contributed, so the count is the cohort size
        cohort = float(len(clipped))
        return AggrState(
            params={name: value * cohort for name, value in aggregated.params.items()},
            metrics=MetricsBundle(scalars={}, histograms={}),
        )

    def finalize(self, aggr_state: AggrState, inter_state: InterState) -> ServerState:
        return ServerState(
            params=aggr_state.params,
            round=inter_state.round + 1,
            metrics=aggr_state.metrics,
        )


def clipped_sum(*, clipping_norm: float) -> _ClippedSum:
    """Build the IFED strategy that releases a per-client-clipped gradient sum.

    Pass it to ``ifed.build_train(strategy=…)`` and drive the resulting plan
    with :func:`opaque.api.federated.clipping.clipped_grad`. Build the plan
    with ``batch_size=None`` so a client's contribution is one gradient over
    all of its own rows; with a smaller local batch, what a client sends
    depends on how the agent runtime accumulates across batches, which a
    per-client sensitivity bound cannot rest on.

    The strategy is meant for that driven session only. Under
    ``ifed.submit`` nothing consumes the sum it releases, and the next round
    would start from it as if it were weights.

    Args:
        clipping_norm: Per-client L2 clipping threshold ``C``.

    Returns:
        A strategy whose round releases ``sum_i clip(g_i, C)``, of sensitivity
        ``C`` under add-or-remove adjacency on clients.
    """
    if clipping_norm <= 0:
        raise ValueError(*(f"clipping_norm must be > 0, got {clipping_norm}",))
    return _ClippedSum(clipping_norm)


__all__ = ["clipped_sum"]
