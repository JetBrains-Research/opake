"""Serialization of torch.Tensor subclasses and vendor structure handles.

``nn.Parameter`` is what ``make_functional`` returns; it must round-trip as a
Parameter, not be dropped by the exact-type registry.  ``optree.PyTreeSpec``
appears in optimizer / MF-noise state and is declared inert (rebuilt from the
template) rather than raising.
"""

from __future__ import annotations

import optree
import torch
import torch.nn as nn

from opake.functional import make_functional
from opake.serialization import from_state_dict, state_dict


def test_named_parameters_roundtrip() -> None:
    model = nn.Linear(3, 2)
    params = dict(model.named_parameters())

    sd = state_dict(params)
    assert sorted(sd) == ["bias", "weight"]

    template = {k: nn.Parameter(torch.zeros_like(v)) for k, v in params.items()}
    out = from_state_dict(template, sd)

    for name, restored in out.items():
        assert isinstance(restored, nn.Parameter)
        assert restored.requires_grad == params[name].requires_grad
        assert torch.equal(restored, params[name])


def test_functional_params_roundtrip() -> None:
    _fmodel, params = make_functional(nn.Linear(4, 3))
    assert all(isinstance(p, nn.Parameter) for p in params)

    sd = state_dict({"params": params})
    assert sorted(sd) == ["params[0]", "params[1]"]

    template = {"params": tuple(nn.Parameter(torch.zeros_like(p)) for p in params)}
    out = from_state_dict(template, sd)
    for restored, original in zip(out["params"], params, strict=True):
        assert isinstance(restored, nn.Parameter)
        assert torch.equal(restored, original)


def test_frozen_parameter_requires_grad_preserved() -> None:
    frozen = nn.Parameter(torch.arange(3.0), requires_grad=False)
    sd = state_dict({"p": frozen})
    template = {"p": nn.Parameter(torch.zeros(3), requires_grad=False)}
    out = from_state_dict(template, sd)
    assert out["p"].requires_grad is False
    assert torch.equal(out["p"], frozen)


def test_pytreespec_is_inert() -> None:
    spec = optree.tree_structure({"a": 0, "b": [1, 2]})
    sd = state_dict({"spec": spec})
    assert sd == {}

    template_spec = optree.tree_structure({"a": 0, "b": [1, 2]})
    out = from_state_dict({"spec": template_spec}, sd)
    assert out["spec"] is template_spec
