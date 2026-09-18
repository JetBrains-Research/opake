# Serialization

Opake keeps training state in explicit values (clip state, noise state,
optimizer state, schedules, [`Accountant`](accounting.md) / [`DpProcess`](accounting.md),
…). Use `opake.serialization` for a single **flat**
`dict[str, Any]` suitable for `torch.save` / `torch.load`.

Restore is **template-driven**: pass a freshly constructed object of the same
shape as at save time; keys present in the checkpoint overwrite leaves, and
missing keys keep template values (forward compatibility when new fields appear).

Dispatch resolves a leaf by exact type and then by `__mro__`, so a subclass
of a registered type — a custom `torch.Tensor` subclass, for example — is
serialized by the nearest base-class handler rather than dropped.
(`torch.nn.Parameter` is common enough to get its own exact-type handler that
preserves the subclass and its `requires_grad` flag on restore, so it does
not rely on the `__mro__` fallback.) A leaf that is neither registered nor a
generic container (dataclass / NamedTuple / tuple / list / mapping) nor a
primitive raises `InputTypeError` on both save and restore instead of being
silently skipped.
Genuinely inert leaves that the template reproduces (vendor structure handles
such as `optree.PyTreeSpec`) are declared with
`opake.serialization.register_template_restored`; nothing is written for
them and the template supplies them on load.

`PerGroup` checkpoints through the same API. Path keys serialize as their
`str()` form under the structural walker — for example, `groups.('a',)` for a
flat leaf, or `groups.('layer', 'weight')` for a nested path — alongside
`values.<group_name>`. When DP bias correction is enabled on Adam-family
optimizers, `phi` is a path-keyed dict from `opt.init` so `from_state_dict`
round-trips without resetting φ. NumPy `ndarray` leaves are supported alongside
`torch.Tensor`.

Domain pages with examples: [Optimizers](optimizers.md), [Accounting](accounting.md).

## Array leaf policy

`torch.Tensor`, `torch.nn.Parameter` and `numpy.ndarray` leaves resolve three
attributes against the template, each under its own rule:

- **Shape** — must match the template exactly; a mismatch raises `CheckpointError`
  for every array leaf kind. Broadcast-compatible shapes (a length-1 buffer
  against a length-*d* slot) are what make the check load-bearing: they would
  otherwise restore without an error and keep training against state the
  accountant no longer prices.
- **Dtype** — taken from the template, and the checkpoint value is cast to it.
  The template carries the live compute dtype, so resuming an fp32 checkpoint
  in bf16 works.
- **Device** — taken from the template for tensors, so a checkpoint read with
  `map_location="cpu"` lands on the training device. `ndarray` leaves have no
  device.

A leaf absent from the checkpoint keeps the template's value; that is the
forward-compatibility rule above, not a mismatch. An error raised while
restoring a registered leaf carries the offending key as an exception note.

## API surface (no per-type `state_dict()`)

Opake centralizes (de)serialization in `opake.serialization.state_dict`
and `opake.serialization.from_state_dict`. Registered types (including
`Accountant`) are written and restored **only**
through these module-level functions — there are no instance methods named
`state_dict` / `from_state_dict` on those classes.

Minimal round trip for an `Accountant`:

```python
from opake.accounting import Accountant, identity
from opake.serialization import from_state_dict, state_dict

acct = Accountant() | identity()
flat = state_dict(acct)
acct2 = from_state_dict(Accountant(), flat)
```

The same pattern applies to clip state, noise state, functional optimizer
state, and any other value that flows through the DP training loop.  Custom
types may register handlers with `opake.serialization.register_serializer`,
or — when a leaf carries no run state and the template reproduces it —
declare it inert with `opake.serialization.register_template_restored`.

::: opake.serialization
    options:
      show_source: true
      heading_level: 2
