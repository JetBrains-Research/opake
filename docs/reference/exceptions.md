# Exceptions

`opake.exceptions` provides stable semantic categories for handling failures
from public Opake APIs:

```python
from opake.exceptions import CalibrationError, CheckpointError, ConfigurationError

try:
    result = calibrate(...)
except CalibrationError:
    # Use different search bounds or inspect the process definition.
    ...
except ConfigurationError:
    # Correct an invalid or incompatible user option.
    ...
except CheckpointError:
    # Do not resume until the complete DP checkpoint is available.
    ...
```

| Exception | Common error family | Use it for |
|---|---|---|
| `OpakeError` | `Exception` | Handling any semantic Opake failure. |
| `ConfigurationError` | `ValueError` | Invalid values, incompatible options, and unsupported user configuration. |
| `CalibrationError` | `ConfigurationError` | Invalid or unsatisfied privacy-calibration searches. |
| `PrivacyBudgetError` | `ConfigurationError` | Invalid privacy-budget definitions. |
| `InputTypeError` | `TypeError` | Arguments whose Python type cannot satisfy an Opake API contract. |
| `OperationError` | `RuntimeError` | An Opake operation that cannot complete in its current state. |
| `CheckpointError` | `OperationError` | Saving, restoring, or resuming an incompatible or incomplete Opake checkpoint. |

Standard Python and third-party errors still surface when they are more
specific, such as an I/O failure while opening a caller-provided path.

::: opake.exceptions
    options:
      show_source: true
      heading_level: 2
