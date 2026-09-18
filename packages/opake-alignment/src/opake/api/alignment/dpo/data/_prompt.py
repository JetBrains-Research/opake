"""Prompt-extraction heuristic for preference and unpaired datasets.

:func:`extract_prompt` ports the ``extract_prompt`` heuristic from
``trl/data_utils.py``. It is a pure-Python dataset transform (no torch
required) that works as a ``datasets.map`` callback or any plain-dict
pipeline.

**Semantics (mirrors TRL):**

- **Preference example** (has ``"chosen"`` and ``"rejected"`` keys, each a
  ``list``): compute the longest common prefix by walking element-wise with
  ``==`` until the first mismatch, set ``"prompt"`` to that prefix, and
  replace ``"chosen"``/``"rejected"`` with the respective remaining suffixes.
- **Already-extracted** (has ``"prompt"``): return the dict unchanged
  (idempotent).
- **Unpaired example** (has ``"prompt"`` and ``"completion"`` but no
  chosen/rejected): return unchanged.

The element type is intentionally unrestricted — dict messages
``{"role": ..., "content": ...}``, integer token IDs, or any other objects
that support ``==`` equality all work correctly.
"""

from __future__ import annotations

__all__ = ["extract_prompt"]


def extract_prompt(example: dict) -> dict:
    """Extract the shared prompt prefix from a preference example.

    Given a single dataset row, returns a new dict with the same keys plus
    a ``"prompt"`` key holding the longest common prefix of ``"chosen"`` and
    ``"rejected"``.  The values for ``"chosen"`` and ``"rejected"`` are
    replaced by their respective suffixes (the parts after the common prefix).

    If the example already contains a ``"prompt"`` key the dict is returned
    **unchanged** (idempotent).  Unpaired examples (``"prompt"`` +
    ``"completion"``, no chosen/rejected) are also returned unchanged.

    All other keys in *example* are preserved without modification.

    Args:
        example: A single dataset record — a plain :class:`dict`.  For
            preference examples the values of ``"chosen"`` and ``"rejected"``
            must be :class:`list` objects.  Elements are compared with ``==``
            and need not be JSON-serialisable.

    Returns:
        A new :class:`dict` with the same keys as *example*.  For preference
        examples without a pre-existing ``"prompt"``, ``"prompt"``,
        ``"chosen"``, and ``"rejected"`` are rewritten as described above.
        All other cases return the original dict unchanged.

    Examples:
        Conversational preference pair::

            >>> example = {
            ...     "chosen":   [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "good"}],
            ...     "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}],
            ... }
            >>> extract_prompt(example)
            {
                "prompt":   [{"role": "user", "content": "hi"}],
                "chosen":   [{"role": "assistant", "content": "good"}],
                "rejected": [{"role": "assistant", "content": "bad"}],
            }

        Idempotent when ``"prompt"`` is already present::

            >>> extract_prompt({"prompt": [...], "chosen": [...], "rejected": [...]})
            # returned unchanged
    """
    # Idempotency: if "prompt" already exists, return unchanged.
    if "prompt" in example:
        return example

    # Only operate on preference examples (both chosen and rejected present).
    if "chosen" not in example or "rejected" not in example:
        return example

    chosen: list = example["chosen"]
    rejected: list = example["rejected"]

    # Walk element-wise; stop at first mismatch.
    prefix_len = 0
    for c_elem, r_elem in zip(chosen, rejected, strict=False):
        if c_elem == r_elem:
            prefix_len += 1
        else:
            break

    prompt = chosen[:prefix_len]
    chosen_suffix = chosen[prefix_len:]
    rejected_suffix = rejected[prefix_len:]

    return {
        **example,
        "prompt": prompt,
        "chosen": chosen_suffix,
        "rejected": rejected_suffix,
    }
