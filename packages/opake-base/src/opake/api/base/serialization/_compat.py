"""Version-tag helpers for serializer evolution.

Reserved for future use: when a registered handler's wire format
changes incompatibly, the new version registers under a tagged key
(e.g. ``"opake.dpsgd.gaussian/2"``) and the loader bridges old tags
forward. Until any handler needs versioning, this module is empty.
"""

from __future__ import annotations

__all__: list[str] = []
