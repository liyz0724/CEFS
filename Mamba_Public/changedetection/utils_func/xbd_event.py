"""Utilities for parsing xBD sample names."""

import re


_EVENT_PATTERN = re.compile(r"^(?P<event>.+?)_(?P<tile>[0-9]{6,})(?:_|$)")


def extract_xbd_event(item_name: str) -> str:
    """Extract an xBD event name from a ChangeMamba list item."""

    normalized = str(item_name).rsplit("/", 1)[-1]
    if normalized.endswith(".png"):
        normalized = normalized[:-4]
    for marker in ("_pre_disaster", "_post_disaster"):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[0]

    match = _EVENT_PATTERN.match(normalized)
    if match is None:
        raise ValueError(
            f"Cannot parse xBD event from {item_name!r}; expected "
            "EVENT_######## or EVENT_########_ROW_COL."
        )
    return match.group("event").lower()
