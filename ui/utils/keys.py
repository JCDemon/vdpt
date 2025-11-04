import time
from typing import Any


# keep the time-based helper for truly ephemeral widgets in dynamic lists
def unique_key(prefix: str) -> str:
    return f"{prefix}-{time.time_ns()}"


def stable_key(prefix: str, *parts: Any) -> str:
    suffix = "-".join(str(p) for p in parts)
    return f"{prefix}-{suffix}" if suffix else prefix


def stable_node_key(section_id: str, node: dict, idx: int) -> str:
    """
    Deterministic key for tree nodes. Prefer an explicit id; otherwise
    fall back to other stable fields; finally fall back to the loop index.
    """

    for field in ("id", "path", "name", "label", "title"):
        v = node.get(field)
        if v not in (None, ""):
            return stable_key("tree-node", section_id, v)

    return stable_key("tree-node", section_id, idx)


def _unique_widget_key(prefix: str, *parts: Any) -> str:  # noqa: N802
    """Use deterministic keys for stateful controls, not time-based ones."""

    return stable_key(prefix, *parts)
