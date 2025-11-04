import time


# keep the time-based helper for truly ephemeral widgets in dynamic lists
def unique_key(prefix: str) -> str:
    return f"{prefix}-{time.time_ns()}"


# NEW: deterministic, rerun-stable keys for stateful controls (tree toggles, selectors)
def stable_key(prefix: str, *parts: object) -> str:
    suffix = "-".join(str(p) for p in parts)
    return f"{prefix}-{suffix}" if suffix else prefix


# OPTIONAL: make the reviewer’s name resolve to the stable one to avoid NameError
def _unique_widget_key(prefix: str, *parts: object) -> str:  # alias for safety
    return stable_key(prefix, *parts)
