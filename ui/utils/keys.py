import time


def unique_key(prefix: str) -> str:
    # Simple per-call unique key; good for loop-created widgets & downloads
    return f"{prefix}-{time.time_ns()}"
