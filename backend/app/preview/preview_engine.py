def preview_operation(op: dict):
    kind = op.get("kind")
    if kind == "segment":
        # TODO: replace with real SAM-like stats
        return {"overlay": "mask@alpha0.5", "affected_pixels": 12345}
    if kind == "filter":
        # TODO: replace with real table diff
        return {"kept_rows": 120, "removed_rows": 30}
    return {"note": "stub"}
