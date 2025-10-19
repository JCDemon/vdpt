from __future__ import annotations
from typing import Dict
from .base import OperationHandler

_REGISTRY: Dict[str, OperationHandler] = {}

def register(handler: OperationHandler):
    _REGISTRY[handler.kind] = handler

def get_handler(kind: str) -> OperationHandler:
    if kind not in _REGISTRY:
        raise KeyError(f"No handler registered for kind={kind}")
    return _REGISTRY[kind]

def kinds():
    return list(_REGISTRY.keys())
