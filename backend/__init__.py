"""VDPT backend package."""

from .app.main import app as _app


def create_app():
    return _app


app = _app

__all__ = ["create_app", "app"]
