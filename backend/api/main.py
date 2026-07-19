"""Legacy entrypoint. Use backend.api.app:app instead."""

from .app import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.app:app",
        host="0.0.0.0",  # nosec B104 - container server bind
        port=8000,
        reload=True,
    )
