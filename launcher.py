from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from streamlit.web import bootstrap


def _resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parent


def _start_backend_in_thread(host: str = "127.0.0.1", port: int = 8000) -> None:
    import socket

    # If something is already listening on the port, don't try to start another server.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex((host, port))
        if result == 0:
            print(f"Backend appears to be listening on {host}:{port}, skipping startup.")
            return

    def _run():
        try:
            import uvicorn

            uvicorn.run("backend.api:app", host=host, port=port, log_level="info")
        except Exception as exc:  # pragma: no cover - runtime error reporting
            print("Failed to start backend uvicorn server:", exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def main() -> None:
    # Start the local FastAPI backend (so the packaged exe serves both frontend and API)
    _start_backend_in_thread()
    # Give the backend a short moment to start before launching Streamlit
    time.sleep(1.0)

    app_path = _resource_root() / "frontend" / "app.py"
    bootstrap.run(str(app_path), False, [], {})


if __name__ == "__main__":
    main()