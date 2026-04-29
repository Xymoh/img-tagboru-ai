from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import bootstrap


def _resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parent


def main() -> None:
    app_path = _resource_root() / "frontend" / "app.py"
    bootstrap.run(str(app_path), False, [], {})


if __name__ == "__main__":
    main()