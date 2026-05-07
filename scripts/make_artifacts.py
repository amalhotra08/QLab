#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_attention.config import ensure_project_dirs
from qlab_attention.plots import make_all_figures
from qlab_attention.reporting import write_all_artifacts


def main() -> None:
    ensure_project_dirs()
    make_all_figures()
    for path in write_all_artifacts():
        print(path)


if __name__ == "__main__":
    main()

