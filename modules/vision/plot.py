"""Re-render result figures from a results folder (e.g. after a Vast run).

Usage::

    python -m modules.vision.plot results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from modules.vision.evaluation.comparison import render_figures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path, nargs="?", default=Path("results"))
    parser.add_argument("--output-dir", type=Path, help="Defaults to <results_dir>/figures.")
    args = parser.parse_args(argv)
    for path in render_figures(args.results_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
