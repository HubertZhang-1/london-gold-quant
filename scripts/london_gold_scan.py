# -*- coding: utf-8 -*-
"""London gold realtime signal scanner CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.scanner import print_snapshot, scan


def main() -> int:
    parser = argparse.ArgumentParser(description="London gold signal scanner")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "london_gold_config.json")
    parser.add_argument("--update", action="store_true", help="force refresh daily data")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)
    snapshot = scan(config, update_data=args.update)
    print_snapshot(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
