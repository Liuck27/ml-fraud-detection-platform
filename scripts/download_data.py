"""Download Kaggle credit card fraud dataset to data/raw/creditcard.csv.

Usage:
    make download-data            # recommended
    python scripts/download_data.py

Requires KAGGLE_USERNAME and KAGGLE_KEY in .env (or already in environment).
Get your API key at https://www.kaggle.com/settings/account.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_env(env_path: Path) -> None:
    """Parse key=value lines from .env and inject missing keys into os.environ."""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def main() -> None:
    # Load .env from project root (two levels up from scripts/)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        _load_env(env_path)

    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")

    if not username or not key or username == "your_kaggle_username":
        print(
            "ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env\n"
            "  1. Go to https://www.kaggle.com/settings/account\n"
            "  2. Click 'Create New Token' — downloads kaggle.json\n"
            "  3. Copy username and key into .env",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set env vars so the kaggle library picks them up
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    try:
        import kaggle  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: kaggle package not installed. Run: make venv && pip install kaggle",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(__file__).parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "creditcard.csv"

    if csv_path.exists():
        size = csv_path.stat().st_size
        print(
            f"Dataset already exists at {csv_path} ({size / 1_048_576:.1f} MB) — skipping download."
        )
        return

    print("Downloading mlg-ulb/creditcardfraud from Kaggle (~66 MB)...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "mlg-ulb/creditcardfraud",
        path=str(out_dir),
        unzip=True,
        quiet=False,
    )

    if not csv_path.exists():
        print(f"ERROR: Download finished but {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Quick sanity check
    with open(csv_path) as f:
        row_count = sum(1 for _ in f) - 1  # subtract header

    print(f"Done. {csv_path} — {row_count:,} rows.")
    if row_count < 280_000:
        print(f"WARNING: expected ~284,807 rows, got {row_count:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
