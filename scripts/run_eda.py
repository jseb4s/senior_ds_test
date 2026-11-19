#!/usr/bin/env python
"""
EDA script — run before training to understand the data.

Usage:
    python scripts/run_eda.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.loader import load_config, load_modeling_dataset
from src.eda.analysis import run_eda

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    config_path = Path(__file__).parents[1] / "config" / "config.yaml"
    config = load_config(config_path)

    db_path = Path(__file__).parents[1] / config["data"]["db_path"]
    logger.info("Loading data from: %s", db_path)

    df = load_modeling_dataset(
        db_path=db_path,
        modeling_population=None,  # Load all statuses for EDA (including Joined)
    )

    # For EDA we include all statuses to understand the full dataset
    logger.info("Full dataset loaded for EDA: %d rows", len(df))
    run_eda(df)


if __name__ == "__main__":
    main()
