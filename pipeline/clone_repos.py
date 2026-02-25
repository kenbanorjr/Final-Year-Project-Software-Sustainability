"""
Clone repositories listed in configs/config.py.
"""

from __future__ import annotations

from pipeline import miner
from pipeline.configs import config


def clone_all_repos() -> None:
    miner.clone_repositories(config.ALL_REPOSITORIES)


def main() -> None:
    clone_all_repos()


if __name__ == "__main__":
    main()
