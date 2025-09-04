# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import ArgumentTypeError
from pathlib import Path
from typing import cast

from pytest import Config, Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--integration",
        default=False,
        action="store_true",
        help="whether to run the integration tests",
    )


def pytest_ignore_collect(collection_path: Path, config: Config) -> bool:
    # Ignore integration tests unless we run `pytest --integration`.
    if "integration" in collection_path.parts:
        return not cast(bool, config.getoption("integration"))

    return False
