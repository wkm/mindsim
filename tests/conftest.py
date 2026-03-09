"""Shared pytest configuration for the test suite."""


def pytest_addoption(parser):
    parser.addoption(
        "--update-rom-baseline",
        action="store_true",
        default=False,
        help="Write discovered ROM as new baseline (replaces existing baselines)",
    )
