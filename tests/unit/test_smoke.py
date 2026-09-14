"""Proves the scaffold itself works: package installs and imports cleanly."""

import dispatch


def test_package_imports() -> None:
    assert dispatch is not None
