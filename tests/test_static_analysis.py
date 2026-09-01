"""The check that would have caught the run_summary UnboundLocalError.

``overfit_temporal_tracking.main()`` shipped reading two names that a ``del``
earlier in the function had already unbound, so every run crashed before writing
``run_summary.json``.  492 tests passed against a guaranteed crash: the three
summary tests in ``test_sparse_tracking.py`` AST-parse the ``summary`` dict
literal and assert key *presence*, which never evaluates the value expressions,
so an unbound name inside the literal is invisible to them.

This is static too, but it is the right static: name resolution rather than key
presence.  ``pyrightconfig.json`` already declares this class fatal
(``"reportUnboundVariable": "error"``) -- pyright is simply never run by the
suite and is not a dev dependency.  pyflakes enforces the same declaration in
CI without adding a node toolchain.

Scoped to the two drivers because that is where the class bites: they are the
only modules with a ``main()`` too GPU-bound for the suite to execute in full,
so an unbound name in one of them has nothing else catching it.  Widening to the
whole repo is cheap if it is ever wanted -- under ``pyrightconfig.json``'s own
exclude set the repo currently has no ``UndefinedName`` hits at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Deliberately a plain import, not pytest.importorskip: pyflakes is listed in
# requirements-dev.txt, and a gate that skips itself when its dependency is
# missing is a gate that reports green on the day it stops running.
from pyflakes import api as pyflakes_api
from pyflakes import messages as pyflakes_messages

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Reading a name that is not bound -- never assigned, or `del`-ed before the
# read, which is what happened here. Only these two classes are asserted on:
# unused imports and placeholder-less f-strings are style, not a crash, and
# folding them in would make the gate too noisy to keep.
_UNBOUND_NAME_MESSAGES = (
    pyflakes_messages.UndefinedName,
    pyflakes_messages.UndefinedLocal,
)

_DRIVERS = ("overfit_temporal_tracking.py", "train_temporal_tracking.py")


class _UnboundNameReporter:
    """A pyflakes reporter that keeps the unbound-name classes and drops the rest."""

    def __init__(self) -> None:
        self.unbound: list[str] = []
        self.failures: list[str] = []

    def unexpectedError(self, filename, msg) -> None:
        self.failures.append(f"{filename}: {msg}")

    def syntaxError(self, filename, msg, lineno, offset, text) -> None:
        self.failures.append(f"{filename}:{lineno}: {msg}")

    def flake(self, message) -> None:
        if isinstance(message, _UNBOUND_NAME_MESSAGES):
            self.unbound.append(str(message))


@pytest.mark.parametrize("driver", _DRIVERS)
def test_the_drivers_never_read_an_unbound_name(driver: str) -> None:
    reporter = _UnboundNameReporter()
    pyflakes_api.checkPath(str(_REPO_ROOT / driver), reporter)

    # A file pyflakes could not parse must fail loudly rather than report clean.
    assert reporter.failures == [], "\n".join(reporter.failures)
    assert reporter.unbound == [], "\n".join(reporter.unbound)
