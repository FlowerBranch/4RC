"""Suite-wide isolation from the launch environment.

The trainer reads WORLD_SIZE (and, above one rank, RANK, LOCAL_RANK,
MASTER_ADDR and MASTER_PORT) from the process environment. A suite run inside a
job step that exports them would otherwise turn every argument-validator test
into a multi-rank one, so each test starts with all five unset; a test that
wants a launch sets them itself through ``monkeypatch.setenv``.
"""

import pytest

LAUNCH_VARIABLES = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def _no_launch_environment(monkeypatch):
    for name in LAUNCH_VARIABLES:
        monkeypatch.delenv(name, raising=False)
