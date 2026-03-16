pytest_plugins = [
    "tests.fixtures.conftest",
    "tests.fixtures.polygon.conftest",
]

import os
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "db_required: requires DATABASE_URL environment variable"
    )


@pytest.fixture(autouse=True)
def skip_if_no_db(request):
    if request.node.get_closest_marker("db_required"):
        if not os.environ.get("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set")
