"""
Shared pytest fixtures.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from src.api.main import app


@pytest.fixture(scope="session")
def api_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_influxdb(monkeypatch):
    """Mock InfluxDB client to avoid real DB calls in unit tests."""
    mock = AsyncMock()
    monkeypatch.setattr("src.pipeline.ingest.InfluxDBClient", MagicMock(return_value=mock))
    return mock
