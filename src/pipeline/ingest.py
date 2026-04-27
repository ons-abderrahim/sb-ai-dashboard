"""
Data Ingestion Pipeline
=======================
Connects to InfluxDB for time-series sensor reads and TimescaleDB for
building metadata. Supports both batch and streaming (async generator) modes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import pandas as pd
import structlog
from influxdb_client import InfluxDBClient
from influxdb_client.client.flux_table import FluxTable

from src.utils.config import settings

logger = structlog.get_logger(__name__)

# Sensor field names as stored in InfluxDB
SENSOR_FIELDS = ["co2_ppm", "temperature_c", "humidity_pct", "motion_events",
                  "door_open_count", "lux", "noise_db", "pir_count"]


class SensorDataIngester:
    """
    Fetches sensor time-series data from InfluxDB.

    Usage:
        async with SensorDataIngester() as ingester:
            df = await ingester.fetch_zone_history("floor_3_open_office", hours=24)
    """

    def __init__(self) -> None:
        self._client: InfluxDBClient | None = None

    async def __aenter__(self) -> "SensorDataIngester":
        self._client = InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            self._client.close()

    # ── Query helpers ──────────────────────────────────────────────────────────

    def _zone_query(self, zone_id: str, hours: int, fields: list[str]) -> str:
        field_filter = " or ".join(f'r["_field"] == "{f}"' for f in fields)
        return f"""
        from(bucket: "{settings.influxdb_bucket}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "sensor_reading")
          |> filter(fn: (r) => r["zone_id"] == "{zone_id}")
          |> filter(fn: (r) => {field_filter})
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        """

    # ── Public API ─────────────────────────────────────────────────────────────

    async def fetch_zone_history(
        self,
        zone_id: str,
        hours: int = 24,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical sensor data for a zone as a pandas DataFrame.

        Args:
            zone_id: Zone identifier (e.g. "floor_3_open_office")
            hours:   Lookback window in hours
            fields:  Sensor fields to fetch; defaults to all SENSOR_FIELDS

        Returns:
            DataFrame with DatetimeIndex and one column per sensor field.
            Missing values are forward-filled then back-filled.
        """
        fields = fields or SENSOR_FIELDS
        query = self._zone_query(zone_id, hours, fields)

        assert self._client is not None, "Use as async context manager"
        query_api = self._client.query_api()

        logger.info("Fetching zone history", zone_id=zone_id, hours=hours)
        tables: list[FluxTable] = query_api.query(query)

        if not tables:
            logger.warning("No data returned", zone_id=zone_id)
            return pd.DataFrame(columns=["timestamp"] + fields)

        rows = []
        for table in tables:
            for record in table.records:
                row = {"timestamp": record.get_time()}
                row.update({f: record.values.get(f) for f in fields})
                rows.append(row)

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df.ffill().bfill()

        logger.info("Fetched sensor data", zone_id=zone_id, rows=len(df))
        return df

    async def fetch_latest(self, zone_id: str) -> dict[str, float]:
        """Fetch the most recent reading for each sensor field in a zone."""
        df = await self.fetch_zone_history(zone_id, hours=1)
        if df.empty:
            return {}
        return df.iloc[-1].to_dict()

    async def stream_zone(
        self,
        zone_id: str,
        interval_seconds: int = 30,
        fields: list[str] | None = None,
    ) -> AsyncGenerator[dict[str, float], None]:
        """
        Async generator that yields the latest sensor snapshot every `interval_seconds`.

        Usage:
            async for snapshot in ingester.stream_zone("floor_3"):
                process(snapshot)
        """
        fields = fields or SENSOR_FIELDS
        while True:
            try:
                snapshot = await self.fetch_latest(zone_id)
                if snapshot:
                    yield snapshot
            except Exception as exc:
                logger.error("Stream error", zone_id=zone_id, error=str(exc))
            await asyncio.sleep(interval_seconds)
