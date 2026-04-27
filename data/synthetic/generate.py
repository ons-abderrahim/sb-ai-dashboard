"""
Synthetic Sensor Data Generator
================================
Generates realistic multi-zone IoT sensor time-series data for development,
testing, and demos. Simulates:
  - Realistic occupancy patterns (weekday/weekend, business hours)
  - CO₂ build-up correlated with occupancy
  - Temperature drift with HVAC response
  - Random anomalies (~2% of readings)

Usage:
    python data/synthetic/generate.py --zones 20 --days 90 --output data/processed/
    python data/synthetic/generate.py --zones 5 --days 7 --seed 42  # reproducible
"""

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Zone templates with distinct occupancy profiles
ZONE_PROFILES = {
    "open_office": {"peak_occ": 40, "start_h": 8, "end_h": 18, "weekend_factor": 0.05},
    "meeting_room": {"peak_occ": 12, "start_h": 9, "end_h": 17, "weekend_factor": 0.0},
    "lab": {"peak_occ": 8, "start_h": 7, "end_h": 22, "weekend_factor": 0.4},
    "cafeteria": {"peak_occ": 80, "start_h": 7, "end_h": 15, "weekend_factor": 0.1},
    "server_room": {"peak_occ": 1, "start_h": 0, "end_h": 24, "weekend_factor": 1.0},
}


def occupancy_at(hour: float, day_of_week: int, profile: dict) -> float:
    """
    Model occupancy as a smooth bell curve centered on business hours.
    Returns fractional occupancy in [0, 1].
    """
    is_weekend = day_of_week >= 5
    base = profile["weekend_factor"] if is_weekend else 1.0

    center = (profile["start_h"] + profile["end_h"]) / 2
    width = (profile["end_h"] - profile["start_h"]) / 2

    if width == 0:
        return 0.0

    gaussian = np.exp(-0.5 * ((hour - center) / (width * 0.6)) ** 2)
    return float(base * gaussian)


def generate_zone(
    zone_id: str,
    zone_type: str,
    start_dt: datetime,
    n_days: int,
    freq_minutes: int = 5,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic sensor readings for a single zone."""
    if rng is None:
        rng = np.random.default_rng()

    profile = ZONE_PROFILES[zone_type]
    timestamps = pd.date_range(start=start_dt, periods=n_days * 24 * 60 // freq_minutes,
                               freq=f"{freq_minutes}min", tz="UTC")

    records = []
    co2_base = 420.0  # Outdoor CO₂ baseline (ppm)
    temp_setpoint = 22.0

    for ts in timestamps:
        hour = ts.hour + ts.minute / 60.0
        dow = ts.dayofweek

        occ_frac = occupancy_at(hour, dow, profile)
        occ_count = int(occ_frac * profile["peak_occ"] * (1 + rng.normal(0, 0.1)))
        occ_count = max(0, occ_count)

        # CO₂ rises with occupancy (each person exhales ~0.2 L/min CO₂)
        co2 = co2_base + occ_count * 18 + rng.normal(0, 12)
        co2 = float(np.clip(co2, 380, 2500))

        # Temperature: HVAC keeps near setpoint, rises with occupancy
        temp_offset = occ_frac * 1.5
        temperature = temp_setpoint + temp_offset + rng.normal(0, 0.3)

        # Humidity: slightly elevated with occupancy
        humidity = 45 + occ_frac * 8 + rng.normal(0, 2)
        humidity = float(np.clip(humidity, 20, 80))

        # Motion events (5-min window)
        motion_events = int(occ_count * 0.3 * rng.poisson(1)) if occ_count > 0 else 0

        # Light level (lux) — correlated with hours & occupancy
        lux_base = 300 if (8 <= hour <= 18) else 5
        lux = max(0, lux_base * (0.7 + occ_frac * 0.3) + rng.normal(0, 20))

        # Anomaly injection (~2% of readings)
        is_anomaly = rng.random() < 0.02
        if is_anomaly:
            anomaly_type = rng.choice(["co2_spike", "temp_spike", "motion_ghost"])
            if anomaly_type == "co2_spike":
                co2 *= rng.uniform(1.3, 1.8)
            elif anomaly_type == "temp_spike":
                temperature += rng.uniform(3, 8)
            else:  # motion_ghost
                motion_events += int(rng.uniform(5, 15))
        else:
            anomaly_type = None

        records.append({
            "timestamp": ts,
            "zone_id": zone_id,
            "zone_type": zone_type,
            "occupancy_count": occ_count,
            "co2_ppm": round(co2, 1),
            "temperature_c": round(float(temperature), 2),
            "humidity_pct": round(humidity, 1),
            "motion_events": motion_events,
            "lux": round(float(lux), 1),
            "noise_db": round(35 + occ_frac * 20 + rng.normal(0, 2), 1),
            "pir_count": motion_events,  # Simplified
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
        })

    return pd.DataFrame(records).set_index("timestamp")


def generate_building(
    n_zones: int = 20,
    n_days: int = 90,
    seed: int | None = None,
    output_dir: Path = Path("data/processed"),
) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    start_dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    zone_types = list(ZONE_PROFILES.keys())

    metadata = []
    all_dfs = []

    for i in range(n_zones):
        zone_type = zone_types[i % len(zone_types)]
        zone_id = f"{zone_type}_{i+1:03d}"
        floor = (i // 5) + 1

        print(f"  Generating zone {i+1}/{n_zones}: {zone_id} ...")
        df = generate_zone(zone_id, zone_type, start_dt, n_days, rng=rng)
        all_dfs.append(df)

        metadata.append({
            "zone_id": zone_id,
            "zone_type": zone_type,
            "floor": floor,
            "area_m2": int(rng.integers(20, 200)),
            "brick_class": f"https://brickschema.org/schema/Brick#{zone_type.title().replace('_', '')}",
        })

    # Save
    combined = pd.concat(all_dfs)
    combined.reset_index().to_parquet(output_dir / "sensor_data.parquet", index=False)

    with open(output_dir / "zone_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Generated {n_zones} zones × {n_days} days")
    print(f"   Rows: {len(combined):,}")
    print(f"   Output: {output_dir}/")
    print(f"   Anomaly rate: {combined['is_anomaly'].mean():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic building sensor data")
    parser.add_argument("--zones", type=int, default=20, help="Number of zones")
    parser.add_argument("--days", type=int, default=90, help="Number of days")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    print(f"🏢 Generating synthetic building data...")
    generate_building(
        n_zones=args.zones,
        n_days=args.days,
        seed=args.seed,
        output_dir=Path(args.output),
    )
