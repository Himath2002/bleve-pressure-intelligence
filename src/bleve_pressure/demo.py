"""Deterministic synthetic BLEVE-like records for exercising the public workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_scenarios(rows: int, *, seed: int, first_id: int) -> pd.DataFrame:
    """Create plausible-looking, explicitly synthetic BLEVE scenario records."""

    rng = np.random.default_rng(seed)
    tank_width = rng.uniform(0.5, 3.0, rows)
    tank_length = rng.uniform(0.8, 10.0, rows)
    tank_height = rng.uniform(0.5, 3.0, rows)
    bleve_height = rng.uniform(0.0, 2.0, rows)
    vapour_height = rng.uniform(0.2, 0.9, rows) * tank_height
    failure_pressure = rng.uniform(5.0, 45.0, rows)
    liquid_ratio = rng.uniform(0.1, 0.9, rows)
    obstacle_distance = rng.uniform(5.0, 20.0, rows)
    obstacle_width = rng.uniform(3.0, 18.0, rows)
    obstacle_height = rng.uniform(3.0, 18.0, rows)
    obstacle_thickness = rng.uniform(0.4, 3.0, rows)
    obstacle_angle = rng.choice([0.0, 15.0, 30.0], rows)
    sensor_id = rng.integers(1, 28, rows)
    sensor_side = rng.integers(1, 6, rows)
    sensor_x = rng.uniform(5.0, 24.0, rows)
    sensor_y = rng.uniform(-9.0, 20.0, rows)
    sensor_z = rng.uniform(-2.5, 16.5, rows)
    status = np.where(rng.random(rows) < 0.38, "Superheated", "Subcooled")
    critical_pressure = rng.choice([37.9, 42.5], rows)
    boiling_temperature = rng.choice([-42.0, -1.0], rows)
    critical_temperature = rng.choice([96.7, 152.0], rows)
    liquid_temperature = rng.uniform(283.0, 425.0, rows)
    vapour_temperature = np.maximum(liquid_temperature, rng.uniform(300.0, 573.0, rows))

    return pd.DataFrame(
        {
            "ID": np.arange(first_id, first_id + rows),
            "Tank Failure Pressure (bar)": failure_pressure,
            "Liquid Ratio (%)": liquid_ratio,
            "Tank Width (m)": tank_width,
            "Tank Length (m)": tank_length,
            "Tank Height (m)": tank_height,
            "BLEVE Height (m)": bleve_height,
            "Vapour Height (m)": vapour_height,
            "Vapour Temperature (K)": vapour_temperature,
            "Liquid Temperature (K)": liquid_temperature,
            "Obstacle Distance to BLEVE (m)": obstacle_distance,
            "Obstacle Width (m)": obstacle_width,
            "Obstacle Height (m)": obstacle_height,
            "Obstacle Thickness (m)": obstacle_thickness,
            "Obstacle Angle": obstacle_angle,
            "Status": status,
            "Liquid Critical Pressure (bar)": critical_pressure,
            "Liquid Boiling Temperature (K)": boiling_temperature,
            "Liquid Critical Temperature (K)": critical_temperature,
            "Sensor ID": sensor_id,
            "Sensor Position Side": sensor_side,
            "Sensor Position x": sensor_x,
            "Sensor Position y": sensor_y,
            "Sensor Position z": sensor_z,
        }
    )


def synthetic_target(frame: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Return a positive proxy target for software demonstrations only."""

    rng = np.random.default_rng(seed)
    distance = np.sqrt(
        frame["Sensor Position x"] ** 2
        + frame["Sensor Position y"] ** 2
        + (frame["Sensor Position z"] - frame["BLEVE Height (m)"]) ** 2
    )
    obstacle_area = frame["Obstacle Width (m)"] * frame["Obstacle Height (m)"]
    angle_factor = np.abs(np.cos(np.deg2rad(frame["Obstacle Angle"])))
    shielding = 1.0 / (
        1.0 + 0.0025 * obstacle_area * angle_factor / frame["Obstacle Distance to BLEVE (m)"]
    )
    thermal = 1.0 + 0.0012 * (
        frame["Liquid Temperature (K)"] - frame["Liquid Temperature (K)"].median()
    )
    state = np.where(frame["Status"].eq("Superheated"), 1.12, 0.94)
    noise = rng.lognormal(mean=0.0, sigma=0.07, size=len(frame))
    pressure = (
        0.82
        * frame["Tank Failure Pressure (bar)"]
        * (0.45 + frame["Liquid Ratio (%)"])
        * thermal
        * state
        * shielding
        * noise
        / np.power(1.0 + distance, 1.35)
    )
    return np.clip(pressure.to_numpy(dtype=float), 0.01, None)
