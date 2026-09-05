"""Physics-guided feature construction for BLEVE simulation records."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalise_status_value(value: object) -> str:
    """Collapse known casing and spelling variants into stable state labels."""

    text = str(value).strip().lower()
    if text in {"", "nan", "none", "<na>"}:
        return "missing"
    if "super" in text or "saper" in text or text.startswith("sap"):
        return "Superheated"
    if "sub" in text:
        return "Subcooled"
    return str(value).strip()


def _clean_text_series(series: pd.Series) -> pd.Series:
    values = series.fillna("missing").astype(str)
    return values.replace({"nan": "missing", "None": "missing", "<NA>": "missing"})


def _temperature_to_kelvin(values: pd.Series, *, declared_unit: str = "auto") -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    finite_values = values[np.isfinite(values)]
    if finite_values.empty or declared_unit == "K":
        return values
    if declared_unit == "C":
        return values + 273.15

    median_value = float(finite_values.median())
    minimum_value = float(finite_values.min())
    return values + 273.15 if minimum_value < 0 or median_value < 200 else values


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable geometry, thermodynamic, obstacle, and sensor features.

    Missing source columns are represented as ``NaN`` so the same function can
    safely align compatible simulation exports before model preprocessing.
    The input frame is never mutated.
    """

    data = frame.copy()
    epsilon = 1e-6

    if "Status" in data.columns:
        data["Status"] = data["Status"].map(normalise_status_value)

    def numeric(column: str) -> pd.Series:
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce")
        return pd.Series(np.nan, index=data.index, dtype=float)

    def first_numeric(*columns: str) -> pd.Series:
        for column in columns:
            if column in data.columns:
                return pd.to_numeric(data[column], errors="coerce")
        return pd.Series(np.nan, index=data.index, dtype=float)

    liquid_ratio = first_numeric("Liquid Ratio (%)", "Liquid Ratio", "Liquid ratio")
    liquid_fraction = pd.Series(
        np.where(liquid_ratio <= 1.5, liquid_ratio, liquid_ratio / 100.0),
        index=data.index,
    ).clip(lower=0, upper=1)

    data["FE_liquid_ratio_raw"] = liquid_ratio
    data["FE_liquid_ratio_fraction"] = liquid_fraction
    data["FE_liquid_ratio_percent_scale"] = liquid_ratio / 100.0
    data["FE_vapour_ratio_fraction"] = 1.0 - liquid_fraction

    tank_length = numeric("Tank Length (m)")
    tank_width = numeric("Tank Width (m)")
    tank_height = numeric("Tank Height (m)")
    vapour_height = numeric("Vapour Height (m)")
    bleve_height = numeric("BLEVE Height (m)")

    tank_volume = tank_length * tank_width * tank_height
    tank_footprint = tank_length * tank_width
    tank_surface = 2.0 * (
        tank_length * tank_width + tank_length * tank_height + tank_width * tank_height
    )
    vapour_volume = tank_length * tank_width * vapour_height
    liquid_height_proxy = np.maximum(tank_height - vapour_height, 0)
    liquid_volume_height_proxy = tank_length * tank_width * liquid_height_proxy
    liquid_volume_ratio_proxy = tank_volume * liquid_fraction
    vapour_volume_ratio_proxy = tank_volume * (1.0 - liquid_fraction)

    data["FE_tank_volume_box"] = tank_volume
    data["FE_tank_footprint"] = tank_footprint
    data["FE_tank_surface_proxy"] = tank_surface
    data["FE_tank_surface_to_volume"] = tank_surface / (tank_volume + epsilon)
    data["FE_tank_length_width_ratio"] = tank_length / (tank_width + epsilon)
    data["FE_tank_height_width_ratio"] = tank_height / (tank_width + epsilon)
    data["FE_tank_length_height_ratio"] = tank_length / (tank_height + epsilon)
    data["FE_tank_aspect_max_min"] = np.maximum.reduce(
        [tank_length, tank_width, tank_height]
    ) / (np.minimum.reduce([tank_length, tank_width, tank_height]) + epsilon)
    data["FE_vapour_height_ratio"] = vapour_height / (tank_height + epsilon)
    data["FE_liquid_height_proxy"] = liquid_height_proxy
    data["FE_liquid_height_ratio"] = liquid_height_proxy / (tank_height + epsilon)
    data["FE_vapour_volume_proxy"] = vapour_volume
    data["FE_liquid_volume_proxy"] = liquid_volume_height_proxy
    data["FE_liquid_volume_ratio_proxy"] = liquid_volume_ratio_proxy
    data["FE_vapour_volume_ratio_proxy"] = vapour_volume_ratio_proxy
    data["FE_bleve_height_ratio"] = bleve_height / (tank_height + epsilon)
    data["FE_bleve_above_tank_midheight"] = bleve_height - 0.5 * tank_height

    sensor_x = numeric("Sensor Position x")
    sensor_y = numeric("Sensor Position y")
    sensor_z = numeric("Sensor Position z")
    sensor_planar = np.sqrt(sensor_x**2 + sensor_y**2)
    sensor_vertical_offset = sensor_z - bleve_height
    sensor_distance = np.sqrt(sensor_x**2 + sensor_y**2 + sensor_vertical_offset**2)
    sensor_origin_distance = np.sqrt(sensor_x**2 + sensor_y**2 + sensor_z**2)

    data["FE_sensor_planar_distance"] = sensor_planar
    data["FE_sensor_vertical_offset"] = sensor_vertical_offset
    data["FE_sensor_abs_vertical_offset"] = np.abs(sensor_vertical_offset)
    data["FE_sensor_3d_distance_bleve"] = sensor_distance
    data["FE_sensor_3d_distance_origin"] = sensor_origin_distance
    data["FE_log_sensor_3d_distance"] = np.log1p(sensor_distance)
    data["FE_inv_sensor_distance"] = 1.0 / (1.0 + sensor_distance)
    data["FE_inv_sensor_distance_sq"] = 1.0 / ((1.0 + sensor_distance) ** 2)
    data["FE_inv_sensor_distance_cu"] = 1.0 / ((1.0 + sensor_distance) ** 3)
    data["FE_abs_sensor_x"] = np.abs(sensor_x)
    data["FE_abs_sensor_y"] = np.abs(sensor_y)
    data["FE_abs_sensor_z"] = np.abs(sensor_z)
    data["FE_sensor_xy_ratio"] = sensor_x / (np.abs(sensor_y) + epsilon)
    data["FE_sensor_z_over_distance"] = sensor_z / (sensor_distance + epsilon)
    data["FE_sensor_planar_over_height"] = sensor_planar / (
        np.abs(sensor_vertical_offset) + 1.0
    )
    data["FE_sensor_height_minus_tank_midheight"] = sensor_z - 0.5 * tank_height

    if "Sensor ID" in data.columns:
        sensor_id = pd.to_numeric(data["Sensor ID"], errors="coerce")
        data["FE_sensor_id_num"] = sensor_id
        data["FE_sensor_id_mod3"] = np.mod(sensor_id.fillna(-1), 3)
        data["FE_sensor_id_mod9"] = np.mod(sensor_id.fillna(-1), 9)
        data["FE_sensor_id_group9"] = np.floor(sensor_id.fillna(-1) / 9.0)
        data["FE_sensor_id_center_distance"] = np.abs(sensor_id - sensor_id.median())

    failure_pressure = numeric("Tank Failure Pressure (bar)")
    data["FE_log_failure_pressure"] = np.log1p(np.maximum(failure_pressure, 0))
    data["FE_pressure_over_distance"] = failure_pressure / (1.0 + sensor_distance)
    data["FE_pressure_over_distance_sq"] = failure_pressure / ((1.0 + sensor_distance) ** 2)
    data["FE_log_pressure_over_distance"] = np.log1p(
        np.maximum(failure_pressure, 0)
    ) - np.log1p(sensor_distance)
    data["FE_pressure_x_inv_distance"] = failure_pressure * data["FE_inv_sensor_distance"]
    data["FE_pressure_x_inv_distance_sq"] = (
        failure_pressure * data["FE_inv_sensor_distance_sq"]
    )
    data["FE_pressure_x_inv_distance_cu"] = (
        failure_pressure * data["FE_inv_sensor_distance_cu"]
    )
    data["FE_pressure_x_liquid_ratio"] = failure_pressure * liquid_ratio
    data["FE_pressure_x_liquid_fraction"] = failure_pressure * liquid_fraction
    data["FE_pressure_over_liquid_ratio"] = failure_pressure / (1.0 + liquid_ratio)
    data["FE_pressure_x_tank_volume"] = failure_pressure * tank_volume
    data["FE_pressure_x_tank_surface"] = failure_pressure * tank_surface
    data["FE_pressure_x_liquid_volume"] = failure_pressure * liquid_volume_ratio_proxy
    data["FE_pressure_x_vapour_volume"] = failure_pressure * vapour_volume_ratio_proxy
    data["FE_log_pressure_x_volume"] = np.log1p(
        np.maximum(failure_pressure * tank_volume, 0)
    )
    data["FE_log_pressure_x_vapour_volume"] = np.log1p(
        np.maximum(failure_pressure * vapour_volume_ratio_proxy, 0)
    )

    obstacle_width = numeric("Obstacle Width (m)")
    obstacle_height = numeric("Obstacle Height (m)")
    obstacle_thickness = numeric("Obstacle Thickness (m)")
    obstacle_distance = numeric("Obstacle Distance to BLEVE (m)")
    obstacle_area = obstacle_width * obstacle_height
    obstacle_volume = obstacle_area * obstacle_thickness

    data["FE_obstacle_area"] = obstacle_area
    data["FE_obstacle_volume_proxy"] = obstacle_volume
    data["FE_obstacle_width_height_ratio"] = obstacle_width / (obstacle_height + epsilon)
    data["FE_obstacle_thickness_height_ratio"] = obstacle_thickness / (
        obstacle_height + epsilon
    )
    data["FE_obstacle_thickness_width_ratio"] = obstacle_thickness / (
        obstacle_width + epsilon
    )
    data["FE_obstacle_area_over_distance"] = obstacle_area / (1.0 + obstacle_distance)
    data["FE_obstacle_area_over_distance_sq"] = obstacle_area / (
        (1.0 + obstacle_distance) ** 2
    )
    data["FE_obstacle_volume_over_distance"] = obstacle_volume / (1.0 + obstacle_distance)
    data["FE_sensor_distance_minus_obstacle_distance"] = sensor_distance - obstacle_distance
    data["FE_obstacle_between_sensor_proxy"] = (obstacle_distance < sensor_distance).astype(float)
    data["FE_obstacle_height_over_sensor_z"] = obstacle_height / (np.abs(sensor_z) + 1.0)
    data["FE_sensor_height_minus_obstacle_midheight"] = sensor_z - 0.5 * obstacle_height
    data["FE_sensor_above_obstacle_midheight_flag"] = (
        sensor_z > 0.5 * obstacle_height
    ).astype(float)

    data["FE_sensor_to_obstacle_center_distance"] = np.sqrt(
        (sensor_x - obstacle_distance) ** 2
        + sensor_y**2
        + (sensor_z - 0.5 * obstacle_height) ** 2
    )
    data["FE_inv_sensor_to_obstacle_center"] = 1.0 / (
        1.0 + data["FE_sensor_to_obstacle_center_distance"]
    )
    data["FE_pressure_over_sensor_to_obstacle"] = failure_pressure / (
        1.0 + data["FE_sensor_to_obstacle_center_distance"]
    )
    data["FE_obstacle_shadow_ratio"] = obstacle_area / (1.0 + sensor_planar**2)

    obstacle_angle = numeric("Obstacle Angle")
    obstacle_angle_radians = np.deg2rad(obstacle_angle)
    sine = np.sin(obstacle_angle_radians)
    cosine = np.cos(obstacle_angle_radians)
    absolute_sine = np.abs(sine)
    absolute_cosine = np.abs(cosine)

    data["FE_obstacle_angle_sin"] = sine
    data["FE_obstacle_angle_cos"] = cosine
    data["FE_obstacle_angle_abs_sin"] = absolute_sine
    data["FE_obstacle_angle_abs_cos"] = absolute_cosine
    data["FE_obstacle_projected_area_sin"] = obstacle_area * absolute_sine
    data["FE_obstacle_projected_area_cos"] = obstacle_area * absolute_cosine
    data["FE_pressure_x_obstacle_cos"] = failure_pressure * absolute_cosine
    data["FE_pressure_x_projected_obstacle_area"] = (
        failure_pressure * obstacle_area * absolute_cosine
    )
    data["FE_shielding_strength_proxy"] = (
        obstacle_area * absolute_cosine / (1.0 + obstacle_distance)
    )
    data["FE_shielded_pressure_distance_proxy"] = (
        failure_pressure * data["FE_shielding_strength_proxy"] / (1.0 + sensor_distance)
    )

    def temperature_first_available(column_specs: list[tuple[str, str]]) -> pd.Series:
        for column, unit in column_specs:
            if column in data.columns:
                return _temperature_to_kelvin(data[column], declared_unit=unit)
        return pd.Series(np.nan, index=data.index, dtype=float)

    critical_temperature = temperature_first_available(
        [
            ("Liquid Critical Temperature (K)", "auto"),
            ("Liquid Critical Temperature", "auto"),
            ("Substance Critical Temperature (K)", "auto"),
            ("Substance Critical Temperature", "C"),
            ("Substance Critical Temperature (C)", "C"),
            ("Critical Temperature (K)", "auto"),
            ("Critical Temperature", "C"),
            ("Critical Temperature (C)", "C"),
        ]
    )
    liquid_temperature = temperature_first_available(
        [("Liquid Temperature (K)", "K"), ("Liquid Temperature", "K")]
    )
    vapour_temperature = temperature_first_available(
        [("Vapour Temperature (K)", "K"), ("Vapour Temperature", "K")]
    )
    boiling_temperature = temperature_first_available(
        [
            ("Liquid Boiling Temperature (K)", "auto"),
            ("Liquid Boiling Temperature", "auto"),
            ("Substance Boiling Temperature (K)", "auto"),
            ("Substance Boiling Temperature", "C"),
            ("Substance Boiling Temperature (C)", "C"),
            ("Boiling Temperature (K)", "auto"),
            ("Boiling Temperature", "C"),
            ("Boiling Temperature (C)", "C"),
        ]
    )
    critical_pressure = first_numeric(
        "Liquid Critical Pressure (bar)",
        "Substance Critical Pressure (bar)",
        "Substance Critical Pressure",
        "Critical Pressure (bar)",
    )

    data["FE_critical_minus_liquid_temp"] = critical_temperature - liquid_temperature
    data["FE_critical_minus_vapour_temp"] = critical_temperature - vapour_temperature
    data["FE_vapour_minus_liquid_temp"] = vapour_temperature - liquid_temperature
    data["FE_liquid_superheat_over_boiling"] = liquid_temperature - boiling_temperature
    data["FE_vapour_superheat_over_boiling"] = vapour_temperature - boiling_temperature
    data["FE_liquid_temp_to_critical"] = liquid_temperature / (
        critical_temperature + epsilon
    )
    data["FE_vapour_temp_to_critical"] = vapour_temperature / (
        critical_temperature + epsilon
    )
    data["FE_failure_to_critical_pressure"] = failure_pressure / (
        critical_pressure + epsilon
    )
    data["FE_log_failure_to_critical_pressure"] = np.log1p(
        np.maximum(failure_pressure, 0)
    ) - np.log1p(np.maximum(critical_pressure, 0))
    data["FE_thermal_gap_combined"] = (
        critical_temperature - liquid_temperature
    ) + (critical_temperature - vapour_temperature)
    data["FE_pressure_x_temp_ratio"] = (
        failure_pressure * liquid_temperature / (critical_temperature + epsilon)
    )

    # Consolidate the frame before the final feature group. Building many
    # interpretable columns incrementally is easier to audit, while this copy
    # prevents a fragmented internal block layout from slowing later inserts.
    data = data.copy()
    sensor_distance_median = (
        sensor_distance.median() if sensor_distance.notna().any() else np.nan
    )
    failure_pressure_median = (
        failure_pressure.median() if failure_pressure.notna().any() else np.nan
    )

    data["FE_negative_bleve_height_flag"] = (bleve_height < 0).astype(float)
    data["FE_sensor_below_bleve_flag"] = (sensor_vertical_offset < 0).astype(float)
    data["FE_sensor_far_flag"] = (
        (sensor_distance > sensor_distance_median).astype(float)
        if pd.notna(sensor_distance_median)
        else pd.Series(0.0, index=data.index)
    )
    data["FE_high_pressure_flag"] = (
        (failure_pressure > failure_pressure_median).astype(float)
        if pd.notna(failure_pressure_median)
        else pd.Series(0.0, index=data.index)
    )

    for column in ["Sensor ID", "Sensor Position Side", "Thermodynamic Profile"]:
        if column not in data.columns:
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = (
                pd.to_numeric(data[column], errors="coerce")
                .round()
                .astype("Int64")
                .astype(str)
                .replace("<NA>", "missing")
            )
        else:
            data[column] = _clean_text_series(data[column])

    def as_text(column: str) -> pd.Series:
        if column in data.columns:
            return _clean_text_series(data[column])
        return pd.Series(["missing"] * len(data), index=data.index)

    data["FE_KEY_sensor_side"] = as_text("Sensor ID") + "_" + as_text(
        "Sensor Position Side"
    )
    data["FE_KEY_sensor_status"] = as_text("Sensor ID") + "_" + as_text("Status")
    data["FE_KEY_side_status"] = as_text("Sensor Position Side") + "_" + as_text("Status")
    data["FE_KEY_thermo_status"] = as_text("Thermodynamic Profile") + "_" + as_text(
        "Status"
    )
    data["FE_KEY_sensor_side_status"] = (
        as_text("Sensor ID")
        + "_"
        + as_text("Sensor Position Side")
        + "_"
        + as_text("Status")
    )

    return data.replace([np.inf, -np.inf], np.nan)
