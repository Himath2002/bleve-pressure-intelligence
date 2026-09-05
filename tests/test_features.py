import numpy as np
import pandas as pd

from bleve_pressure.features import add_physics_features, normalise_status_value


def _scenario() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Status": ["saperheated"],
            "Liquid Ratio (%)": [60.0],
            "Tank Length (m)": [10.0],
            "Tank Width (m)": [4.0],
            "Tank Height (m)": [5.0],
            "Vapour Height (m)": [2.0],
            "BLEVE Height (m)": [3.0],
            "Tank Failure Pressure (bar)": [20.0],
            "Obstacle Width (m)": [3.0],
            "Obstacle Height (m)": [2.0],
            "Obstacle Thickness (m)": [0.25],
            "Obstacle Distance to BLEVE (m)": [6.0],
            "Obstacle Angle": [30.0],
            "Sensor ID": [4],
            "Sensor Position Side": ["east"],
            "Sensor Position x": [8.0],
            "Sensor Position y": [6.0],
            "Sensor Position z": [3.0],
            "Liquid Temperature (K)": [420.0],
            "Vapour Temperature (K)": [430.0],
            "Liquid Critical Temperature (K)": [510.0],
            "Liquid Boiling Temperature (K)": [350.0],
            "Liquid Critical Pressure (bar)": [42.0],
            "Thermodynamic Profile": ["demo"],
        }
    )


def test_status_normalisation_handles_known_variants() -> None:
    assert normalise_status_value("SUPERHEATED") == "Superheated"
    assert normalise_status_value("saperheated") == "Superheated"
    assert normalise_status_value(" subcooled ") == "Subcooled"
    assert normalise_status_value(None) == "missing"


def test_physics_features_are_finite_for_valid_scenario() -> None:
    source = _scenario()

    result = add_physics_features(source)

    assert source.columns.tolist() != result.columns.tolist()
    assert result.loc[0, "Status"] == "Superheated"
    assert result.loc[0, "FE_liquid_ratio_fraction"] == 0.6
    assert result.loc[0, "FE_tank_volume_box"] == 200.0
    assert result.loc[0, "FE_sensor_3d_distance_bleve"] == 10.0
    engineered = result.filter(regex="^FE_").select_dtypes(include=[np.number])
    assert np.isfinite(engineered.to_numpy()).all()


def test_missing_optional_columns_do_not_raise() -> None:
    result = add_physics_features(pd.DataFrame({"Status": ["Subcool"]}))

    assert result.loc[0, "Status"] == "Subcooled"
    assert "FE_tank_volume_box" in result.columns
