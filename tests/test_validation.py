import pandas as pd
import pytest

from bleve_pressure.validation import validate_prediction_frame


def test_valid_prediction_frame_preserves_required_contract() -> None:
    frame = pd.DataFrame({"ID": [1, 2], "Target Pressure (bar)": [0.25, 1.75]})

    result = validate_prediction_frame(frame, expected_rows=2)

    assert list(result.columns) == ["ID", "Target Pressure (bar)"]
    assert result["Target Pressure (bar)"].tolist() == [0.25, 1.75]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"ID": [1], "Target Pressure (bar)": [0.0]}), "strictly positive"),
        (pd.DataFrame({"ID": [1, 1], "Target Pressure (bar)": [0.3, 0.4]}), "unique"),
        (pd.DataFrame({"ID": [1], "Target Pressure (bar)": [float("nan")]}), "non-finite"),
    ],
)
def test_invalid_prediction_frames_are_rejected(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_prediction_frame(frame, expected_rows=len(frame))
