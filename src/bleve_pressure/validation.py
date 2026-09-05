"""Prediction-contract checks shared by notebooks and automated tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_rows: int,
    id_column: str = "ID",
    target_column: str = "Target Pressure (bar)",
) -> pd.DataFrame:
    """Return a validated copy of a prediction frame.

    The export contract requires an identifier and one finite, strictly
    positive pressure estimate per input scenario. A copy is returned so that
    downstream file writing cannot mutate the caller's frame unexpectedly.
    """

    required: Sequence[str] = (id_column, target_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {', '.join(missing)}")

    result = frame.loc[:, list(required)].copy()
    if len(result) != expected_rows:
        raise ValueError(
            f"Prediction row count mismatch: got {len(result)}, expected {expected_rows}."
        )
    if result[id_column].isna().any():
        raise ValueError("Prediction identifiers contain missing values.")
    if result[id_column].duplicated().any():
        raise ValueError("Prediction identifiers must be unique.")

    values = pd.to_numeric(result[target_column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Predictions contain missing or non-finite values.")
    if (values <= 0).any():
        raise ValueError("Predictions must be strictly positive.")

    result[target_column] = values.astype(float)
    return result
