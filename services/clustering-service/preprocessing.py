
import pandas as pd

from schemas import FEATURES, validate_sensor_data


def prepare_sensor_data(data):
    """Validasi dan susun data sensor menjadi DataFrame."""
    validated_data = validate_sensor_data(data)

    return pd.DataFrame(
        [[validated_data[feature] for feature in FEATURES]],
        columns=FEATURES,
    )
