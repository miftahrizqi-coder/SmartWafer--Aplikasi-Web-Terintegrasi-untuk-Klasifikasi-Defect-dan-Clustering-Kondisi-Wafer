
FEATURES = [
    "temperature_c",
    "pressure_torr",
    "gas_flow_sccm",
    "etch_rate_nm_min",
    "voltage_v",
    "current_ma",
]

FEATURE_LABELS = {
    "temperature_c": "Suhu",
    "pressure_torr": "Tekanan",
    "gas_flow_sccm": "Aliran Gas",
    "etch_rate_nm_min": "Etch Rate",
    "voltage_v": "Tegangan",
    "current_ma": "Arus",
}


def validate_sensor_data(data):
    """Validasi enam fitur sensor sebelum prediksi clustering."""
    if not isinstance(data, dict):
        raise ValueError("Data sensor harus berupa objek JSON.")

    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        raise ValueError(
            "Fitur sensor belum lengkap: " + ", ".join(missing)
        )

    validated = {}

    for feature in FEATURES:
        try:
            value = float(data[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"Nilai {feature} harus berupa angka."
            )

        if not __import__("math").isfinite(value):
            raise ValueError(
                f"Nilai {feature} harus berupa angka yang valid."
            )

        validated[feature] = value

    return validated
