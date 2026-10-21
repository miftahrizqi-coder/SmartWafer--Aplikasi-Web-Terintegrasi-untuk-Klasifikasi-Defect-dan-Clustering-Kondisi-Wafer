
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from preprocessing import feature_engineering

app = Flask(__name__)

# Lokasi model: folder models/classification di root proyek
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "classification" / "best_model.joblib"

model = joblib.load(MODEL_PATH)

REQUIRED_FEATURES = [
    "temperature_c",
    "pressure_torr",
    "gas_flow_sccm",
    "etch_rate_nm_min",
    "voltage_v",
    "current_ma",
    "process_step",
    "electrical_power_mw",
]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "service": "classification-service",
        "status": "ok"
    }), 200



@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request harus berupa JSON object."
        }), 400

    # voltage_v dan current_ma diperlukan untuk menghitung
    # electrical_power_mw, jadi keduanya wajib dikirim.
    input_fields = [
        "temperature_c",
        "pressure_torr",
        "gas_flow_sccm",
        "etch_rate_nm_min",
        "voltage_v",
        "current_ma",
        "process_step",
    ]

    missing_fields = [
        field for field in input_fields
        if field not in data or data[field] is None or data[field] == ""
    ]

    if missing_fields:
        return jsonify({
            "error": "Ada fitur wafer yang belum diisi.",
            "missing_fields": missing_fields,
        }), 400

    try:
        numeric_fields = [
            "temperature_c",
            "pressure_torr",
            "gas_flow_sccm",
            "etch_rate_nm_min",
            "voltage_v",
            "current_ma",
        ]

        row = {
            field: float(data[field])
            for field in numeric_fields
        }

        if not all(
            __import__("math").isfinite(value)
            for value in row.values()
        ):
            raise ValueError("Nilai harus berupa angka yang valid.")

    except (TypeError, ValueError):
        return jsonify({
            "error": "Fitur numerik harus berupa angka yang valid."
        }), 400

    row["process_step"] = str(data["process_step"])

    # Terapkan feature engineering yang sama dengan training.
    input_df = pd.DataFrame([row])
    input_df = feature_engineering(input_df)

    # Pastikan fitur sesuai dengan model.
    input_df = input_df[REQUIRED_FEATURES]

    try:
        prediction = int(model.predict(input_df)[0])

        response = {
            "service": "classification-service",
            "prediction": prediction,
            "defect_status": (
                "Defect" if prediction == 1 else "Normal"
            ),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            response["probabilities"] = {
                str(label): round(float(probability), 4)
                for label, probability in zip(
                    model.classes_, probabilities
                )
            }

        return jsonify(response), 200

    except Exception as error:
        app.logger.exception("Prediksi klasifikasi gagal")
        return jsonify({
            "error": "Terjadi kesalahan saat melakukan prediksi.",
            "detail": str(error),
        }), 500



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
