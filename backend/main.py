import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==================================================
# 1. KONFIGURASI PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "classification"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ==================================================
# 2. MEMUAT MODEL DAN METADATA
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
else:
    metadata = {}

FEATURE_COLUMNS = metadata.get(
    "feature_columns",
    [
        "temperature_c",
        "pressure_torr",
        "gas_flow_sccm",
        "etch_rate_nm_min",
        "voltage_v",
        "current_ma",
        "process_step",
        "electrical_power_mw",
    ],
)

# ==================================================
# 3. INISIALISASI FASTAPI
# ==================================================

app = FastAPI(
    title="SmartWafer Classification API",
    description="API untuk klasifikasi defect wafer semikonduktor",
    version="1.0.0",
)

# Sesuaikan origin dengan alamat frontend saat development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ==================================================
# 4. SKEMA INPUT
# ==================================================

class WaferInput(BaseModel):
    temperature_c: float
    pressure_torr: float
    gas_flow_sccm: float
    etch_rate_nm_min: float
    voltage_v: float
    current_ma: float
    process_step: str

# ==================================================
# 5. ENDPOINT BERANDA
# ==================================================

@app.get("/")
def home():
    return {
        "application": "SmartWafer",
        "status": "running",
        "message": "Classification API is ready",
    }

# ==================================================
# 6. ENDPOINT HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }

# ==================================================
# 7. INFORMASI MODEL
# ==================================================

@app.get("/model-info")
def model_info():
    return {
        "target": metadata.get(
            "target_column", "defect_label"
        ),
        "selected_model": metadata.get(
            "selected_model", "unknown"
        ),
        "features": FEATURE_COLUMNS,
        "classes": metadata.get(
            "class_mapping",
            {
                "0": "Tidak Defect",
                "1": "Defect",
            },
        ),
    }

# ==================================================
# 8. ENDPOINT PREDIKSI
# ==================================================

@app.post("/predict")
def predict(data: WaferInput):
    try:
        # Ubah input menjadi satu baris DataFrame.
        input_data = pd.DataFrame(
            [data.model_dump()]
        )

        # Feature engineering harus sama dengan training.
        input_data["electrical_power_mw"] = (
            input_data["voltage_v"]
            * input_data["current_ma"]
        )

        # Pastikan fitur yang dibutuhkan model tersedia.
        missing_features = [
            col
            for col in FEATURE_COLUMNS
            if col not in input_data.columns
        ]

        if missing_features:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Fitur model belum lengkap",
                    "missing_features": missing_features,
                },
            )

        # Urutan fitur mengikuti metadata training.
        input_data = input_data[FEATURE_COLUMNS]

        # Prediksi kelas.
        prediction = int(model.predict(input_data)[0])

        result: dict[str, Any] = {
            "prediction": prediction,
            "label": (
                "Defect"
                if prediction == 1
                else "Tidak Defect"
            ),
        }

        # Tambahkan probabilitas jika model mendukungnya.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.named_steps[
                "classifier"
            ].classes_

            result["probabilities"] = {
                str(int(cls)): float(prob)
                for cls, prob in zip(classes, probabilities)
            }

        return result

    except HTTPException:
        raise
    except Exception as exc:
        # Detail error dicatat pada terminal server,
        # tidak dikirim ke pengguna API.
        print(f"Prediction error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Prediksi gagal. Periksa log backend.",
        ) from exc