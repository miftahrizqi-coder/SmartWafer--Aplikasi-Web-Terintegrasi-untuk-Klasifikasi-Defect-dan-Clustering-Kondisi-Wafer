import json
import sys
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

# Agar `preprocessing.py` di root proyek bisa di-import. File ini juga
# diperlukan bila model memakai class buatan sendiri (mis. IsolationScorer).
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ==================================================
# 2. FEATURE ENGINEERING (harus sama dengan training)
# ==================================================

try:
    import preprocessing

    def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        return preprocessing.feature_engineering(df)

except ImportError:
    # Cadangan: rumus yang sama dengan preprocessing.py
    def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["electrical_power_mw"] = df["voltage_v"] * df["current_ma"]
        return df.drop(columns=["wafer_id"], errors="ignore")

# ==================================================
# 3. MEMUAT MODEL DAN METADATA
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

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

# Threshold hasil pemilihan di notebook (maksimalkan F2 pada skor out-of-fold).
DECISION_THRESHOLD = float(metadata.get("decision_threshold", 0.5))

# Nilai process_step yang dikenal model (diambil dari OneHotEncoder hasil training).
try:
    ALLOWED_PROCESS_STEPS = [
        str(v)
        for v in model.named_steps["prep"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .categories_[0]
    ]
except Exception:
    ALLOWED_PROCESS_STEPS = []  # validasi dilewati bila struktur model berbeda


def get_score(m, features: pd.DataFrame) -> float:
    """Skor risiko defect: probabilitas kelas 1, atau decision_function."""
    if hasattr(m, "predict_proba"):
        return float(m.predict_proba(features)[0, 1])
    return float(m.decision_function(features)[0])


# ==================================================
# 4. INISIALISASI FASTAPI
# ==================================================

app = FastAPI(
    title="SmartWafer Classification API",
    description="API untuk klasifikasi defect wafer semikonduktor",
    version="1.1.0",
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
# 5. SKEMA INPUT
# ==================================================


class WaferInput(BaseModel):
    temperature_c: float = Field(allow_inf_nan=False)
    pressure_torr: float = Field(gt=0, allow_inf_nan=False)
    gas_flow_sccm: float = Field(gt=0, allow_inf_nan=False)
    etch_rate_nm_min: float = Field(gt=0, allow_inf_nan=False)
    voltage_v: float = Field(gt=0, allow_inf_nan=False)
    current_ma: float = Field(gt=0, allow_inf_nan=False)
    process_step: str = Field(min_length=1)


# ==================================================
# 6. ENDPOINT BERANDA
# ==================================================


@app.get("/")
def home():
    return {
        "application": "SmartWafer",
        "status": "running",
        "message": "Classification API is ready",
    }


# ==================================================
# 7. ENDPOINT HEALTH CHECK
# ==================================================


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


# ==================================================
# 8. INFORMASI MODEL
# ==================================================


@app.get("/model-info")
def model_info():
    return {
        "target": metadata.get("target_column", "defect_label"),
        "selected_model": metadata.get("selected_model", "unknown"),
        "features": FEATURE_COLUMNS,
        "process_steps": ALLOWED_PROCESS_STEPS,
        "decision_threshold": DECISION_THRESHOLD,
        "classes": metadata.get(
            "class_mapping",
            {"0": "Tidak Defect", "1": "Defect"},
        ),
        "cv_metrics": metadata.get("cv_metrics", {}),
        "warning": metadata.get("warning", ""),
    }


# ==================================================
# 9. ENDPOINT PREDIKSI
# ==================================================


@app.post("/predict")
def predict(data: WaferInput):
    # Validasi nilai kategori agar typo tidak diam-diam diabaikan model.
    if ALLOWED_PROCESS_STEPS and data.process_step not in ALLOWED_PROCESS_STEPS:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "process_step tidak dikenal",
                "allowed": ALLOWED_PROCESS_STEPS,
            },
        )

    try:
        # Ubah input menjadi satu baris DataFrame lalu terapkan
        # feature engineering yang sama dengan training.
        input_data = apply_feature_engineering(
            pd.DataFrame([data.model_dump()])
        )

        # Urutan fitur mengikuti metadata training.
        input_data = input_data[FEATURE_COLUMNS]

        # Keputusan memakai skor + threshold hasil training,
        # bukan model.predict() yang memakai threshold 0,5.
        score = get_score(model, input_data)
        prediction = int(score >= DECISION_THRESHOLD)

        result: dict[str, Any] = {
            "prediction": prediction,
            "label": "Defect" if prediction == 1 else "Tidak Defect",
            "score": score,
            "threshold": DECISION_THRESHOLD,
        }

        # Probabilitas hanya ada bila model mendukungnya. Untuk model ber-class
        # weight / SMOTE nilainya tidak terkalibrasi; tampilkan sebagai skor risiko.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            result["probabilities"] = {
                str(int(cls)): float(prob)
                for cls, prob in zip(model.classes_, probabilities)
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