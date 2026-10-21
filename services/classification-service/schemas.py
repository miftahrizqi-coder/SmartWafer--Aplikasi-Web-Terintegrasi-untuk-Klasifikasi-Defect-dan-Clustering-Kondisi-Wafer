
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Data sensor yang dikirim untuk klasifikasi defect wafer."""

    temperature: float = Field(..., ge=0, le=150)
    vibration: float = Field(..., ge=-50, le=150)
    humidity: float = Field(..., ge=0, le=100)
    pressure: float = Field(..., ge=0, le=10)
    energy_consumption: float = Field(..., ge=0, le=10)


class PredictionResponse(BaseModel):
    """Hasil prediksi klasifikasi defect wafer."""

    prediction: Any
    status: str = "success"
