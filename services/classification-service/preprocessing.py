
import pandas as pd


def feature_engineering(df):
    """Menerapkan feature engineering yang sama dengan proses training."""
    df = df.copy()

    if {"voltage_v", "current_ma"}.issubset(df.columns):
        df["electrical_power_mw"] = (
            df["voltage_v"] * df["current_ma"]
        )

    df = df.drop(columns=["wafer_id"], errors="ignore")
    return df
