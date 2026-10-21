import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    return df

def feature_engineering(df):
    df = df.copy()

    if {"voltage_v", "current_ma"}.issubset(df.columns):
        df["electrical_power_mw"] = (
            df["voltage_v"] * df["current_ma"]
        )

    df = df.drop(columns=["wafer_id"], errors="ignore")
    return df

def split_features_target(df):
    X = df.drop(columns=["defect_label"])
    y = df["defect_label"]
    return X, y
