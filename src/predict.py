import joblib
import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "model.joblib")

model = joblib.load(model_path)

def prepare_features(df):
    df["HouseAge"] = 2025 - df["YearBuilt"]
    df["TotalBathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"]
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    return df

def predict_price(input_dict):
    df = pd.DataFrame([input_dict])
    df = prepare_features(df)
    prediction = model.predict(df)[0]
    return prediction