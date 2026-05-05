import pandas as pd
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")

# -----------------------------
# LOAD DATA TEMPLATE + MODEL
# -----------------------------
df_template = pd.read_csv(DATA_PATH)

if "Id" in df_template.columns:
    df_template = df_template.drop(columns=["Id"])

df_template = df_template.drop(columns=["SalePrice"])

model = joblib.load(MODEL_PATH)

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="House Price Prediction API")


# -----------------------------
# INPUT SCHEMA
# -----------------------------
class HouseInput(BaseModel):
    GrLivArea: float
    BedroomAbvGr: int
    FullBath: int
    HalfBath: int
    YearBuilt: int
    TotalBsmtSF: float


# -----------------------------
# PREP FUNCTION
# -----------------------------
def prepare_input(user_input: dict):
    input_df = df_template.iloc[[0]].copy()

    for key, value in user_input.items():
        input_df[key] = value

    # Feature Engineering (MUST MATCH TRAINING)
    input_df["HouseAge"] = 2025 - input_df["YearBuilt"]
    input_df["TotalBathrooms"] = input_df["FullBath"] + 0.5 * input_df["HalfBath"]
    input_df["TotalSF"] = input_df["GrLivArea"] + input_df["TotalBsmtSF"]

    return input_df


# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict")
def predict(data: HouseInput):
    try:
        input_df = prepare_input(data.dict())
        prediction = model.predict(input_df)[0]

        return {
            "predicted_price": float(prediction),
            "status": "success"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }