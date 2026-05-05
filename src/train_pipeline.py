import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# -----------------------------
# PATH HANDLING
# -----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "train.csv")
model_path = os.path.join(base_dir, "models", "model.joblib")

print("📂 Loading data from:", data_path)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(data_path)

# -----------------------------
# TARGET
# -----------------------------
target = "SalePrice"

# Drop ID column if exists
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df["HouseAge"] = 2025 - df["YearBuilt"]
df["TotalBathrooms"] = df["FullBath"] + (0.5 * df["HalfBath"])
df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]

# -----------------------------
# SPLIT FEATURES
# -----------------------------
X = df.drop(columns=[target])
y = df[target]

# -----------------------------
# IDENTIFY COLUMN TYPES (FIXED)
# -----------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

# -----------------------------
# PREPROCESSING
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -----------------------------
# MODEL PIPELINE
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("🚀 Training model...")
model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("📊 Improved RMSE:", rmse)

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, model_path)

print("✅ Model saved at:", model_path)