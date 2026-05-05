# ==============================
# EXPLAIN MODEL WITH SHAP (FINAL)
# ==============================

import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # ensure plot window opens

# -----------------------------
# PATHS
# -----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "train.csv")
model_path = os.path.join(base_dir, "models", "model.joblib")

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
print("📂 Loading data...")
df = pd.read_csv(data_path)

print("📦 Loading model...")
model = joblib.load(model_path)

print("✅ Model loaded successfully")

# -----------------------------
# PREPARE FEATURES
# (must match training exactly)
# -----------------------------
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

X = df.drop("SalePrice", axis=1)

# 🔥 SAME FEATURE ENGINEERING AS TRAINING
X["HouseAge"] = 2025 - X["YearBuilt"]

X["TotalBathrooms"] = (
    X["FullBath"].fillna(0)
    + 0.5 * X["HalfBath"].fillna(0)
)

X["TotalSF"] = (
    X["GrLivArea"].fillna(0)
    + X["TotalBsmtSF"].fillna(0)
)

# -----------------------------
# SAMPLE (for speed)
# -----------------------------
X_sample = X.sample(100, random_state=42)

# -----------------------------
# PREPROCESS
# -----------------------------
print("⚙️ Preprocessing...")
X_processed = model.named_steps["preprocessor"].transform(X_sample)

# 🔥 CRITICAL: sparse -> dense + numeric
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

X_processed = X_processed.astype(float)

# -----------------------------
# EXTRACT REGRESSOR
# -----------------------------
rf_model = model.named_steps["regressor"]

# -----------------------------
# SHAP (STABLE API)
# -----------------------------
print("🧠 Generating SHAP values...")

explainer = shap.Explainer(rf_model)
shap_values = explainer(X_processed)

# -----------------------------
# PLOT
# -----------------------------
print("📊 Showing SHAP plot...")

shap.plots.beeswarm(shap_values, max_display=15)

plt.show()

print("✅ Done")