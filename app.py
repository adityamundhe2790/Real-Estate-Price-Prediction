import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Real Estate Valuation System", layout="wide")

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

model = load_model()
data = load_data()

# -----------------------------
# TITLE
# -----------------------------
st.title("Real Estate Valuation System")

# -----------------------------
# INPUT UI
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Property Configuration")

    area = st.slider("Living Area (sq ft)", 500, 5000, 1200)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)
    year_built = st.slider("Year Built", 1950, 2025, 2005)
    basement = st.slider("Basement Area", 0, 2000, 500)

# -----------------------------
# BUILD INPUT DATAFRAME
# -----------------------------
input_df = pd.DataFrame({
    "GrLivArea": [area],
    "BedroomAbvGr": [bedrooms],
    "FullBath": [bathrooms],
    "HalfBath": [0],
    "YearBuilt": [year_built],
    "TotalBsmtSF": [basement]
})

# -----------------------------
# FEATURE ENGINEERING (CRITICAL)
# -----------------------------
input_df["HouseAge"] = 2025 - input_df["YearBuilt"]
input_df["TotalBathrooms"] = input_df["FullBath"] + (0.5 * input_df["HalfBath"])
input_df["TotalSF"] = input_df["GrLivArea"] + input_df["TotalBsmtSF"]

# -----------------------------
# FIX MISSING COLUMNS (SAFE)
# -----------------------------
model_features = model.named_steps["preprocessor"].feature_names_in_

for col in model_features:
    if col not in input_df.columns:
        if col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                input_df[col] = data[col].median()
            else:
                input_df[col] = data[col].mode()[0]
        else:
            input_df[col] = 0

# Ensure correct column order
input_df = input_df[model_features]

# -----------------------------
# PREDICTION
# -----------------------------
prediction = model.predict(input_df)[0]

# -----------------------------
# OUTPUT UI
# -----------------------------
with col2:
    st.subheader("Valuation Results")

    c1, c2, c3 = st.columns(3)

    avg_price = data["SalePrice"].mean()

    with c1:
        st.metric("Predicted Value", f"₹ {int(prediction):,}")

    with c2:
        st.metric("Market Average", f"₹ {int(avg_price):,}")

    with c3:
        signal = "Overvalued" if prediction > avg_price else "Undervalued"
        st.metric("Valuation Signal", signal)

# -----------------------------
# DATA INSIGHTS
# -----------------------------
st.markdown("---")
st.subheader("Market Insights")

colA, colB = st.columns(2)

with colA:
    st.write("Price Distribution")
    st.bar_chart(data["SalePrice"])

with colB:
    st.write("Living Area vs Price")
    st.scatter_chart(data[["GrLivArea", "SalePrice"]])

# -----------------------------
# SIMPLE SHAP (SAFE)
# -----------------------------
st.markdown("---")
st.subheader("Feature Importance (Optional)")

if st.button("Show Feature Importance"):
    try:
        import shap

        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["regressor"]

        X_processed = preprocessor.transform(input_df)

        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        X_processed = np.asarray(X_processed, dtype=np.float32)

        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_processed)

        fig = plt.figure()
        shap.summary_plot(shap_values, X_processed, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error("SHAP failed, but prediction still works.")