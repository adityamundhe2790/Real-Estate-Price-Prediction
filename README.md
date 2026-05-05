# Real Estate Price Prediction Dashboard

A machine learning-powered dashboard that predicts house prices and provides market insights through an interactive UI.

---

## 🚀 Features

- Price prediction using Random Forest Regressor
- Feature engineering (HouseAge, TotalSF, TotalBathrooms)
- Interactive dashboard built with Streamlit
- Real-time predictions based on user inputs
- Market analytics:
  - Price distribution
  - Area vs price comparison
  - Correlation insights
- Clean dark-themed fintech-style UI

---

## 🧠 Tech Stack

- Python
- Scikit-learn
- Pandas & NumPy
- Streamlit
- Matplotlib

---

## 📂 Project Structure
House-Price-Prediction/
│
├── data/ # Dataset
├── src/ # Training pipeline
├── models/ # Saved model
├── app.py # Streamlit dashboard
├── requirements.txt
└── README.md

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python src/train_pipeline.py
python -m streamlit run app.py
