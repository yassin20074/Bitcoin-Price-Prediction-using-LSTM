# 🚀 DeepBTC — Enterprise-Grade Bi-LSTM Crypto Forecasting System

DeepBTC is a production-structured deep learning system designed for high-quality Bitcoin time series forecasting using a **Bidirectional LSTM architecture**. The project demonstrates an end-to-end ML pipeline following professional software engineering standards, including modular design, data preprocessing, model regularization, evaluation, and visualization.

---

## 📌 Executive Summary

This system leverages historical Bitcoin market data and applies advanced sequence modeling techniques to capture complex temporal dependencies. The architecture is optimized using regularization (Dropout) and EarlyStopping to ensure generalization and prevent overfitting.

---

## 🧠 Model Architecture

- Bidirectional LSTM (50 units)
- Stacked LSTM (50 units)
- Dropout (0.2)
- Dense Layers (25 → 1)
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)

---

## ⚙️ Pipeline Overview

1. Data Ingestion via `yfinance`
2. Normalization using `MinMaxScaler`
3. Sliding Window Sequence Generation (60 timesteps)
4. 80/20 Time-Series Split
5. LSTM Reshaping (samples, timesteps, features)
6. Training with EarlyStopping
7. Evaluation (MSE, MAE, R²)
8. Visualization of Real vs Predicted Prices

---

## 📊 Performance Metrics

- MSE — Squared prediction error
- MAE — Absolute prediction deviation
- R² — Explained variance score

---

## 🛠️ Tech Stack

Python • TensorFlow/Keras • NumPy • Pandas • Scikit-learn • Matplotlib • yfinance

---

## 🔮 Future Enhancements
Multi-feature inputs (OHLCV)
Technical Indicators (RSI, MACD)
Hyperparameter Optimization
REST API Deployment (FastAPI)
Real-time Prediction Service

## Created by : yassin sanad
