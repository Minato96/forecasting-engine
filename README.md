# 📉 Neural Time-Series Forecasting Engine

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-RealTime-red)

> **A production-ready deep learning pipeline for multivariate time-series forecasting, capable of real-time inference via a high-performance API.**

---

## 🚀 Overview

This project implements an end-to-end system for forecasting complex temporal data (e.g., energy usage, climate metrics). Unlike traditional statistical approaches (ARIMA), this engine leverages **Long Short-Term Memory (LSTM)** networks to capture non-linear and long-range temporal dependencies in multivariate signals.

The system is engineered with **deployment in mind**, featuring a custom sliding-window data pipeline and a lightweight inference server that delivers sub-second predictions.

---

## ⚡ Key Features

- **Deep Learning Core**  
  Stacked LSTM architecture with Dropout regularization to handle noisy sensor data.

- **Custom Data Engineering**  
  A reusable `WindowGenerator` class that manages rolling windows, feature scaling, and forecasting horizons.

- **Real-Time Inference**  
  Deployed via **FastAPI**, simulating a production-grade microservice for streaming or IoT workloads.

- **Multivariate Forecasting**  
  Consumes multiple correlated signals (e.g., pressure + temperature) to improve predictive accuracy.

---

## 📐 System Architecture

The pipeline transforms continuous data streams into supervised learning windows.

### 1. Data Pipeline (`WindowGenerator`)

- **Input:** Continuous raw time-series CSV
- **Transformation:** Converts sequences into `(Input, Label)` pairs  
  - Input Width: 24 hours (history)  
  - Label Width: 1 hour (forecast)  
  - Shift: 1 hour
- **Normalization:** Z-score standardization (μ = 0, σ = 1) for numerical stability

### 2. Model Architecture (`LSTM`)

- **Layer 1:** LSTM (64 units) – sequential encoding  
- **Layer 2:** Dropout (0.2) – regularization  
- **Layer 3:** LSTM (32 units) – temporal compression  
- **Layer 4:** Dense (1) – linear regression output

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|--------|-----------|--------|
| Deep Learning | TensorFlow / Keras | Model definition & training |
| Data Processing | Pandas, NumPy | Vectorized time-series handling |
| API Framework | FastAPI | High-concurrency inference |
| Server | Uvicorn | ASGI production server |

---

## 📂 Project Structure

~~~text
├── src/
│   ├── window_generator.py    # Core sliding-window logic
│   ├── model.py               # LSTM architecture
│   └── train.py               # Training pipeline
├── app.py                     # Inference API (Product layer)
├── forecasting_engine.h5      # Trained model artifact
└── requirements.txt
~~~

---

## ⚡ Quick Start

### 1. Installation

~~~bash
git clone https://github.com/yourusername/neural-forecasting-engine.git
cd neural-forecasting-engine
pip install -r requirements.txt
~~~

### 2. Train the Model

Automatically downloads the **Jena Climate Dataset** (Max Planck Institute), processes it, and trains the LSTM.

~~~bash
python -m src.train
~~~

Output:  
`forecasting_engine.h5` saved to the project root.

---

### 3. Launch the API

~~~bash
python app.py
~~~

Server running at:  
`http://0.0.0.0:8000`

---

## 🔌 API Documentation

**Endpoint:** `POST /predict`  

**Description:**  
Accepts a historical sequence (past 24 time steps) and returns the forecast for the next step.

### Sample Request

~~~json
{
  "history": [
    [980.5, -3.2],
    [981.2, -3.1]
  ]
}
~~~

### Sample Response

~~~json
{
  "predicted_temperature_normalized": -3.05
}
~~~

---

## 📊 Performance

- **Dataset:** Jena Climate (2009–2016)
- **Metric:** Mean Absolute Error (MAE)
- **Result:** ~0.08 (normalized) on validation set
- **Inference Latency:** < 45 ms per request on CPU

---

## Author

**Charan**  
AI Engineer & Deep Learning Practitioner
