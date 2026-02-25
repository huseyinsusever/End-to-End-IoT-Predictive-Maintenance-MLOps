# End-to-End-IoT-Predictive-Maintenance-MLOps

IoT Predictive Maintenance System

![Python Details](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-00a393)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED)
![Tests](https://img.shields.io/badge/Tests-Pytest-success)

An end-to-end Machine Learning Operations (MLOps) project demonstrating predictive maintenance for manufacturing equipment. This system utilizes both unsupervised and supervised learning to detect anomalies in real-time and predict the Remaining Useful Life (RUL) of industrial IoT sensors.

## 🌟 Key Features

- **Data Engineering**: Automated generation of realistic synthetic IoT sensor data (temperature, vibration, pressure) with degradation patterns.
- **Anomaly Detection**: Unsupervised learning using `IsolationForest` to detect sudden equipment failures and abnormal spikes.
- **RUL Prediction**: Supervised predictive modeling using `RandomForestRegressor` to estimate machinery lifespan.
- **Microservices Architecture**: 
  - A robust backend using **FastAPI** to serve real-time predictions.
  - A dynamic front-end dashboard using **Streamlit** and **Plotly** for real-time visualization.
- **Containerization**: Fully Dockerized for seamless deployment using `docker-compose`.
- **CI/CD Ready**: Integrated unit testing via `pytest`.

## 🏗️ Architecture

1. **Data Layer**: Local `data/` directory mimicking a data lake.
2. **Model Training Pipeline**: Standalone scripts to engineer features (rolling means/std) and train models.
3. **Serving Layer**: FastAPI exposes a `/predict` endpoint that ingests streaming JSON data.
4. **Presentation Layer**: Streamlit polls the backend and provides operational insights and alerts.

## 🚀 Quick Start (Docker - Recommended)

The easiest way to run the application is using Docker Compose.

```bash
# 1. Generate Synthetic Data and Train Models locally
python src/simulate_data.py
python src/train.py

# 2. Build and run containers
docker-compose up --build
```
- **Dashboard**: `http://localhost:8501`
- **API Docs**: `http://localhost:8000/docs`

## 💻 Manual Setup (Virtual Environment)

If you prefer to run the application natively without Docker:

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Data Generation & Model Training
python src/simulate_data.py
python src/train.py

# 4. Start the API Server (Terminal 1)
uvicorn src.api:app --reload

# 5. Start the Dashboard (Terminal 2)
streamlit run src/app.py
```

## 🧪 Testing

The project includes unit tests for the API endpoints to ensure code reliability.

```bash
# Run pytest
pytest tests/
```

## 👨‍💻 Author Notes

This project was built to demonstrate proficiency in applied machine learning, software engineering best practices, and MLOps deployment strategies required for modern AI Engineering roles.
