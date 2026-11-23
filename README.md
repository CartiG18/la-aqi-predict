# Los Angeles AQI Prediction System

A machine learning pipeline for estimating air quality (PM2.5 concentration) in the Los Angeles basin. This system ingests real-time data from PurpleAir sensors, processes it to generate features, trains a Random Forest model that predicts PM2.5 from temperature and humidity, and serves predictions via both a REST API and an interactive Dashboard.

## Features

*   **Automated Data Ingestion:** Robust extraction from PurpleAir API with retry logic and Pydantic validation.
*   **Geospatial Visualization:** Keeps sensor locations (lat/lon) to map PM2.5 levels across the basin.
*   **Machine Learning Pipeline:** Modular design for Feature Engineering, Training, and Evaluation.
*   **Production API:** FastAPI application for real-time inference, containerized with Docker.
*   **Interactive Dashboard:** Streamlit app for data visualization, model testing, and geospatial exploration.

## Project Structure

```
la-aqi-predict/
├── .github/workflows/   # CI/CD (GitHub Actions)
├── data/                # Data storage (gitignored)
│   ├── raw/             # Raw CSV snapshots
│   └── processed/       # Cleaned data for modeling
├── src/
│   ├── api/             # FastAPI Serving Layer
│   │   └── main.py
│   ├── app/             # Streamlit Dashboard
│   │   └── dashboard.py
│   ├── data/            # Data Ingestion
│   │   └── ingest.py
│   ├── features/        # Feature Engineering
│   │   └── build_features.py
│   └── models/          # Model Training
│       ├── train_model.py
│       └── aqi_model.joblib  # Trained model artifact
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
└── .env                 # Secrets (API Keys)
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd la-aqi-predict
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment:**
    Create a `.env` file in the root directory:
    ```env
    PURPLE_AIR_API_KEY=your_api_key_here
    ```

## Usage

### 1. Run the Data Pipeline (ETL & Training)
Execute these scripts in order to fetch data, process it, and retrain the model.

```bash
# 1. Ingest Data (fetches snapshot from PurpleAir)
python src/data/ingest.py

# 2. Engineer Features (cleans and preps data)
python src/features/build_features.py

# 3. Train Model (trains Random Forest and saves artifact)
python src/models/train_model.py
```

### 2. Launch the Interactive Dashboard
Visualize the data and test the model with a UI.

```bash
streamlit run src/app/dashboard.py
```

### 3. Start the Inference API
Serve predictions via REST API.

```bash
uvicorn src.api.main:app --reload
```
*   **Docs:** Go to `http://127.0.0.1:8000/docs` to test the API interactively.
*   **Sample Request:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"temperature": 75.0, "humidity": 45.0}'
    ```

## Docker Deployment

Build and run the inference API in a container (the dashboard is not included; run it locally as above):

```bash
docker build -t la-aqi-predictor .
docker run -p 8000:8000 la-aqi-predictor
```

## 🔬 Methodology

*   **Data Source:** PurpleAir (PM2.5, Temperature, Humidity) — a single snapshot of current 10-minute averages.
*   **Preprocessing:** Negative-value clipping, dropping rows with missing PM2.5.
*   **Features:** Temperature, humidity, and their interaction term.
*   **Model:** Random Forest Regressor (Scikit-Learn).
*   **Evaluation:** MAE, RMSE, R-Squared.

### Limitations

*   **Not a forecast:** the model estimates *current* PM2.5 from current conditions. True forecasting would require historical time-series data.
*   **PM2.5, not AQI:** predictions are PM2.5 concentrations (µg/m³), bucketed into AQI categories; they are not converted to the numeric AQI scale.
*   **Sensor temperature bias:** PurpleAir's `temperature` field is measured inside the sensor housing and typically reads several °F above ambient, so the model learns from that biased signal.
*   **Model/library coupling:** `aqi_model.joblib` must be loaded with the same scikit-learn version it was trained with (pinned in `requirements.txt`). Update the pin whenever you retrain.
