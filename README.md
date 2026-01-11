# Los Angeles AQI Forecasting System 🌫️☀️

A robust, production-ready machine learning pipeline for forecasting Air Quality Index (AQI) in the Los Angeles basin. This system ingests real-time data from PurpleAir sensors, processes it to generate features, trains a Random Forest predictive model, and serves predictions via both a REST API and an interactive Dashboard.

## 🚀 Features

*   **Automated Data Ingestion:** Robust extraction from PurpleAir API with retry logic and Pydantic validation.
*   **Geospatial Processing:** Handles sensor locations (lat/lon) for mapping and spatial analysis.
*   **Machine Learning Pipeline:** Modular design for Feature Engineering, Training, and Evaluation.
*   **Production API:** FastAPI application for real-time inference, containerized with Docker.
*   **Interactive Dashboard:** Streamlit app for data visualization, model testing, and geospatial exploration.

## 📂 Project Structure

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
│       └── train_model.py
├── tests/               # Unit tests
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
└── .env                 # Secrets (API Keys)
```

## 🛠️ Setup

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

## 🏃‍♂️ Usage

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

## 🐳 Docker Deployment

Build and run the entire application container:

```bash
docker build -t la-aqi-predictor .
docker run -p 8000:8000 la-aqi-predictor
```

## 🔬 Methodology

*   **Data Source:** PurpleAir (PM2.5, Temperature, Humidity).
*   **Preprocessing:** Outlier clipping, missing value handling.
*   **Model:** Random Forest Regressor (Scikit-Learn).
*   **Evaluation:** MAE, RMSE, R-Squared.

## 🤝 Contributing

1.  Fork the repo
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit changes (`git commit -m 'Add amazing feature'`)
4.  Push to branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request
