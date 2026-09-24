import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class AQIFeatureEngineer:
    def __init__(self, input_path: Optional[Path] = None, output_path: Optional[Path] = None):
        self.input_path = input_path or PROJECT_ROOT / "data" / "raw" / "raw_la_aqi.csv"
        self.output_path = output_path or PROJECT_ROOT / "data" / "processed" / "processed_la_aqi.csv"

    def load_data(self) -> pd.DataFrame:
        """Loads raw data."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file {self.input_path} not found.")
        return pd.read_csv(self.input_path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data and handles missing values.
        """
        # Rename columns to standard internal names
        df = df.rename(columns={"pm2.5_10minute": "pm2_5"})

        # Ensure correct types
        df['pm2_5'] = pd.to_numeric(df['pm2_5'], errors='coerce')
        
        # Drop rows where target (pm2_5) is missing
        df = df.dropna(subset=['pm2_5'])
        
        # Handle outliers (simple IQR method or clipping)
        # For AQI, values > 500 are rare but possible. Let's clip negative values.
        df['pm2_5'] = df['pm2_5'].clip(lower=0)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates model features from temperature and humidity.

        The ingested data is a single snapshot (current 10-minute averages), so
        there are no temporal features; forecasting would require historical data.
        Latitude/longitude are kept for the dashboard map but not used as features.
        """
        # High Temp + Low Humidity often correlates with fires/dust
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

        return df

    def run(self):
        print(f"Loading data from {self.input_path}...")
        df = self.load_data()
        
        print("Preprocessing...")
        df = self.preprocess(df)
        
        print("Engineering features...")
        df = self.engineer_features(df)
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Saved processed data to {self.output_path}")
        print(df.head())

if __name__ == "__main__":
    engineer = AQIFeatureEngineer()
    engineer.run()
