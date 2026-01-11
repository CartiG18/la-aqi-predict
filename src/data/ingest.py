import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

class SensorData(BaseModel):
    sensor_index: int
    pm2_5: float = Field(alias="pm2.5_10minute")
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    latitude: float
    longitude: float

class PurpleAirLoader:
    """
    Robust data loader for PurpleAir API.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PURPLE_AIR_API_KEY")
        if not self.api_key:
            raise ValueError("PurpleAir API key is missing. Set PURPLE_AIR_API_KEY in .env")
        
        self.base_url = "https://api.purpleair.com/v1/sensors"
        self.headers = {"X-API-Key": self.api_key}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def fetch_data(self, bbox: List[float] = [-118.9, 34.3, -118.1, 33.7]) -> pd.DataFrame:
        """
        Fetches sensor data with retries.
        bbox: [nwlng, nwlat, selng, selat]
        """
        params = {
            "fields": "pm2.5_10minute,temperature,humidity,latitude,longitude",
            "nwlng": bbox[0], "nwlat": bbox[1],
            "selng": bbox[2], "selat": bbox[3]
        }
        
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'data' not in data or 'fields' not in data:
                raise ValueError("Unexpected API response structure")
                
            cols = data['fields']
            rows = data['data']
            
            # Validate and Parse
            validated_rows = []
            for row in rows:
                record = dict(zip(cols, row))
                try:
                    # Validate row using Pydantic
                    # Pydantic will strip extra fields if not configured, but here we explicitly select
                    validated_data = SensorData(**record)
                    validated_rows.append(validated_data.model_dump(by_alias=True))
                except ValidationError as e:
                    # Log invalid rows if necessary, skip for now to ensure robustness
                    continue
            
            if not validated_rows:
                print("Warning: No valid data points found after validation.")
                return pd.DataFrame()

            df = pd.DataFrame(validated_rows)
            return df

        except requests.RequestException as e:
            print(f"API Request failed: {e}")
            raise

    def save_raw(self, df: pd.DataFrame, filename: str = "raw_la_aqi.csv"):
        """Saves raw data to the data/raw directory."""
        output_path = PROJECT_ROOT / "data" / "raw" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")

if __name__ == "__main__":
    try:
        loader = PurpleAirLoader()
        print("Fetching PurpleAir data...")
        df = loader.fetch_data()
        if not df.empty:
            loader.save_raw(df)
            print(df.head())
    except Exception as e:
        print(f"Ingestion failed: {e}")
