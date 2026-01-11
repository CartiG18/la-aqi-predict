import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class AQIPredictorTrainer:
    def __init__(self, input_path: Path = None, model_path: Path = None):
        self.input_path = input_path or PROJECT_ROOT / "data" / "processed" / "processed_la_aqi.csv"
        self.model_path = model_path or PROJECT_ROOT / "src" / "models" / "aqi_model.joblib"
        
    def load_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Processed data not found at {self.input_path}")
        return pd.read_csv(self.input_path)

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into features (X) and target (y).
        Target: pm2_5
        Features: temperature, humidity, temp_humidity_interaction
        """
        features = ['temperature', 'humidity', 'temp_humidity_interaction']
        target = 'pm2_5'
        
        X = df[features]
        y = df[target]
        
        # Split 80/20
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        print("Training Random Forest Regressor...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        print("\n--- Model Evaluation ---")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R^2:  {r2:.4f}")
        
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def save_model(self, model):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        print(f"\nModel saved to {self.model_path}")

    def run(self):
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        model = self.train(X_train, y_train)
        self.evaluate(model, X_test, y_test)
        self.save_model(model)

if __name__ == "__main__":
    trainer = AQIPredictorTrainer()
    trainer.run()
