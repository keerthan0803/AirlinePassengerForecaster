# Future forecasting
 
"""
====================================================
Module : forecast.py
Project: Airline Passenger Forecasting
Purpose: Forecast Future Passenger Counts
====================================================
"""
 
import numpy as np
import joblib
 
from tensorflow.keras.models import load_model
 
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
 
 
class Forecaster:
 
    def __init__(self):
 
        self.data_path = "data/airline_passengers.csv"
 
        self.model_path = "models/lstm_model.keras"
 
        self.scaler_path = "models/scaler.pkl"
 
        self.sequence_length = 12
 
    def forecast(self, future_months=12):
        import os

        if future_months <= 0:
            raise ValueError(f"future_months must be a positive integer, got {future_months}")

        # --------------------------
        # Validate File Existence
        # --------------------------
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{self.model_path}'. "
                "Please run model training first."
            )

        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Data scaler not found at '{self.scaler_path}'. "
                "Please run preprocessing first."
            )

        # --------------------------
        # Load Dataset & Preprocess
        # --------------------------
        try:
            loader = DataLoader(self.data_path)
            df = loader.load_data()
            
            preprocessor = Preprocessor()
            scaled_df = preprocessor.scale_data(df)
        except Exception as e:
            raise RuntimeError(f"Error preparing data for forecasting: {e}")

        # --------------------------
        # Load Model
        # --------------------------
        try:
            model = load_model(self.model_path)
        except Exception as e:
            raise IOError(f"Failed to load Keras model from '{self.model_path}'. Details: {e}")

        # --------------------------
        # Load Scaler
        # --------------------------
        try:
            scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise IOError(f"Failed to load scaler from '{self.scaler_path}'. Details: {e}")

        # --------------------------
        # Last 12 Months
        # --------------------------
        if len(scaled_df) < self.sequence_length:
            raise ValueError(
                f"Data length ({len(scaled_df)}) is less than required sequence length ({self.sequence_length}) "
                "to construct historical seed sequence."
            )

        last_sequence = scaled_df.values[-self.sequence_length:]
 
        future_predictions = []
 
        # --------------------------
        # Forecast Loop
        # --------------------------
        try:
            for _ in range(future_months):
 
                input_data = last_sequence.reshape(
                    1,
                    self.sequence_length,
                    1
                )
 
                prediction = model.predict(
                    input_data,
                    verbose=0
                )
 
                future_predictions.append(prediction[0,0])
 
                last_sequence = np.vstack(
                    (
                        last_sequence[1:],
                        prediction
                    )
                )
        except Exception as e:
            raise RuntimeError(f"Failed during prediction forecasting loop. Details: {e}")
 
        # --------------------------
        # Convert Back
        # --------------------------
        try:
            future_predictions = np.array(
                future_predictions
            ).reshape(-1,1)
 
            future_predictions = scaler.inverse_transform(
                future_predictions
            )
        except Exception as e:
            raise ValueError(f"Failed to inverse transform forecasted values. Details: {e}")
 
        print("\nFuture Forecast Completed.")
 
        return future_predictions
if __name__ == "__main__":
 
    forecaster = Forecaster()
 
    future = forecaster.forecast(future_months=12)
 
    print("\nNext 12 Month Forecast\n")
 
    for i, value in enumerate(future, start=1):
 
        print(f"Month {i} : {value[0]:.2f}")
   
 