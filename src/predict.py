# # Predictions
 
"""
====================================================
Module : predict.py
Project: Airline Passenger Forecasting
Purpose: Predict Passenger Counts
====================================================
"""
 
import joblib
import numpy as np
 
from tensorflow.keras.models import load_model
 
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.sequence_generated import SequenceGenerator
from src.train_test_split import TimeSeriesSplit
 
 
class Predictor:
 
    def __init__(self):
 
        self.data_path = "data/airline_passengers.csv"
 
        self.model_path = "models/lstm_model.keras"
 
        self.scaler_path = "models/scaler.pkl"
 
    def predict(self):
        import os

        # ----------------------------
        # Validate File Existence
        # ----------------------------
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{self.model_path}'. "
                "Please run model training first (e.g. python -m src.train)."
            )

        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Data scaler not found at '{self.scaler_path}'. "
                "Please run preprocessing first."
            )

        # ----------------------------
        # Load Dataset & Preprocess
        # ----------------------------
        try:
            loader = DataLoader(self.data_path)
            df = loader.load_data()
            
            preprocessor = Preprocessor()
            scaled_df = preprocessor.scale_data(df)
            
            generator = SequenceGenerator(sequence_length=12)
            X, y = generator.create_sequences(scaled_df)
            
            splitter = TimeSeriesSplit(train_size=0.80)
            X_train, X_test, y_train, y_test = splitter.split(X, y)
        except Exception as e:
            raise RuntimeError(f"Error preparing data for predictions: {e}")

        # ----------------------------
        # Load Model
        # ----------------------------
        try:
            model = load_model(self.model_path)
            print("\nModel Loaded Successfully.")
        except Exception as e:
            raise IOError(f"Failed to load Keras model from '{self.model_path}'. Details: {e}")

        # ----------------------------
        # Load Scaler
        # ----------------------------
        try:
            scaler = joblib.load(self.scaler_path)
            print("Scaler Loaded Successfully.")
        except Exception as e:
            raise IOError(f"Failed to load scaler from '{self.scaler_path}'. Details: {e}")

        # ----------------------------
        # Predict
        # ----------------------------
        try:
            predictions = model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"Failed to make predictions on test features. Details: {e}")

        # ----------------------------
        # Convert back to original scale
        # ----------------------------
        try:
            predictions = scaler.inverse_transform(predictions)
            y_test = scaler.inverse_transform(y_test)
        except Exception as e:
            raise ValueError(f"Failed to perform inverse scale transform on predictions. Details: {e}")

        print("\nPrediction Completed.")

        return y_test, predictions
if __name__ == "__main__":
 
    predictor = Predictor()
 
    actual, predicted = predictor.predict()
 
    print("\nFirst 10 Predictions\n")
 
    for i in range(10):
 
        print(
            f"Actual : {actual[i][0]:.2f}   "
            f"Predicted : {predicted[i][0]:.2f}"
        )
 