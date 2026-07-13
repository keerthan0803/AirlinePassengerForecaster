# RMSE, MAE, MSE
 
"""
====================================================
Module : evaluate.py
Project: Airline Passenger Forecasting
Purpose: Evaluate LSTM Model Performance
====================================================
"""
 
import numpy as np
import os
 
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
 
from src.predict import Predictor
 
 
class Evaluator:
 
    """
    Evaluate model performance.
    """
 
    def __init__(self):
        pass
 
    def evaluate(self):
 
        # -----------------------------
        # Generate Predictions
        # -----------------------------
        try:
            predictor = Predictor()
            actual, predicted = predictor.predict()
        except Exception as e:
            raise RuntimeError(f"Failed to obtain predictions for evaluation. Details: {e}")
 
        # Validate inputs
        if len(actual) == 0 or len(predicted) == 0:
            raise ValueError("Cannot evaluate: actual values or predicted values are empty.")
            
        if len(actual) != len(predicted):
            raise ValueError(f"Mismatched data lengths. Actuals: {len(actual)}, Predictions: {len(predicted)}")

        # -----------------------------
        # Calculate Metrics
        # -----------------------------
        try:
            mae = mean_absolute_error(
                actual,
                predicted
            )
     
            mse = mean_squared_error(
                actual,
                predicted
            )
     
            rmse = np.sqrt(mse)
        except Exception as e:
            raise ValueError(f"Failed to calculate metric values. Details: {e}")
 
        print("\nModel Evaluation")
 
        print(f"\nMAE  : {mae:.4f}")
 
        print(f"MSE  : {mse:.4f}")
 
        print(f"RMSE : {rmse:.4f}")
 
        # -----------------------------
        # Save Metrics
        # -----------------------------
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            outputs_dir = os.path.join(project_root, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
     
            with open(
                os.path.join(outputs_dir, "metrics.txt"),
                "w"
            ) as file:
     
                file.write("Model Evaluation\n")
     
                file.write("=====================\n")
     
                file.write(f"MAE  : {mae:.4f}\n")
     
                file.write(f"MSE  : {mse:.4f}\n")
     
                file.write(f"RMSE : {rmse:.4f}\n")
        except Exception as e:
            raise IOError(f"Failed to write evaluation report file. Details: {e}")
 
        print("\nMetrics Saved Successfully.")
 
        return mae, mse, rmse
   
if __name__ == "__main__":
 
    evaluator = Evaluator()
 
    mae, mse, rmse = evaluator.evaluate()
