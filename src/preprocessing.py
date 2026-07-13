# Scaling & preprocessing
 
"""
====================================================
Module : preprocessing.py
Project: Airline Passenger Forecasting
Purpose: Scale the dataset using MinMaxScaler
====================================================
"""
 
# Import required libraries
import os
import joblib
import pandas as pd
 
from sklearn.preprocessing import MinMaxScaler
 
 
class Preprocessor:
    """
    Preprocess the time series dataset.
    """
 
    def __init__(self):
        """
        Initialize the scaler.
        """
 
        self.scaler = MinMaxScaler(feature_range=(0, 1))
 
    def scale_data(self, df):
        """
        Scale the Passengers column.
 
        Parameters
        ----------
        df : pandas.DataFrame
 
        Returns
        -------
        scaled_df : pandas.DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
 
        if "Passengers" not in df.columns:
            raise KeyError("The input DataFrame must contain a 'Passengers' column to be scaled.")
 
        print("\nOriginal Data")
        print(df.head())
 
        # Scale the Passengers column
        try:
            scaled_values = self.scaler.fit_transform(df[["Passengers"]])
        except Exception as e:
            raise ValueError(f"Failed to fit and transform the data using MinMaxScaler. Details: {e}")
 
        # Convert to DataFrame
        scaled_df = pd.DataFrame(
            scaled_values,
            columns=["Passengers"],
            index=df.index
        )
 
        # print("\nScaled Data")
        # print(scaled_df.head())
 
        # Save the scaler
        try:
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.scaler, "models/scaler.pkl")
        except Exception as e:
            raise IOError(f"Failed to save scaler file to 'models/scaler.pkl'. Details: {e}")
 
        print("\nScaler saved successfully.")
 
        return scaled_df
 
if __name__ == "__main__":
 
    from src.data_loader import DataLoader
 
    DATA_PATH = "data/airline_passengers.csv"
 
    # Load data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
 
    # Scale data
    preprocessor = Preprocessor()
 
    scaled_df = preprocessor.scale_data(df)
 
    print("\nScaled Dataset")
    print(scaled_df.head())
 