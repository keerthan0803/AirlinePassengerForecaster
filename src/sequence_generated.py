# Create time-series sequences
"""
====================================================
Module : sequence_generator.py
Project: Airline Passenger Forecasting
Purpose: Generate sequences for RNN/LSTM
====================================================
"""
 
import numpy as np
 
 
class SequenceGenerator:
    """
    Convert a time series into input-output sequences.
    """
 
    def __init__(self, sequence_length=12):
        """
        Parameters
        ----------
        sequence_length : int
            Number of previous time steps used
            to predict the next value.
        """
 
        self.sequence_length = sequence_length
 
    def create_sequences(self, scaled_df):
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be a positive integer, got {self.sequence_length}")

        # Convert dataframe to numpy array
        try:
            data = scaled_df.values
        except Exception as e:
            raise TypeError(f"Expected a DataFrame or array-like object with '.values' attribute. Details: {e}")

        if len(data) <= self.sequence_length:
            raise ValueError(
                f"Dataset length ({len(data)}) is too short for sequence_length ({self.sequence_length}). "
                "More data points are required."
            )

        X = []
        y = []
 
        # Sliding Window
        try:
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:i+self.sequence_length])
                y.append(data[i+self.sequence_length])
        except Exception as e:
            raise RuntimeError(f"Failed to generate sliding window sequences. Details: {e}")
 
        # Convert to NumPy arrays
        X = np.array(X)
        y = np.array(y)
 
        print("\nSequence Generation Completed.")
 
        print(f"Input Shape : {X.shape}")
 
        print(f"Output Shape : {y.shape}")
 
        return X, y
 
if __name__ == "__main__":
 
    from src.data_loader import DataLoader
    from src.preprocessing import Preprocessor
 
    DATA_PATH = "data/airline_passengers.csv"
 
    # Load dataset
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
 
    # Scale dataset
    preprocessor = Preprocessor()
    scaled_df = preprocessor.scale_data(df)
 
    # Generate sequences
    generator = SequenceGenerator(sequence_length=12)
 
    X, y = generator.create_sequences(scaled_df)
 
    print("\nFirst Input Sequence\n")
    print(X[0])
 
    print("\nFirst Target\n")
    print(y[0])
 
# Input Shape : (132, 12, 1)
# Output Shape : (132, 1)
 
# 132   Number of training samples
# 12    Time steps (12 months)
# 1 One feature (Passengers)
 
 