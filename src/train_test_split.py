"""
====================================================
Module : train_test_split.py
Project: Airline Passenger Forecasting
Purpose: Split sequences into Train and Test sets
====================================================
"""
 
class TimeSeriesSplit:
 
    def __init__(self, train_size=0.8):
        """
        Parameters
        ----------
        train_size : float
            Percentage of data used for training.
        """
 
        self.train_size = train_size
 
    def split(self, X, y):
        if not (0 < self.train_size < 1):
            raise ValueError(f"train_size must be between 0 and 1 (exclusive), got {self.train_size}")

        if len(X) != len(y):
            raise ValueError(f"Inputs X (length {len(X)}) and targets y (length {len(y)}) must have the same length.")

        if len(X) == 0:
            raise ValueError("Inputs X and y are empty. Cannot split empty sequences.")

        # Calculate split index
        split_index = int(len(X) * self.train_size)
        if split_index == 0 or split_index == len(X):
            raise ValueError(
                f"Split index calculated as {split_index} leaves either train or test empty. "
                f"Data size: {len(X)}, train_size: {self.train_size}"
            )
 
        # Split input sequences
        X_train = X[:split_index]
        X_test = X[split_index:]
 
        # Split target values
        y_train = y[:split_index]
        y_test = y[split_index:]
 
        print("\nTrain-Test Split Completed.")
 
        print(f"\nTraining Samples : {len(X_train)}")
        print(f"Testing Samples  : {len(X_test)}")
 
        print("\nTraining Shape")
 
        print(X_train.shape)
 
        print("\nTesting Shape")
 
        print(X_test.shape)
 
        return X_train, X_test, y_train, y_test
if __name__ == "__main__":
 
    from src.data_loader import DataLoader
    from src.preprocessing import Preprocessor
    from src.sequence_generated import SequenceGenerator
 
    DATA_PATH = "data/airline_passengers.csv"
 
    # Load Dataset
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
 
    # Scale Dataset
    preprocessor = Preprocessor()
    scaled_df = preprocessor.scale_data(df)
 
    # Generate Sequences
    generator = SequenceGenerator(sequence_length=12)
 
    X, y = generator.create_sequences(scaled_df)
 
    # Train-Test Split
    splitter = TimeSeriesSplit(train_size=0.8)
 
    X_train, X_test, y_train, y_test = splitter.split(X, y)
 
    print("\nFirst Training Sample")
 
    print(X_train[0])
 
    print("\nFirst Training Target")
 
    print(y_train[0])
 