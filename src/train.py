"""
====================================================
Module : train.py
Project: Airline Passenger Forecasting
Purpose: Train the LSTM Model
====================================================
"""
 
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.sequence_generated import SequenceGenerator
from src.train_test_split import TimeSeriesSplit
from src.model import ModelBuilder
 
 
class ModelTrainer:
 
    def __init__(self):
 
        self.data_path = "data/airline_passengers.csv"
 
    def train(self):
        try:
            # ----------------------------
            # Step 1 : Load Dataset
            # ----------------------------
            loader = DataLoader(self.data_path)
            df = loader.load_data()
        except Exception as e:
            raise RuntimeError(f"Model training failed at Step 1 (Data Loading): {e}")

        try:
            # ----------------------------
            # Step 2 : Preprocess
            # ----------------------------
            preprocessor = Preprocessor()
            scaled_df = preprocessor.scale_data(df)
        except Exception as e:
            raise RuntimeError(f"Model training failed at Step 2 (Data Preprocessing): {e}")

        try:
            # ----------------------------
            # Step 3 : Generate Sequences
            # ----------------------------
            generator = SequenceGenerator(sequence_length=12)
            X, y = generator.create_sequences(scaled_df)
        except Exception as e:
            raise RuntimeError(f"Model training failed at Step 3 (Sequence Generation): {e}")

        try:
            # ----------------------------
            # Step 4 : Train Test Split
            # ----------------------------
            splitter = TimeSeriesSplit(train_size=0.80)
            X_train, X_test, y_train, y_test = splitter.split(X, y)
        except Exception as e:
            raise RuntimeError(f"Model training failed at Step 4 (Train-Test Split): {e}")

        try:
            # ----------------------------
            # Step 5 : Build Model
            # ----------------------------
            builder = ModelBuilder(
                model_type="lstm",
                input_shape=(12,1)
            )
            model = builder.build_model()
        except Exception as e:
            raise RuntimeError(f"Model training failed at Step 5 (Model Building): {e}")

        try:
            # ----------------------------
            # Step 6 : Train Model
            # ----------------------------
            print("\nTraining Started...\n")
            history = model.fit(
                X_train,
                y_train,
                epochs=100,
                batch_size=8,
                validation_data=(X_test,y_test),
                verbose=1
            )
            print("\nTraining Completed Successfully.")
        except Exception as e:
            raise RuntimeError(f"Model training failed at Step 6 (Model Training): {e}")

        try:
            # ----------------------------
            # Step 7 : Save Model
            # ----------------------------
            import os
            os.makedirs("models", exist_ok=True)
            model.save("models/lstm_model.keras")
            print("\nModel Saved Successfully.")
        except Exception as e:
            raise RuntimeError(f"Model training failed at Step 7 (Model Saving): {e}")

        return model, history
   
if __name__ == "__main__":
 
    trainer = ModelTrainer()
 
    model, history = trainer.train()
 