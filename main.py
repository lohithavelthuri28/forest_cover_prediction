import os
import joblib
import pandas as pd
from src.data_preprocessing import load_and_preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model

DATA_PATH = "data/train.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # Load & preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = \
        load_and_preprocess_data(DATA_PATH)

    # Train model
    model = train_model(X_train, y_train)

    # Save model artifacts
    joblib.dump(model, f"{MODEL_DIR}/forest_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.pkl")

    # --- NEW: Save feature means ---
    df = pd.read_csv(DATA_PATH)
    feature_means = df.drop("Cover_Type", axis=1).mean()
    joblib.dump(feature_means.to_dict(), f"{MODEL_DIR}/feature_means.pkl")

    # Evaluate
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
