import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import joblib
import os

# Set MLFlow tracking URI to a local sqlite db for persistence or env var
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("School Success Prediction")

def load_and_clean_data():
    # Load ethical data
    try:
        df = pd.read_csv("./data/ethical.csv")
    except FileNotFoundError:
        print("Ethical data not found. Please ensure data/ethical.csv exists.")
        return pd.DataFrame()

    # Drop columns not in the requested input list
    # User requested inputs imply we should drop: school, Medu, Fedu, goout
    # And we MUST KEEP 'source' to get source_por
    # G1 and G2 are restored
    cols_to_drop = ["school", "Medu", "Fedu", "goout"]
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)

    # Clean duplicates just in case
    df = df.drop_duplicates()
            
    return df

def prepare_features(df, target_cols=["G3", "Pass"]):
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Save the columns structure for prediction alignment
    feature_columns = [col for col in df_encoded.columns if col not in target_cols]
    joblib.dump(feature_columns, "./models/model_columns.joblib")
    
    return df_encoded, feature_columns


def train_classification(df, feature_cols):
    # Pass column creation
    df_encoded = df.copy()
    if "Pass" not in df_encoded.columns:
        df_encoded["Pass"] = df_encoded["G3"].apply(lambda x: 1 if x >= 10 else 0)
    
    # Ensure encoding is consistent (we might need to re-encode if we passed raw df, but here we pass encoded)
    # The previous prepare_features returned encoded df. "Pass" might need to be added BEFORE encoding if created from G3?
    # Actually create Pass first.
    
    X = df_encoded[feature_cols]
    y = df_encoded["Pass"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="Random Forest Classification Pass"):
        # Model params from notebook - using RandomForestClassifier as per ethical data analysis
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1-score", f1_score(y_test, predictions))

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 42)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Random Forest Classification - Accuracy: {accuracy}")
        return model

if __name__ == "__main__":
    print("Cleaning data...")
    df = load_and_clean_data()
    
    # Add Pass column before encoding to ensure any potential logic depending on it (none for encoding, but good practice)
    df["Pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    
    df_encoded, feature_columns = prepare_features(df, target_cols=["G3", "Pass"])
    
    clf_model = train_classification(df_encoded, feature_columns)
    
    # Save models locally for FastAPI
    joblib.dump(clf_model, "./models/model.joblib")
    print("Models and columns saved.")
