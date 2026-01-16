import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib
import os

# Set MLFlow tracking URI to a local sqlite db for persistence
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("School Success Prediction")

def load_and_clean_data():
    # Load ethical data
    try:
        df = pd.read_csv("./data/ethical.csv")
    except FileNotFoundError:
        print("Ethical data not found. Please ensure data/ethical.csv exists.")
        return pd.DataFrame()

    # Drop 'source' as it's not a student characteristic we input
    if "source" in df.columns:
        df = df.drop(columns=["source"])

    # Clean duplicates just in case
    df = df.drop_duplicates()
    
    # Remove outliers for G1, G2, G3
    for col_name in ["G1", "G2", "G3"]:
        if col_name in df.columns:
            Q1 = df[col_name].quantile(0.25)
            Q3 = df[col_name].quantile(0.75)
            IQR = Q3 - Q1
            threshold = 1.5
            outliers = df[(df[col_name] < Q1 - threshold * IQR) | (df[col_name] > Q3 + threshold * IQR)]
            df = df.drop(outliers.index)
            
    return df

def prepare_features(df, target_cols=["G3", "Pass"]):
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Save the columns structure for prediction alignment
    feature_columns = [col for col in df_encoded.columns if col not in target_cols]
    joblib.dump(feature_columns, "model_columns.joblib")
    
    return df_encoded, feature_columns

def train_regression(df_encoded, feature_cols):
    X = df_encoded[feature_cols]
    y = df_encoded["G3"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="Linear Regression G3"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "linear_regression_model")
        
        print(f"Regression - RMSE: {rmse}, R2: {r2}")
        return model

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
    
    with mlflow.start_run(run_name="Decision Tree Classification Pass"):
        # Model params from notebook
        model = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='gini')
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("criterion", 'gini')
        mlflow.sklearn.log_model(model, "decision_tree_model")
        
        print(f"Classification - Accuracy: {accuracy}")
        return model

if __name__ == "__main__":
    df = load_and_clean_data()
    
    # Add Pass column before encoding to ensure any potential logic depending on it (none for encoding, but good practice)
    df["Pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    
    df_encoded, feature_columns = prepare_features(df, target_cols=["G3", "Pass"])
    
    reg_model = train_regression(df_encoded, feature_columns)
    clf_model = train_classification(df_encoded, feature_columns)
    
    # Save models locally for FastAPI
    joblib.dump(reg_model, "./models/regression_model.joblib")
    joblib.dump(clf_model, "./models/classification_model.joblib")
    print("Models and columns saved.")
