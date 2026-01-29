import pandas as pd
from . import schemas


def preprocess_input(data: schemas.StudentInput, columns: list):
    """Preprocess input data and align with training columns."""
    df = pd.DataFrame([data.dict()])
    
    # Categorical Columns to encode (Must match those in training)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Align columns
    # Add missing cols with 0
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Keep only model cols and ensure order
    df_final = df_encoded[columns]
    
    return df_final
