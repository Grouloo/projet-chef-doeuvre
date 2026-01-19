import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
import sys
from sqlalchemy.orm import Session
from . import models, schemas, database
import subprocess

# Setup Logger
logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add("./logs/api.log", rotation="10 MB")





app = FastAPI(title="Prédiction de la réussite des élèves")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Global variables for models
reg_model = None
clf_model = None
model_cols = None

@app.on_event("startup")
def load_artifacts():
    global clf_model, model_cols
    try:
        if os.path.exists("./models/model.joblib"):
            clf_model = joblib.load("./models/model.joblib")
        if os.path.exists("./models/model_columns.joblib"):
            model_cols = joblib.load("./models/model_columns.joblib")
        logger.info("Models and artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

# Helper: Preprocess Input
def preprocess_input(data: schemas.StudentInput, columns: list):
    df = pd.DataFrame([data.dict()])
    
    # Categorical Columns to encode (Must match those in training)
    # Ideally we should infer this, but for now we follow the schema types
    # Since get_dummies is dynamic, we just apply it and align.
    
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

# Endpoints

@app.get("/health")
def health_check():
    status = {
        "status": "ok",
        "models_loaded": {
            "classification": clf_model is not None,
            "columns": model_cols is not None
        }
    }
    return status

@app.post("/predict", response_model=schemas.PredictionOutput)
def predict(input_data: schemas.StudentInput, db: Session = Depends(database.get_db)):
    if not clf_model or not model_cols:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Preprocess
        processed_data = preprocess_input(input_data, model_cols)
        
        # Predict
        # Classification -> Pass/Fail
        pass_prob = clf_model.predict_proba(processed_data)[0]
        pass_pred = clf_model.predict(processed_data)[0]
        
        label = "Pass" if pass_pred == 1 else "Fail"
        
        result = schemas.PredictionOutput(
            prediction_score=None,
            prediction_label=label,
            probabilities={"fail": pass_prob[0], "pass": pass_prob[1]},
            model_type="classification"
        )
        
        # Log to DB
        log_entry = models.PredictionLog(
            inputs=input_data.dict(),
            prediction=result.dict()
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        
        logger.info(f"Prediction made: {result.dict()}")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/retrain")
def trigger_training(background_tasks: BackgroundTasks):
    def run_training_script():
        logger.info("Starting training script...")
        try:
            subprocess.run(["python3", "train.py"], check=True)
            # Re-load models logic could be added here
            load_artifacts() 
            logger.info("Training finished and models reloaded.")
        except Exception as e:
            logger.error(f"Training failed: {e}")

    background_tasks.add_task(run_training_script)
    return {"message": "Training started in background"}

# Web Interface Routes

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/logs", response_class=HTMLResponse)
async def read_logs(request: Request, db: Session = Depends(database.get_db)):
    logs = db.query(models.PredictionLog).order_by(models.PredictionLog.timestamp.desc()).limit(50).all()
    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs})

@app.get("/logs.json")
async def read_logs_json(db: Session = Depends(database.get_db)):
    logs = db.query(models.PredictionLog).order_by(models.PredictionLog.timestamp.desc()).limit(50).all()
    return logs

@app.get("/dataset", response_class=HTMLResponse)
async def view_dataset(request: Request):
    # Load dataset sample
    try:
        df = pd.read_csv("./data/ethical.csv")
        data_preview = df.head(20).to_dict(orient="records")
        columns = df.columns.tolist()
    except:
        data_preview = []
        columns = []
        
    return templates.TemplateResponse("dataset.html", {"request": request, "data": data_preview, "columns": columns})

@app.get("/dataset.json")
async def view_dataset_json():
    try:
        df = pd.read_csv("./data/ethical.csv")
        return df.to_dict(orient="records")
    except:
        return {"error": "Dataset not found"}
