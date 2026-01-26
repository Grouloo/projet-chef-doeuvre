#!/bin/sh
set -e

# Initialize database
echo "Running database migrations..."
alembic upgrade head

# Train model
echo "Training model..."
python train.py

# Start application
echo "Starting application..."
uvicorn app.main:app --host 0.0.0.0 --port 8000