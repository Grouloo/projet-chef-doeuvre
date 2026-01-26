FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY models /app/models
COPY data /app/data
COPY logs /app/logs
COPY app /app/app
COPY train.py /app/train.py
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini

# Copy and setup entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose the port
EXPOSE 8000

# Run the application
CMD ["./entrypoint.sh"]
