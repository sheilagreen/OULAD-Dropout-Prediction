FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY src/ ./src/
COPY models/ ./models/

# Expose the API port
EXPOSE 5000

# Run the Flask API
CMD ["python", "src/app.py"]
