# Dockerfile
# This file tells Docker how to build a container
# that runs our FastAPI backend

FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker caching optimization)
# If requirements.txt doesn't change, Docker reuses this layer
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Tell Docker which port the app uses
EXPOSE 8000

# Command to run when container starts
CMD ["python", "run_step7.py"]
