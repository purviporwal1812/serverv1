# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /serverv1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your app code and models into the container
COPY . .

# Expose the port Flask will use
EXPOSE 10000

# Run the app using Gunicorn (1 worker to stay within memory limits)
CMD ["sh", "-c", "gunicorn --timeout 600 --workers 1 --preload --bind 0.0.0.0:${PORT} app:app"]
