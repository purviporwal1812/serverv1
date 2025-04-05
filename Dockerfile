# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /serverv1

# Install system dependencies: wget for downloading and unzip for extracting
RUN apt-get update && apt-get install -y wget unzip && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download the Kaggle dataset using the public URL and extract it.
# This will download the zip file and extract its contents into the 'models' directory.
RUN wget -O dataset.zip "https://www.kaggle.com/api/v1/datasets/download/rishitasharma999/major-project-server?" \
    && unzip dataset.zip -d models \
    && rm dataset.zip

# Set an environment variable for the model path (adjust if needed)
ENV MODEL_PATH=models/final_model.keras

# Copy the rest of your application code into the container
COPY . .

# Expose the port your Flask app listens on
EXPOSE 10000

# Run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
