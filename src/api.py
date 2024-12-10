from fastapi import FastAPI, UploadFile
from main import DataIngestion, DataTransformation, ModelTrainer
import os
import pandas as pd

app = FastAPI()

# Initialize your pipeline classes
data_ingestion = DataIngestion(data_path="src/dataset/loan_approval_dataset.csv")
data_transformation = DataTransformation()
model_trainer = ModelTrainer()

# Endpoint for data ingestion
@app.post("/ingest-data/")
async def ingest_data(file: UploadFile):
    file_location = f"uploaded_files/{file.filename}"
    os.makedirs("uploaded_files", exist_ok=True)
    
    # Save uploaded file
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Set raw data path for ingestion
    data_ingestion.set_data_path(file_location)
    
    # Call data ingestion
    raw_data = data_ingestion.load_data(data_ingestion.raw_data_path)
    return {"message": "Data ingestion completed", "raw_data_sample": raw_data.head(5).to_dict()}

# Endpoint for data transformation
@app.post("/transform-data/")
def transform_data():
    data_path = data_ingestion.raw_data_path
    raw_data = pd.read_csv(data_path)

    # Perform data transformation
    transformed_data = data_transformation.transform(raw_data)
    transformed_data_path = "transformed_data.csv"
    transformed_data.to_csv(transformed_data_path, index=False)
    
    return {"message": "Data transformation completed", "transformed_data_sample": transformed_data.head(5).to_dict()}

# Endpoint for model training
@app.post("/train-model/")
def train_model():
    """
    Endpoint for training a model using the transformed data.
    Ensure the transformed data exists before calling this endpoint.
    """
    model_trainer.train()
    return {"message": "Model training completed", "model_metrics": model_trainer.metrics}
