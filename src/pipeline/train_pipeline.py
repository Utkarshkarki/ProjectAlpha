import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training_pipeline():
    """Orchestrates the full training pipeline.
    Steps:
    1. Ingest raw data and split into train/test CSVs.
    2. Transform the CSVs into feature arrays and save the preprocessor.
    3. Train a model on the transformed data and persist it.
    """
    try:
        # Step 1: Data ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        
        # Step 2: Data transformation
        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)
        
        # Step 3: Model training
        trainer = ModelTrainer()
        result = trainer.train(train_arr, test_arr)
        
        print("Training pipeline completed. Results:")
        print(result)
        print(f"Preprocessor saved at: {preprocessor_path}")
    except Exception as e:
        print(f"Error in training pipeline: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    run_training_pipeline()
