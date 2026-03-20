import os
import joblib
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logger
from src.components.data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Loaded model and preprocessor successfully.")
        except Exception as e:
            logger.error(f"Failed to load model or preprocessor: {e}")
            raise CustomException(e, sys)

    def predict(self, input_dict: dict) -> str:
        """Accept a dictionary of raw feature values, apply preprocessing, and return the income class.
        Returns '>50K' or '<=50K'.
        """
        try:
            # Convert input dict to DataFrame with a single row
            input_df = pd.DataFrame([input_dict])
            # Apply the same preprocessing pipeline used during training
            processed = self.preprocessor.transform(input_df)
            # Model expects 2D array
            pred = self.model.predict(processed)
            # Decode label (0/1) back to original string using the same encoder logic
            # In training we used LabelEncoder on the target; 0 corresponds to '<=50K', 1 to '>50K'
            return ">50K" if pred[0] == 1 else "<=50K"
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Simple test harness
    sample = {
        "age": 37,
        "workclass": "Private",
        "fnlwgt": 284582,
        "education": "Bachelors",
        "education.num": 13,
        "marital.status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital.gain": 0,
        "capital.loss": 0,
        "hours.per.week": 40,
        "native.country": "United-States"
    }
    pipe = PredictPipeline()
    print(pipe.predict(sample))
