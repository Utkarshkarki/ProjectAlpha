import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from src.exception import CustomException
from src.logger import logger
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion

class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')
    max_iter = 200

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def train(self, train_arr: np.ndarray, test_arr: np.ndarray):
        """Train a LogisticRegression model on the provided arrays.
        The last column of each array is assumed to be the target label.
        """
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logger.info("Starting model training using LogisticRegression")
            model = LogisticRegression(max_iter=self.trainer_config.max_iter, n_jobs=-1)
            model.fit(X_train, y_train)

            # Evaluation
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            logger.info(f"Model evaluation - Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")

            # Save model
            os.makedirs(os.path.dirname(self.trainer_config.model_path), exist_ok=True)
            joblib.dump(model, self.trainer_config.model_path)
            logger.info(f"Trained model saved to {self.trainer_config.model_path}")

            return {"model_path": self.trainer_config.model_path, "accuracy": acc, "roc_auc": auc}
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)
    trainer = ModelTrainer()
    result = trainer.train(train_arr, test_arr)
    print(result)
