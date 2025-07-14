import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logger
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
            categorical_columns = [
                'workclass', 'education', 'marital.status', 'occupation', 
                'relationship', 'race', 'sex', 'native.country'
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")
            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")

            logger.info("Converting '?' to NaN for proper missing value handling")
            train_df = train_df.replace('?', np.nan)
            test_df = test_df.replace('?', np.nan)

            string_columns = train_df.select_dtypes(include=['object']).columns
            for col in string_columns:
                train_df[col] = train_df[col].astype(str).str.strip()
                test_df[col] = test_df[col].astype(str).str.strip()

            logger.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'income'
            if target_column_name not in train_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in training data")

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logger.info(f"Transformed train features shape: {input_feature_train_arr.shape}")
            logger.info(f"Transformed test features shape: {input_feature_test_arr.shape}")

            label_encoder = LabelEncoder()
            target_feature_train_encoded = label_encoder.fit_transform(target_feature_train_df).reshape(-1, 1)
            target_feature_test_encoded = label_encoder.transform(target_feature_test_df).reshape(-1, 1)

            logger.info(f"Reshaped target train shape: {target_feature_train_encoded.shape}")
            logger.info(f"Reshaped target test shape: {target_feature_test_encoded.shape}")

            if input_feature_train_arr.shape[0] != target_feature_train_encoded.shape[0]:
                raise ValueError(
                    f"Shape mismatch: Features {input_feature_train_arr.shape}, Target {target_feature_train_encoded.shape}"
                )

            train_arr = np.c_[input_feature_train_arr, target_feature_train_encoded]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_encoded]

            logger.info(f"Final train array shape: {train_arr.shape}")
            logger.info(f"Final test array shape: {test_arr.shape}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logger.error(f"Error occurred during data transformation: {str(e)}")
            raise CustomException(e, sys)