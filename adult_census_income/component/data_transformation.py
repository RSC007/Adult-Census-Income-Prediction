import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline

from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.logger import logging
from adult_census_income.constant import *
from adult_census_income.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from adult_census_income.entity.entity_config import DataTransformationConfig
from adult_census_income.util.util import save_object

class DataTranformation:
    def __init__(self, data_ingestion_artifact_config: DataIngestionArtifact, data_tranformation_config: DataTransformationConfig) -> None:
        try:
            self.data_ingestion_artifact_config=data_ingestion_artifact_config,
            self.data_tranformation_config=data_tranformation_config
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact_config[0].train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact_config[0].test_file_path)

            return train_df, test_df
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    @staticmethod
    def remove_descrete_and_not_need_columns(dataframe):
        try:
            logging.info(f"Removing not need columns...")
            df = dataframe.copy()
            not_need_columns = ["education", "fnlwgt", "marital-status", "capital-gain", "capital-loss"]
            for column in not_need_columns:
                if column in df.columns:
                    df.drop(column, axis=1, inplace=True)

            return df
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    @staticmethod
    def get_labeled_data_frame(df):
        try:
            logging.info(f"Encoding the columns...")
            le = LabelEncoder()
            dataframe = df.copy()
            for column in dataframe.columns:
                dataframe[column] = le.fit_transform(dataframe[column]) 

            return dataframe
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    @staticmethod
    def get_tranform_data(df):
        try:
            # Remove unnessecery columns
            dataframe = DataTranformation.remove_descrete_and_not_need_columns(df)

            # replace the "?"
            dataframe.replace("?", np.NaN, inplace=True)
            # forward fill
            dataframe.fillna(method="ffill", inplace=True)

            # LabelEncoding to convert the sting into numeric for classification
            dataframe = DataTranformation.get_labeled_data_frame(dataframe)
            return dataframe
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def save_transformed_data(self, train_df, test_df):
        try:
            os.makedirs(self.data_tranformation_config.transformed_train_dir, exist_ok=True)
            os.makedirs(self.data_tranformation_config.transformed_test_dir, exist_ok=True)

            transformed_train_file_path = os.path.join(self.data_tranformation_config.transformed_train_dir, "train.csv")
            transformed_test_file_path = os.path.join(self.data_tranformation_config.transformed_test_dir, "test.csv")

            logging.info(f"Saving data...")
            train_df.to_csv(transformed_train_file_path)
            test_df.to_csv(transformed_test_file_path)

            return transformed_train_file_path, transformed_test_file_path
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def initiate_data_tranformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"started data transformation.")
            train_df, test_df = self.get_train_and_test_df()

            preprocessed_object = Pipeline(steps=[
                ("tranformed_data", Encoder(columns=NOT_NEED_COLUMNS))
            ])

            # transormed the data
            train_df = preprocessed_object.fit_transform(train_df)
            test_df = preprocessed_object.fit_transform(test_df)

            # Saving the tranform model 
            save_object(obj=preprocessed_object, file_path=self.data_tranformation_config.preprocessed_object_file_path)

            # save the tranformend data
            train_file_path, test_file_path = self.save_transformed_data(train_df, test_df)
            data_transformed_artifact = DataTransformationArtifact(
                is_transformed=True,
                message=f"Data Transformed completed.",
                transformed_train_file_path=train_file_path,
                transformed_test_file_path=test_file_path
            )

            logging.info(f"data tranformatiom : [{data_transformed_artifact}]")
            return data_transformed_artifact
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n")


class Encoder(BaseEstimator,TransformerMixin):
    def __init__(self, columns=NOT_NEED_COLUMNS) -> None:
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            le = LabelEncoder()
            generated_feature = X.copy()

            for column in generated_feature.columns:
                if column in NOT_NEED_COLUMNS:
                    generated_feature.drop(columns=[column], axis=1, inplace=True)
                else:
                    generated_feature[column] = le.fit_transform(generated_feature[column])

            # replace the "?"
            generated_feature.replace("?", np.NaN, inplace=True)
            # forward fill
            generated_feature.fillna(method="ffill", inplace=True)

            return generated_feature
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e