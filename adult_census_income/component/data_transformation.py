import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.logger import logging
from adult_census_income.constant import *
from adult_census_income.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from adult_census_income.entity.entity_config import DataTransformationConfig

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

    def remove_descrete_and_not_need_columns(self, dataframe):
        try:
            logging.info(f"Removing not need columns...")
            df = dataframe.copy()
            not_need_columns = ["education", "fnlwgt", "marital-status", "capital-gain", "capital-loss"]
            df.drop(not_need_columns, axis=1, inplace=True)

            return df
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def get_labeled_data_frame(self, df):
        try:
            logging.info(f"Encoding the columns...")
            le = LabelEncoder()
            dataframe = df.copy()
            for column in dataframe.columns:
                dataframe[column] = le.fit_transform(dataframe[column]) 

            return dataframe
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def get_tranform_data(self, df):
        try:
            # Remove unnessecery columns
            dataframe = self.remove_descrete_and_not_need_columns(df)

            # replace the "?"
            dataframe.replace("?", np.NaN, inplace=True)
            # forward fill
            dataframe.fillna(method="ffill", inplace=True)

            # LabelEncoding to convert the sting into numeric for classification
            dataframe = self.get_labeled_data_frame(dataframe)
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

            # transormed the data
            train_df = self.get_tranform_data(train_df)
            test_df = self.get_tranform_data(test_df)

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