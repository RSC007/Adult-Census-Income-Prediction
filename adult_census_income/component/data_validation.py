import os
import sys
import numpy as np

from adult_census_income.entity.artifact_entity import DataIngestionArtifact
from adult_census_income.entity.entity_config import DataValidationConfig
from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.constant import *
from adult_census_income.logger import logging
from adult_census_income.util.util import read_csv_file

class DataValidation:
    def __init__(self, data_ingestion_artifact_config: DataIngestionArtifact, data_validation_config: DataValidationConfig) -> None:
        try:
            self.data_ingestion_artifact_config=data_ingestion_artifact_config
            self.data_validaiton_config=data_validation_config
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def is_train_test_file_exists(self) -> bool:
        try:
            logging.info(f"Checking is training and testing file is available")
            is_train_file_exist=False
            is_test_file_exist=False

            is_test_file_exist = os.path.exists(self.data_ingestion_artifact_config.test_file_path)
            is_train_file_exist = os.path.exists(self.data_ingestion_artifact_config.train_file_path)

            is_available = is_train_file_exist and is_test_file_exist

            if not is_available:
                test_file_path = self.data_ingestion_artifact_config.test_file_path
                train_file_path = self.data_ingestion_artifact_config.train_file_path
                message=f"Training file: {train_file_path} or Testing file: {test_file_path} not present"
                raise Exception(message)

            return is_available
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def remove_descrete_and_not_need_columns(self, dataframe):
        try:
            df = dataframe.copy()
            not_need_columns = ["education", "fnlwgt", "marital-status", "capital-gain", "capital-loss"]
            df.drop(not_need_columns, axis=1, inplace=True)
            logging.info(f"Remove not need colimns: [{not_need_columns}]")
            return df

        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def validate_dataset_schema(self):
        try:
            validation_status = False

            # remove the columns
            train_df = read_csv_file(self.data_ingestion_artifact_config.train_file_path)
            train_df = self.remove_descrete_and_not_need_columns(dataframe=train_df)

            # replace the "?"
            train_df.replace("?", np.NaN, inplace=True)
            # forward fill
            train_df.fillna(method="ffill", inplace=True)

            logging.info(f"Data Validation is complete")
            return validation_status
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def initiate_data_validation(self):
        try:
            self.is_train_test_file_exists()
            self.validate_dataset_schema()
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e