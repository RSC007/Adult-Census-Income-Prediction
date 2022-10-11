import json
import os
import sys
import numpy as np
import pandas as pd
from evidently import dashboard
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

from adult_census_income.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
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
            logging.info(f"Remove not need columns: [{not_need_columns}]")
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

    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact_config.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact_config.test_file_path)

            return train_df, test_df
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def get_and_save_data_drift_report(self):
        try:
            self.is_train_test_file_exists()
            self.validate_dataset_schema()

            profile = Profile(sections=[DataDriftProfileSection()])

            train_df, test_df = self.get_train_and_test_df()

            profile.calculate(train_df, test_df)

            report = json.loads(profile.json())

            report_file_path = self.data_validaiton_config.report_file_path

            report_dir = os.path.dirname(report_file_path)

            os.makedirs(report_dir, exist_ok=True)

            with open(report_file_path, 'w') as report_file:
                json.dump(report, report_file, indent=6)

            return report
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def save_data_drift_report_page(self):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df, test_df = self.get_train_and_test_df()
            dashboard.calculate(train_df, test_df)

            report_page_file_path = self.data_validaiton_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)

            dashboard.save(report_page_file_path)
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def is_data_drift_found(self):
        try:
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def initiate_data_validation(self):
        try:
            self.is_train_test_file_exists()
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validaiton_config.schema_file_path,
                report_file_path=self.data_validaiton_config.report_file_path,
                report_page_file_path=self.data_validaiton_config.report_page_file_path,
                is_validated=True,
                message="Data validation perform successfully"
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")