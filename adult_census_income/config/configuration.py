import os
import sys

from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.logger import logging
from adult_census_income.constant import *
from adult_census_income.util.util import read_yaml_file

from adult_census_income.entity.entity_config import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, TrainingPipelineConfig


class Configuration:

    def __init__(self,
                 config_file_path: str = CONFIG_FILE_PATH,
                 current_time_stamp: str = CURRENT_TIME_STAMP
                 ) -> None:
        try:
            self.config_info = read_yaml_file(file_path=config_file_path)
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = current_time_stamp
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            logging.info(f"Data ingestion configuration started...")
            artifact_dir = self.training_pipeline_config.artifact_dir
            
            data_ingestion_config = self.config_info[DATA_INGESTION_CONFIG_KEY]

            raw_data_dir = os.path.join(artifact_dir, data_ingestion_config[DATA_INGESTION_ARTIFACT_DIR_KEY], self.time_stamp, data_ingestion_config[DATA_INGESTION_RAW_DIR_KEY])

            ingestion_train_dir = os.path.join(raw_data_dir, data_ingestion_config[DATA_INGESTION_TRAIN_DIR_KEY])

            ingestion_test_dir = os.path.join(raw_data_dir, data_ingestion_config[DATA_INGESTION_TEST_DIR_KEY])

            data_ingestion_config = DataIngestionConfig(
                raw_data_dir=raw_data_dir,
                ingestion_train_dir=ingestion_train_dir,
                ingestion_test_dir=ingestion_test_dir
            )

            logging.info(f"Data Ingestion Completed: {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def get_data_validation_config(self):
        try:
            logging.info(f"="*100)
            logging.info(f"Data Validation configuration started...")

            artifact_dir = self.training_pipeline_config.artifact_dir

            data_validation_artifact_dir = os.path.join(
                artifact_dir,
                DATA_VALIDATION_ARTIFACT_DIR_NAME,
                self.time_stamp
            )

            data_validation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY]


            schema_file_path = os.path.join(
                ROOT_DIR,
                data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],
                data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]
            )

            report_file_path = os.path.join(
                data_validation_artifact_dir,
                data_validation_config[DATA_VALIDATION_REPORT_FILE_NAME_KEY]
            )

            report_page_file_path = os.path.join(
                data_validation_artifact_dir,
                data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY]
            )

            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path,
                report_file_path=report_file_path,
                report_page_file_path=report_page_file_path
            )

            logging.info(f"Data Validation complete: {data_validation_config}")
            return data_validation_config
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e


    def get_data_trasformation_config(self):
        try:
            logging.info(f"="*100)
            logging.info(f"Data Tranform configuration started...")

            artifact_dir = self.training_pipeline_config.artifact_dir
            data_transformation_config = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

            data_transformation_artifact_dir = os.path.join(
                artifact_dir,
                DATA_TRANSFORMATION_ARTIFACT_DIR,
                self.time_stamp
            )

            transformed_train_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config[DATA_TRANSFORMATION_DIR_NAME_KEY],
                data_transformation_config[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY]
            )

            transformed_test_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config[DATA_TRANSFORMATION_DIR_NAME_KEY],
                data_transformation_config[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY]
            )

            preprocessed_object_file_path = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                data_transformation_config[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]
            )

            data_transformation_config = DataTransformationConfig(
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir,
                preprocessed_object_file_path=preprocessed_object_file_path
            )
            logging.info(f"Data Tranformation Config: {data_transformation_config}")
            return data_transformation_config
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e


    def get_training_pipeline_config(self):
        try:
            training_pipeline_config_info = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR, training_pipeline_config_info[TRAINING_PIPELINE_NAME_KEY], training_pipeline_config_info[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])

            traininig_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"Training Pipeline cong: {training_pipeline_config_info}")
            return traininig_pipeline_config
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e
