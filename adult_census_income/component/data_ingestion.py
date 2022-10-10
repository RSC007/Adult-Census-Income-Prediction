import sys
import os
import numpy as np
import pandas as pd
from adult_census_income.constant import *


from adult_census_income.entity.entity_config import DataIngestionConfig
from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.logger import logging
from adult_census_income.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys)

    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:

            adult_census_income_dataset_dir_path = os.path.join("dataset", "adult.csv")


            logging.info(f"Reading csv file: [{adult_census_income_dataset_dir_path}]")
            adult_census_income_data_frame = pd.read_csv(adult_census_income_dataset_dir_path)

            logging.info(f"Splitting data into train and test")

            strat_train_set = adult_census_income_data_frame
            strat_test_set = adult_census_income_data_frame

            train_file_path = os.path.join(self.data_ingestion_config.ingestion_train_dir, "train.csv")

            test_file_path = os.path.join(self.data_ingestion_config.ingestion_test_dir, "test.csv")

            os.makedirs(self.data_ingestion_config.ingestion_train_dir, exist_ok=True)
            os.makedirs(self.data_ingestion_config.ingestion_test_dir, exist_ok=True)

            strat_train_set.to_csv(train_file_path, index=False)
            strat_test_set.to_csv(test_file_path, index=False)

            logging.info(f"Data is splited")
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data ingestion completed successfully."
                                                            )
            logging.info(
                f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            return self.split_data_as_train_test()
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")
