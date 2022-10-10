import os,sys

from adult_census_income.config.configuration import Configuration
from adult_census_income.logger import logging
from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.entity.entity_config import DataIngestionConfig
from adult_census_income.entity.artifact_entity import DataIngestionArtifact
from adult_census_income.component.data_ingestion import DataIngestion

class Pipeline:

    def __init__(self,config: Configuration = Configuration()) -> None:
        try:
            self.config=config

        except Exception as e:
            raise AdutlCensusIncomeException(e,sys) from e

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise AdutlCensusIncomeException(e,sys) from e    


    def start_data_validation(self):
        pass

    def start_data_transformation(self):
        pass

    def start_model_trainer(self):
        pass

    def start_model_evaluation(self):
        pass

    def start_model_pusher(self):
        pass

    def run_pipeline(self):
        try:
            #data ingestion

            data_ingestion_artifact = self.start_data_ingestion()

        except Exception as e:
            raise AdutlCensusIncomeException(e,sys) from e