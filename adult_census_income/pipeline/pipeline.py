import os,sys
from adult_census_income.component.data_transformation import DataTranformation

from adult_census_income.component.data_validation import DataValidation
from adult_census_income.component.model_training import ModelTrainer
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


    def start_data_validation(self, data_ingestion_artifact):
        try:
            data_validation = DataValidation(data_ingestion_artifact_config=data_ingestion_artifact, data_validation_config=self.config.get_data_validation_config())
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifact):
        try:
            data_tranformation = DataTranformation(data_ingestion_artifact_config=data_ingestion_artifact, data_tranformation_config=self.config.get_data_trasformation_config())
            return data_tranformation.initiate_data_tranformation()
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def start_model_trainer(self, model_tranformed_artifact):
        try:
            data_trainer = ModelTrainer(model_tranformed_artifact=model_tranformed_artifact, model_trainer_config=self.config.get_model_trainer_config())
            return data_trainer.initiate_model_trainer()
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def start_model_evaluation(self):
        pass

    def start_model_pusher(self):
        pass

    def run_pipeline(self):
        try:

            logging.info(f"{'-=-'*50} Data Ingestion {'-=-'*50}")
            data_ingestion_artifact = self.start_data_ingestion()

            logging.info(f"{'-=-'*50} Data Validation {'-=-'*50}")
            self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

            logging.info(f"{'-=-'*50} Data Transformation {'-=-'*50}")
            model_tranformed_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)

            logging.info(f"{'-=-'*50} Data Trainer {'-=-'*50}")
            self.start_model_trainer(model_tranformed_artifact)

        except Exception as e:
            raise AdutlCensusIncomeException(e,sys) from e