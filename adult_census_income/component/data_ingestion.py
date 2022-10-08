import sys,os

from adult_census_income.entity.entity_config import DataIngestionConfig
from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.logger import logging

class DataIngestion:

    def __init__(self,data_ingestion_config:sDataIngestionConfig ):
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AdutlCensusIncomeException(e,sys)


    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            pass
        except Exception as e:
            raise AdutlCensusIncomeException(e,sys) from e