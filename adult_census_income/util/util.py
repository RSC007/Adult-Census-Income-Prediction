import os
import sys
import yaml
import pandas as pd
from adult_census_income.exception import AdutlCensusIncomeException

def read_yaml_file(file_path: str):
    """
    Read a YAML file and return the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise AdutlCensusIncomeException(e, sys) from e

def read_csv_file(file_path: str):
    """
    Read csv file and return the contents as dataframe.
    file_path: str
    """
    try:
        dataframe = pd.read_csv(file_path)
        return dataframe
    except Exception as e:
        raise AdutlCensusIncomeException(e, sys) from e