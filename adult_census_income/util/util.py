import yaml
from adult_census_income.exception import AdutlCensusIncomeException
import os
import sys

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