from flask import Flask
import os
import sys
import json
from flask import send_file, abort, render_template

from adult_census_income.constant import *
from adult_census_income.util.util import read_yaml_file
from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.logger import logging
from adult_census_income.config.configuration import Configuration
from adult_census_income.pipeline.pipeline import Pipeline
from adult_census_income.component.model_training import AdultCensusIncomeEstimatorModel
from adult_census_income.constant import CONFIG_DIR


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "housing"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


app=Flask(__name__)


@app.route("/",methods=['GET','POST'])
def index():
    return render_template('/home/cat/Desktop/ML/Adult-Census-Income-Prediction/templates/index.html')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     try:
#         print
#         return render_template('index.html')
#     except Exception as e:
#         return str(e)


if __name__=="__main__":
    app.run(debug=True)