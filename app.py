from collections import namedtuple
from flask import Flask, request
import os
import sys
import dill 
import pandas as pd
import numpy as np
from flask import send_file, abort, render_template

from adult_census_income.constant import *
from adult_census_income.entity.predict import AdultCensusIncomePredictor
from adult_census_income.util.util import read_yaml_file
from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.logger import get_log_dataframe, logging
from adult_census_income.constant import CONFIG_DIR



ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "adult_census_income"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

ADULT_CENSUS_DATA_KEY = "adult_census_data"
MEDIAN_ADULT_CENSUS_VALUE_KEY = "median_adult_census_value"

AdlutCensusData = namedtuple("HousingData",['age', 'workclass', 'education', 'occupation', 'relationship', 'race', 'sex', 'hours', 'country'])


app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


# @app.route('/view_experiment_hist', methods=['GET', 'POST'])
# def view_experiment_history():
#     experiment_df = Pipeline.get_experiments_status()
#     context = {
#         "experiment": experiment_df.to_html(classes='table table-striped col-12')
#     }
#     return render_template('experiment_history.html', context=context)

@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    print(req_path)
    if req_path == "logs":
        os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
        # Joining the base and the requested path
        logging.info(f"req_path: {req_path}")
        abs_path = os.path.join(req_path)
        # Return 404 if path doesn't exist
        if not os.path.exists(os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME, abs_path)):
            return abort(404)

        # Show directory contents
        files = {os.path.join(abs_path, file): file for file in os.listdir(os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME, abs_path))}

        result = {
            "files": files,
            "parent_folder": os.path.dirname(abs_path),
            "parent_label": abs_path
        }
        return render_template('log_files.html', result=result)
    else:
        # Check if path is a file and serve
        if os.path.isfile(os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME, "logs", req_path)):
            log_df = get_log_dataframe(os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME, "logs", req_path))
            context = {"log": log_df.to_html(classes="table-striped", index=False)}
            return render_template('log.html', context=context)


@app.route('/artifact', defaults={'req_path': PIPELINE_FOLDER_NAME})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs(PIPELINE_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    abs_path = os.path.join(req_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)



def get_latest_model():
        try:
            model_path = os.path.join(ROOT_DIR, PIPELINE_NAME, "artifact", "model_trainer")
            for file in os.listdir(model_path):
                print(file)

            model_path =  os.path.join(model_path, os.listdir(model_path)[-1], "trained_model", "model.pkl")
            print(model_path)

            return dill.load(open(model_path, "rb"))
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        ADULT_CENSUS_DATA_KEY: None,
        MEDIAN_ADULT_CENSUS_VALUE_KEY: None
    }
    # predict = AdultCensusIncomePredictor()
    # predict.predict()

    model_ref = get_latest_model()
    print("model_pathmodel_pathmodel_pathmodel_path")
    

    

    if request.method == 'POST':
        age = float(request.form['age'])
        workclass = float(request.form['workclass'])
        education = float(request.form['education-num'])
        occupation = float(request.form['occupation'])
        relationship = float(request.form['relationship'])
        race = float(request.form['race'])
        sex = float(request.form['sex'])
        hours = float(request.form['hours-per-week'])
        country = request.form['country']


        adlut_census_data = AdlutCensusData(age=age,
                                   workclass=workclass,
                                   education=education,
                                   occupation=occupation,
                                   relationship=relationship,
                                   race=race,
                                   sex=sex,
                                   hours=hours,
                                   country=country,
                                   )

        print("adlut_census_data", model_ref)
        predict = model_ref.predict(pd.DataFrame(adlut_census_data))
        print("su0cess")
        print(predict)
        housing_df = adlut_census_data.get_housing_input_data_frame()
        housing_predictor = AdultCensusIncomePredictor(model_dir=MODEL_DIR)
        median_housing_value = housing_predictor.predict(X=housing_df)
        context = {
            ADULT_CENSUS_DATA_KEY: adlut_census_data.get_housing_data_as_dict(),
            MEDIAN_ADULT_CENSUS_VALUE_KEY: median_housing_value,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


if __name__=="__main__":
    app.run(debug=True)