from flask import Flask
import os
import sys
import json
from flask import send_file, abort, render_template

from adult_census_income.constant import *
from adult_census_income.util.util import read_yaml_file
from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.logger import get_log_dataframe, logging
from adult_census_income.config.configuration import Configuration
from adult_census_income.pipeline.pipeline import Pipeline
from adult_census_income.component.model_training import AdultCensusIncomeEstimatorModel
from adult_census_income.constant import CONFIG_DIR



ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "adult_census_income"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)

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


if __name__=="__main__":
    app.run(debug=True)