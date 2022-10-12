from collections import namedtuple
import importlib
import os
from statistics import mode
import sys
from typing import List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.logger import logging
from adult_census_income.constant import *
from adult_census_income.util.util import read_yaml_file, save_object
from adult_census_income.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from adult_census_income.entity.entity_config import ModelTrainerConfig

MODULE_KEY = "module"
CLASS_KEY = "class"
GRID_SEARCH_KEY = 'grid_search'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"

InitializedModelDetail = namedtuple("InitializedModelDetail", [
                                    "model_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score",
                                                             ])
BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score", ])

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_classification_report", "test_classification_report", "train_accuracy",
                                 "test_accuracy", "index_number"])

class AdultCensusIncomeEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:

    def __init__(self, model_tranformed_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig) -> None:
        try:
            self.models_config = read_yaml_file(file_path=MODEL_FILE_PATH)
            self.data_transform_artifact = model_tranformed_artifact
            self.model_trainer_config = model_trainer_config

            self.grid_searched_best_model_list = None
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def get_configured_model_ref_list(self, model_list: list):
        try:
            """
            This function refence the model.
            return: Model reference with its perameters
            """
            initialize_model_ref_list = []

            for model_number, params in model_list.items():
                model_initialization_config = self.models_config[MODEL_SELECTION_KEY][model_number]

                # taking the model reference
                model_obj_ref = ModelFactory.class_for_name(
                    module=params[MODULE_KEY], class_name=params[CLASS_KEY])

                model = model_obj_ref()

                if PARAM_KEY in model_initialization_config:
                    model_object_property = model_initialization_config[PARAM_KEY]
                    model_obj_ref = ModelFactory.update_property_of_class(
                        instance_ref=model, property_data=model_initialization_config)

                param_grid_search = None
                if SEARCH_PARAM_GRID_KEY in model_initialization_config:
                    param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]

                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"

                model_initialization_config = InitializedModelDetail(
                    model_number=model_number,
                    model=model,
                    param_grid_search=param_grid_search,
                    model_name=model_name)

                initialize_model_ref_list.append(model_initialization_config)
            return model_initialization_config
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature,
                                      output_feature) -> GridSearchedBestModel:
        """
        excute_grid_search_operation(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """
        try:
            # instantiating GridSearchCV class

            grid_search_cv_ref = ModelFactory.class_for_name(module=self.models_config[GRID_SEARCH_KEY][MODULE_KEY],
                                                             class_name=self.models_config[GRID_SEARCH_KEY][CLASS_KEY]
                                                             )

            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.models_config[GRID_SEARCH_KEY][PARAM_KEY])

            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__} Started." {"<<"*30}'
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__}" completed {"<<"*30}'
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_
                                                             )

            return grid_searched_best_model
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self, initialized_model_list, input_feature, output_feature):
        try:
            self.grid_searched_best_model_list = []
            # for initialized_model_list in initialized_model_list:
            grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                initialized_model=initialized_model_list,
                input_feature=input_feature,
                output_feature=output_feature
            )
            self.grid_searched_best_model_list.append(
                grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    def initiate_model_trainer(self, base_accuracy=0.6):
        try:
            logging.info("Started Initializing model from config file")
            model_list = self.models_config[MODEL_SELECTION_KEY]
            initialized_model_list = self.get_configured_model_ref_list(
                model_list)
            logging.info(f"Initialized model: {initialized_model_list}")

            dataframe = pd.read_csv(
                self.data_transform_artifact.transformed_train_file_path)

            X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(
                ["salary"], axis=1), dataframe["salary"], test_size=0.2, random_state=2)
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X_train,
                output_feature=y_train
            )

            best_model = ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                        base_accuracy=base_accuracy)

            trained_model_file_path=self.model_trainer_config.trained_model_file_path

            model_list = [model.best_model for model in grid_searched_best_model_list ]
            metric_info:MetricInfoArtifact = evaluate_regression_model(model_list=model_list,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,base_accuracy=base_accuracy)

            logging.info(f"Best found model on both training and testing dataset.")

            model_object = metric_info.model_object
            # Pickel file is not created yet
            # adult_census_income_model = AdultCensusIncomeEstimatorModel(preprocessing_object=preprocessing_obj,trained_model_object=model_object)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            # save_object(file_path=trained_model_file_path,obj=adult_census_income_model)

            model_trainer_artifact=  ModelTrainerArtifact(is_trained=True,message="Model Trained successfully",
            trained_model_file_path=trained_model_file_path,
            train_rmse=metric_info.train_classification_report,
            test_rmse=metric_info.test_classification_report,
            train_accuracy=metric_info.train_accuracy,
            test_accuracy=metric_info.test_accuracy,
            )
            print(model_trainer_artifact)
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e


class ModelFactory:
    def __init__(self):
        try:
            self.config: dict = read_yaml_file(file_path=MODEL_FILE_PATH)

            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = dict(
                self.config[GRID_SEARCH_KEY][PARAM_KEY])

            self.models_initialization_config: dict = dict(
                self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    @staticmethod
    def class_for_name(module: str, class_name: str):
        try:
            # raise error is module not imported properly
            import_module = importlib.import_module(module)

            logging.info(
                f"Executing command: from {module} import {class_name}")

            class_ref = getattr(import_module, class_name)
            return class_ref
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref: object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data paremeter required dictonary.")

            for key, value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)

            return instance_ref
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy=0.6
                                                          ) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(
                        f"Acceptable model found:{grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(
                    f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise AdutlCensusIncomeException(e, sys) from e


def evaluate_regression_model(model_list: list, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, base_accuracy: float = 0.6) -> MetricInfoArtifact:
    """
    Description:
    This function compare multiple regression model return best model
    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature
    return
    It retured a named tuple

    MetricInfoArtifact = namedtuple("MetricInfo",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])
    """
    try:

        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)  # getting model name based on model object
            logging.info(
                f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")

            # Getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculating r squared score on training and testing dataset
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            # Calculating mean squared error on training and testing dataset
            train_classification_report = classification_report(y_train, y_train_pred)
            test_classification_report = classification_report(y_test, y_test_pred)

            diff_test_train_acc = abs(train_acc - test_acc)

            # logging all important metric
            logging.info(f"{'>>'*30} Accuracy {'<<'*30}")
            logging.info(f"Train Score\t\t Test Score\t\t Train Test Accuracy diff")
            logging.info(f"{train_acc}\t\t {test_acc}\t\t {diff_test_train_acc}")

            logging.info(f"{'>>'*30} Classification Report {'<<'*30}")
            logging.info(f"Train Classification Report: [{train_classification_report}].")
            logging.info(f"Test Classification Report: [{test_classification_report}].")

            logging.info(f"{'>>'*30} Confusion Mertrix {'<<'*30}")
            logging.info(f"Train : [{confusion_matrix(y_train_pred, y_train)}].")
            logging.info(f"Test : [{confusion_matrix(y_test_pred, y_test)}].")

            # if model accuracy is greater than base accuracy and train and test score is within certain thershold
            # we will accept that model as accepted model
            if test_acc >= base_accuracy and diff_test_train_acc < 0.08:
                base_accuracy = test_acc
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                          model_object=model,
                                                          train_classification_report=train_acc,
                                                          test_classification_report=test_acc,
                                                          train_accuracy=train_acc,
                                                          test_accuracy=test_acc,
                                                          index_number=index_number)

                logging.info(
                    f"Acceptable model found {metric_info_artifact}. ")
            index_number += 1
        if metric_info_artifact is None:
            logging.info(
                f"No model found with higher accuracy than base accuracy")
        return metric_info_artifact
    except Exception as e:
        raise AdutlCensusIncomeException(e, sys) from e
