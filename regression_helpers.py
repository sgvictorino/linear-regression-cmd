from sklearn import linear_model
from sklearn.model_selection import cross_validate, LeaveOneOut, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pickle
import json
import math

from error_handler import ErrorHandler

from regressor_types import Regressor


class RegressionHelpers:
    @staticmethod
    def update_or_overwrite_regression_model(input_data: list, output_data: list, regressor: Regressor, get_cross_validation_metrics: bool, model_file_path: str, train_size: float = 1.0):
        try:
            model_file = open(model_file_path, "wb")
            regressor = regressor.value
            new_model = regressor().fit(
                np.array(input_data), np.array(output_data))
            pickle.dump(new_model, model_file)
            if len(input_data) == 1:
                return "{}"

            X_train, X_test, y_train, y_test = train_test_split(
                input_data, output_data, train_size=train_size if train_size * len(output_data) > 2 else 0.99, random_state=42)

            test_model = regressor().fit(
                np.array(X_train), np.array(y_train))

            y_pred = test_model.predict(X_test)

            # return the model metrics
            metrics = dict(mse=mean_squared_error(
                y_test, y_pred), r2=r2_score(y_test, y_pred))
            # and return the cross-validation metrics using dict.update(), if activated
            if get_cross_validation_metrics and len(input_data) > 1:
                cvMetrics = ("explained_variance", "max_error", "neg_mean_absolute_error",
                             "neg_mean_squared_error", "neg_median_absolute_error")

                cvMetricScores = cross_validate(regressor(), np.array(X_test), np.array(
                    y_test), scoring=cvMetrics, cv=LeaveOneOut(), return_train_score=True)

                # make a subset of the metric scores that excludes anything with NaNs in them
                # this is a way to get around a "RuntimeError: dictionary changed size during iteration" exception
                cvMetricScoresWithoutInvalidResults = dict(cvMetricScores)
                for scoreNparrayKey in cvMetricScores:
                    # while we're at it, turn the numpy arrays into Python lists
                    cvMetricScores[scoreNparrayKey] = cvMetricScores[scoreNparrayKey].tolist(
                    )
                    if not math.isnan(cvMetricScores[scoreNparrayKey][0]):
                        cvMetricScoresWithoutInvalidResults[scoreNparrayKey] = cvMetricScores[scoreNparrayKey]

                metrics.update(cvMetricScoresWithoutInvalidResults)

            metricsWithoutNaN = dict(metrics)
            # transform NaN values into undefined
            for metricKey in metrics:
                if type(metrics[metricKey]) is not list and math.isnan(metrics[metricKey]):
                    metricsWithoutNaN.pop(metricKey)

            if len(metricsWithoutNaN) == 1 and metricsWithoutNaN["mse"] == 0:
                return "{}"

            return json.dumps(metricsWithoutNaN)
        except Exception as ex:
            ErrorHandler.command_failed(ex)

    @staticmethod
    def get_regression_model(model_file_path: str):
        try:
            model_file = open(model_file_path, "rb")
            return pickle.load(model_file)
        except EOFError:
            return None

    @staticmethod
    def perform_regression_model_prediction(input_to_predict: list, model_file_path: str):
        try:
            model = RegressionHelpers.get_regression_model(
                model_file_path)
            if model == None:
                return None
            return list(model.predict(np.array(input_to_predict)))
        except Exception as ex:
            ErrorHandler.command_failed(ex)
