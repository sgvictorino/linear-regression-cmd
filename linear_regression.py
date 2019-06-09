"""
    A state-less command-line tool that trains linear regression models.
    Copyright 2019 Solomon Victorino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from sklearn import linear_model
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pickle
import json
import math

from error_handler import ErrorHandler


class LinearRegression:
    @staticmethod
    def update_or_overwrite_linear_regression_model(input_data: list, output_data: list, get_cross_validation_metrics: bool, model_file_path: str):
        try:
            model_file = open(model_file_path, "wb")
            new_model = linear_model.LinearRegression().fit(
                np.array(input_data), np.array(output_data))
            pickle.dump(new_model, model_file)

            # return the model metrics
            metrics = dict(mse=mean_squared_error(output_data, new_model.predict(
                input_data)), r2=r2_score(output_data, new_model.predict(input_data)))
            # and return the cross-validation metrics using dict.update(), if activated
            if get_cross_validation_metrics:
                cvMetrics = ("explained_variance", "max_error", "neg_mean_absolute_error",
                             "neg_mean_squared_error", "neg_median_absolute_error")

                cvMetricScores = cross_validate(linear_model.LinearRegression(), np.array(input_data), np.array(
                    output_data), scoring=cvMetrics, cv=LeaveOneOut(), return_train_score=True)

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
            return json.dumps(metrics)
        except Exception as ex:
            ErrorHandler.command_failed(ex)

    @staticmethod
    def get_linear_regression_model(model_file_path: str) -> linear_model.LinearRegression:
        try:
            model_file = open(model_file_path, "rb")
            return pickle.load(model_file)
        except EOFError:
            return None

    @staticmethod
    def perform_linear_regression_model_prediction(input_to_predict: list, model_file_path: str):
        try:
            model = LinearRegression.get_linear_regression_model(
                model_file_path)
            if model == None:
                return None
            return list(model.predict(np.array(input_to_predict)))
        except Exception as ex:
            ErrorHandler.command_failed(ex)
