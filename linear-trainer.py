#!/usr/bin/env python3

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

from argparse import ArgumentParser
import sys
import json
import os

from linear_regression import LinearRegression

actions = []
next_data_train = False
next_data_predict = False
next_data_set_model_path = False

model_path = None


def help_screen():
    print(
        """
    linear-trainer command-line help
    Train linear regressions with this handy, state-less command-line tool.

    Examples:
        --predict '{"input_data": [[1, 2, 1]]}' -m my_model.pickle
        --train '{"input_data": [[1, 1, 2]], "output_data": [4]}' -m my_model.pickle
        --train '{"input_data": [[1, 2, 3], [3, 2, 1]], "output_data": [1, 2]}' --predict '{"input_data": [[1, 2, 3]]}' -m my_model.pickle
    """)
    sys.exit(0)


if len(sys.argv[1:]) == 0:
    help_screen()
try:
    for arg in sys.argv[1:]:
        if arg in ("--help", "-h") or len(sys.argv[1:]) == 0:
            help_screen()
        if next_data_train:
            actions.append(("train", json.loads(arg)))
            next_data_train = False
        elif next_data_predict:
            actions.append(("predict", json.loads(arg), None))
            next_data_predict = False
        elif next_data_set_model_path:
            model_path = arg
            next_data_set_model_path = False
        else:
            if arg in ("--train", "-t"):
                next_data_train = True
            elif arg in ("--predict", "-p"):
                next_data_predict = True
            elif arg in ("--model", "-m"):
                next_data_set_model_path = True
            else:
                sys.exit(Exception("Invalid argument '" + arg + "'!"))
    if model_path == None:
        sys.exit(Exception("Model name not provided!"))
    else:
        try:
            if len(os.path.dirname(model_path)) > 0:
                os.makedirs(os.path.dirname(model_path))
        except OSError as ex:
            sys.exit(ex)
    if len(actions) == 0:
        help_screen()
    for action in actions:
        if action[0] == "train":
            response = LinearRegression.update_or_overwrite_linear_regression_model(
                action[1]["input_data"], action[1]["output_data"], model_path)
        elif action[0] == "predict":
            response = LinearRegression.perform_linear_regression_model_prediction(
                action[1]["input_data"], model_path)
        if response:
            print(response)
except Exception as ex:
    sys.exit(Exception(ex))
