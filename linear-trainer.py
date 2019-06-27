#!/usr/bin/env python3

from argparse import ArgumentParser
import sys
import json
import os
import ast
import traceback

from utils import *

from regression_helpers import RegressionHelpers
from regressor_types import Regressor

actions = []
next_data_train = False
next_data_predict = False
next_data_set_model_path = False
next_data_set_train_size = False

regressor = None
train_size = None
model_path = None


def help_screen():
    print(
        """
    linear-trainer command-line help
    Train linear regressions with this handy, state-less command-line tool.

    Examples:
        --predict '{"input_data": [[1, 2, 1]]}' -m my_model.pickle
        --train '{"input_data": [[1, 1, 2]], "output_data": [4]}' -m my_model.pickle --cross-validation
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
            get_cross_validation_metrics = False
            if "--cross-validation" in sys.argv[1:] or "-c" in sys.argv[1:]:
                get_cross_validation_metrics = True

            regressor = Regressor.LINEAR
            if "--gradient-boosting-regression" in sys.argv[1:] or "-gb" in sys.argv[1:]:
                regressor = Regressor.GRADIENT_BOOSTER

            actions.append(("train", json.loads(
                arg), regressor, get_cross_validation_metrics, train_size))
            next_data_train = False
        elif next_data_predict:
            actions.append(("predict", json.loads(arg), None))
            next_data_predict = False
        elif next_data_set_model_path:
            model_path = arg
            next_data_set_model_path = False
        elif next_data_set_train_size:
            train_size = arg
            next_data_set_train_size = False
        else:
            if arg in ("--train", "-t"):
                next_data_train = True
            elif arg in ("--predict", "-p"):
                next_data_predict = True
            elif arg in ("--model", "-m"):
                next_data_set_model_path = True
            elif arg in ("--train-size"):
                next_data_set_train_size = True
            elif arg in ("--cross-validation", "-c"):
                # do nothing; this is handled after data is passed following the --train flag
                pass
            elif arg in ("--gradient-boosting-regression", "-gb"):
                # do nothing; this is also handled after data is passed following the --train flag
                pass
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
            response = RegressionHelpers.update_or_overwrite_regression_model(
                action[1]["input_data"], action[1]["output_data"], action[2], action[3], model_path, get_first_not_none_item_in_sequence([call_function_with_args_if_value_not_none(ast.literal_eval, action[4], action[4]), call_function_with_args_if_value_not_none(ast.literal_eval, train_size, train_size), 0.99]))
        elif action[0] == "predict":
            response = RegressionHelpers.perform_regression_model_prediction(
                action[1]["input_data"], model_path)
        if response:
            print(response)
except Exception as ex:
    sys.exit(traceback.format_exc())
