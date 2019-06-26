#!/usr/bin/env python3

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
            actions.append(("train", json.loads(
                arg), get_cross_validation_metrics))
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
            elif arg in ("--cross-validation", "-c"):
                # do nothing; this is handled after data is passed following the --train flag
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
            response = LinearRegression.update_or_overwrite_linear_regression_model(
                action[1]["input_data"], action[1]["output_data"], action[2], model_path)
        elif action[0] == "predict":
            response = LinearRegression.perform_linear_regression_model_prediction(
                action[1]["input_data"], model_path)
        if response:
            print(response)
except Exception as ex:
    sys.exit(Exception(ex))
