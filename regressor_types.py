from enum import Enum
from sklearn import linear_model
from sklearn import ensemble


class Regressor(Enum):
    LINEAR = linear_model.LinearRegression
    GRADIENT_BOOSTER = ensemble.GradientBoostingRegressor
