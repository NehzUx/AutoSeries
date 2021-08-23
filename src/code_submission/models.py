import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class LGBMRegressor:
    def __init__(self, params=None):
        self.model = None

        self.params = {
            'params': {"objective": "regression", "metric": "l2", 'verbosity': -1, "seed": 0, 'two_round': False,
                       'num_leaves': 20, 'learning_rate': 0.05, 'bagging_fraction': 0.9, 'bagging_freq': 3,
                       'feature_fraction': 0.9, 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5,
                       'lambda_l2': 0.5, 'min_data_in_leaf': 50
                       },
            'early_stopping_rounds': 100,
            'num_boost_round': 1000,
            'verbose_eval': 20
        }

        if params is not None:
            self.params = params

    def fit(self, X_train, y_train, categorical_feature=None, X_eval=None, y_eval=None):
        self.feature_name = list(X_train.columns)

        if X_eval is None or y_eval is None:
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        self.model = lgb.train(train_set=lgb_train, valid_sets=lgb_eval, valid_names='eval', **self.params)

        return self

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("You must fit first!")

        return self.model.predict(X_test)

    def score(self):
        return dict(zip(self.feature_name, self.model.feature_importance('gain')))

