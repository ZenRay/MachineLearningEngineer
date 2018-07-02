#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The module is used to build the model
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import cross_validate, GridSearchCV


def xgb_dmatrix(data, features, label):
    """
    Create the new data that can be used in the xgboost model, which need DMatrix
    datatype

    Params:
        (DataFrame) data - need to be transformed to the DMatrix datatype
        (list) features - features will be used in the train model
        (str) label - the feature is a predicted label
    Result:
        (DMatrix) ddata - XGBoost core datatype, DMatrix
    """
    # extract the essential data with featues from data
    sample_data = data.copy().loc[:, features]
    if label in features:
        ddata = xgb.DMatrix(sample_data.drop(label, axis=1), 
            sample_data[label], missing=0)
    else:
        ddata = xgb.DMatrix(sample_data, data[label], missing=0)

    return ddata

def xgb_naive_model(params, dtrain, num_boost_round, early_stopping_rounds, evals,
                    feval, verbose_eval=True, **kwargs):
    """
    Create the XGBoost naive model

    Result:
        (booster) model - a trained booster model
    """
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round,feval=feval,
        early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval,
        evals=evals, **kwargs)

    return model