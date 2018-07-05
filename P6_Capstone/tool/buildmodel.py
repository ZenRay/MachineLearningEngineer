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


def dmatrix(data, features, **kwargs):
    """
    Create the DMatrix which is the xgboost datatype

    Params:
        (DataFrame) data - need to be transformed to the DMatrix datatype
        (list | string) features - need to be kept in the DMatrix data

    Result:
        (DMatrix) result - DMatrix data contains features
    """
    # extract the data with features
    sample_data = data.copy().loc[:, features]
    result = xgb.DMatrix(sample_data, **kwargs)

    return result

def xgb_dmatrix(data, features, label, **kwargs):
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
            sample_data[label], **kwargs)
    else:
        ddata = xgb.DMatrix(sample_data, data[label], **kwargs)

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

def down_model(model, features, feat_name, model_name, report_name=None,
                report_option=False):
    """
    After the model trianed by features. Save the features into a file, and 
    save the model into a file
    
    Params:
        (Booster) model - model is trained 
        (list) features - list or array is used to train model
        (string) feat_name - a file with path stores the features
        (string) model_name - a file stores the model
    """

    # create the features map
    with open(feat_name, "w") as file:
        for index, feature in enumerate(features):
            file.write("i\t{0}\t{1}\tq\n".format(index, feature))

    # save the model
    model.save_model(model_name)

    # save the model report
    if report_option:
        model.dump_model(report_name, fmap=feat_name, with_stats=True)