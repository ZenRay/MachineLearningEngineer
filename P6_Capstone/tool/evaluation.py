#!/usr/bin/env python
# -*-coding:utf-8 -*-

import pandas as pd
import numpy as np

"""
the evaluation module contains evaluation functions that like loss function and
so on
"""

def rmspe(y_true, y_predict):
    """
    Param:
        (array) y_true - true labels
        (array) y_predict - predict labels
    Result:
        (float) result - evaluation socre
    """
    if not(isinstance(y_predict, (np.ndarray, pd.core.series.Series)) and \
        isinstance(y_true, (np.ndarray, pd.core.series.Series))):
        raise "The type of y_true and y_predict must be ndarray, Series!"

    if len(y_true) != len(y_predict):
        raise "Length between y_predict and y_true isn't same!"

    n = len(y_predict)
    percent = (y_true - y_predict) / y_true

    score = np.sqrt(np.power(percent, 2).sum() * 1/n)
    
    return score

def rmspe_xgb(y_true, y_predict):
    """
    the function is used in the xgboost evaluation
    Param:
        (sequence) y_true - true labels
        (sequence) y_predict - predict labels
    Result:
        (float) result - evaluation socre
    """
    score = rmspe(y_true, np.array(y_predict.get_label()))
    return "RMSPE", score

def get_result(y_predict, data, path, file_name):
    """
    The function is used to convert the predict value into the exponential value.
    Store the predict values, so that check the result.

    Param:
        y_predict - sequence value which store the predict value
        (DataFrame) data - original data that it's dataframe
        (string) path - the path is a directory that stores the result
        (string) file_name - a csv file store information that is exported from
                                the data to , so that check the prediction
    Result:
        (DataFrame) data - store the predict values
    """
    # duplicate the data
    data = data.copy()

    predict = np.expm1(y_predict)
    data["Sales"] = predict

    # sort the data by Id feature
    data.sort_values("Id", inplace=True)
    # export the data into csv file
    data[["Id", "Sales"]].to_csv(path + file_name, index=False)
    return data