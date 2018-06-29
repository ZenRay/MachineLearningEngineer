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
    
    return "The rmspe score is %.4f .\n" % score