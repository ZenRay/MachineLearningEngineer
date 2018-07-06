#!/usr/bin/env python
# -*-coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def vis_features_score(model, color_option=True, feature_sort_option=True, **kwargs):
    """
    visualize the features score

    Params:
        (Booster) model - XGBoost model
        (bool) color_option - if true, split the feature according by quantile
        (bool) feature_sort_option - if true, sort the feature by the score

        kwargs - other arguments is used to plot figure
                color_up - matplotlib color, if score is greater than a value, 
                            use the color_up
                color_down - matplotlib color, otherwise, use the color_down
                color_mid - matplotlib color
                percent_up - float value in [0, 1]. choose the high percent value, 
                        so that the base value choose the color_up or color_down
                percent_down - float value in [0, 1]. choose the low percent value, 
                        so that the base value choose the color
                figsize - tuple, choose the figure size
    """
    # create the feature series
    features_map = pd.Series(model.get_fscore()).sort_values(ascending=
        feature_sort_option)

    # create the color sequence that is used to color bar
    def color_map(x):
        base_value_up = features_map.quantile(q=kwargs["percent_up"])
        base_value_down = features_map.quantile(q=kwargs["percent_down"])

        if x >= base_value_up:
            return kwargs["color_up"]
        elif x >= base_value_down:
            return kwargs["color_mid"]
        else:
            return kwargs["color_down"]

    if color_option:
        color_seq = features_map.apply(color_map)
    else:
        color_seq = None
    
    # bar plot about  the features
    ax = features_map.plot(kind="barh", figsize=kwargs["figsize"], color=color_seq)
    plt.title("Feature Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)

    # adjust the bar color
    for text, color_name in zip(ax.axes.get_yticklabels(), color_seq):
        text.set_color(color_name)
        text.set_fontsize(10)

    plt.show()
    return ax