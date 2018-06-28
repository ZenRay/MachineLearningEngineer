#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from zipfile import ZipFile
"""
The preprocess module is a tool that collects all the data, preprocesses the 
data. At the same time, utility the tool creates the features and the target. 

"""

def load_data(path, file_name, dtype:dict =None, converters:dict =None):
    """
    read the data from the zip file that contains a csv file

    Params:
        (str) path - the directory stores those files.
        (str) file_name - name of the file that is compressed in the zipfile
        (Dict) dtype - specify data type of the dataframe
        (Dict) converters - dict stores the convert function
    Result:
        (DataFrame) df - load the data into the dataframe
    """

    with ZipFile(path + file_name + ".zip", "r") as zip_file:
        with zip_file.open(file_name, "r") as file:
            df = pd.read_csv(file, dtype=dtype, converters=converters)

    return df

def create_dummies(df, dummy_variable, drop_variables=[]):
    """
    transform the category varibale by using get_dummy method

    Params:
        (DataFrame) df - dataframe which need to transform the variable
        (str) dummy_variable - string that is a category variable need transform
        (list) drop_variables - variables would be dropped
    Result:
        (DataFrame) df - dataframe contains dummy variables
    """
    
    dummies = pd.get_dummies(df[dummy_variable], prefix=dummy_variable)
    df = df.merge(dummies, on=df.index, validate="1:1", how="left")

    # drop duplicate meaning variable
    drop_variables.append(dummy_variable)

    if "key_0" in df.columns:
        drop_variables.append("key_0")
        df.drop(drop_variables, axis=1, inplace=True)
    else:
        df.drop(columns=drop_variables, axis=1, inplace=True)
    
    return df

def collect_data(train_data, test_data, store_data):
    """
    collect the dataset into a new variable

    Params:
        (DataFrame) train_data - dataframe stores the train dataset
        (DataFrame) test_data - dataframe stores the test dataset
        (DataFrame) store_data - dataframe stores the store dataset
    Result:
        (DataFrame) result - dataframe store the dataset collected
    """

    # create a tag to differentiate between the train dataset and the test dataset
    train_data.loc[:, "Tag"] = "train"
    test_data.loc[:, "Tag"] = "test"

    # assign 1 to the open with missing value in the test dataset
    test_data["Open"].fillna(1, inplace=True)
    
    # drop the dupilcate feature in the test dataset
    test_data.drop("Id", axis=1, inplace=True)

    # drop the sales 0 with the open 0 in the train dataset
    train_data = train_data.loc[((train_data["Sales"] !=0) & (train_data["Open"]!=0))]

    # fill the missing value about the store's competitionDistance, 
    # CompetitionOpenSinceMonth and competitionsinceyear with median value
    store_data["CompetitionDistance"].fillna(
        store_data["CompetitionDistance"].median(), inplace=True)
    store_data["CompetitionOpenSinceMonth"].fillna(
        store_data["CompetitionOpenSinceMonth"].median(),inplace=True)
    store_data["CompetitionOpenSinceYear"].fillna(
        store_data["CompetitionOpenSinceYear"].median(),inplace=True)
    
    # dummy variable about the category variable that like storetype, assortment
    # in the store dataset
    store_data = create_dummies(store_data, "StoreType", drop_variables=["StoreType_d"])

    store_data = create_dummies(store_data, "Assortment", drop_variables=["Assortment_c"])

    # dummy variable about the category variable that like stateholiday in the
    # train dataset andt in the test dataset
    state_holiday = {"a":"Public", "b":"Easter", "c":"Christmas", "0":"No"}
    train_data["StateHoliday"] = train_data.loc[:, "StateHoliday"].map(state_holiday)
    test_data["StateHoliday"] = test_data.loc[:, "StateHoliday"].map(state_holiday)

    train_data = create_dummies(train_data, "StateHoliday", drop_variables=["StateHoliday_No"])

    test_data = create_dummies(test_data, "StateHoliday", ["StateHoliday_No"])

    # transform the sales and the customers about the train dataset by using log
    train_data["Sales"] = train_data["Sales"].apply(np.log1p)
    train_data["Customers"] = train_data["Customers"].apply(np.log1p)

    # merge the train dataset, the test dataset and the test dataset into the result
    result = pd.concat([train_data, test_data], ignore_index=True, sort=False)
    result = result.merge(store_data, on="Store", sort=False)

    # dummy variable about the dayofweek in the dataset
    result = result.merge(pd.get_dummies(result["DayOfWeek"], prefix="DayOfWeek"),
        on=result.index)
    result.drop(["DayOfWeek", "DayOfWeek_7", "key_0"], axis=1, inplace=True)

    # parse the date into month, year, dayofmonth, weekofyear, dayofyear
    result["Year"] = result["Date"].dt.year
    result["Month"] = result["Date"].dt.month
    result["DayOfMonth"] = result["Date"].dt.day
    result["WeekOfYear"] = result["Date"].dt.weekofyear
    result["DayOfYear"] = result["Date"].dt.dayofyear

    
    return result