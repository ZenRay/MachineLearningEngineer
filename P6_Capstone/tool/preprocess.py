#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from zipfile import ZipFile
from datetime import datetime
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
    # drop_variables.append(dummy_variable)

    if "key_0" in df.columns:
        drop_variables.append("key_0")
        df.drop(drop_variables, axis=1, inplace=True)
    else:
        df.drop(columns=drop_variables, axis=1, inplace=True)

    return df

def collect_data(train_data, test_data, store_data, duplicate_features=[]):
    """
    collect the dataset into a new variable

    Params:
        (DataFrame) train_data - dataframe stores the train dataset
        (DataFrame) test_data - dataframe stores the test dataset
        (DataFrame) store_data - dataframe stores the store dataset
        (list) duplicate_features - features that information is duplicated
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
    duplicate_features.extend(["StoreType", "Assortment"])
    store_data = create_dummies(store_data, "StoreType", ["StoreType_d"])

    store_data = create_dummies(store_data, "Assortment", ["Assortment_c"])

    # dummy variable about the category variable that like stateholiday in the
    # train dataset andt in the test dataset
    duplicate_features.extend(["StateHoliday"])
    state_holiday = {"a":"Public", "b":"Easter", "c":"Christmas", "0":"No"}
    train_data["StateHoliday"] = train_data.loc[:, "StateHoliday"].map(state_holiday)
    test_data["StateHoliday"] = test_data.loc[:, "StateHoliday"].map(state_holiday)

    train_data = create_dummies(train_data, "StateHoliday", ["StateHoliday_No"])

    test_data = create_dummies(test_data, "StateHoliday", ["StateHoliday_No"])

    # transform the sales and the customers about the train dataset by using log
    train_data["Sales"] = train_data["Sales"].apply(np.log1p)
    train_data["Customers"] = train_data["Customers"].apply(np.log1p)

    # merge the train dataset, the test dataset and the test dataset into the result
    result = pd.concat([train_data, test_data], ignore_index=True, sort=False)
    result = result.merge(store_data, on="Store", sort=False)

    # dummy variable about the dayofweek in the dataset
    duplicate_features.extend(["DayOfWeek"])
    result = result.merge(pd.get_dummies(result["DayOfWeek"], prefix="DayOfWeek"),
        on=result.index)
    result.drop(["DayOfWeek_7", "key_0"], axis=1, inplace=True)

    # parse the date into month, year, dayofmonth, weekofyear, dayofyear
    result["OpenYear"] = result["Date"].dt.year.astype("uint16")
    result["OpenMonth"] = result["Date"].dt.month.astype("uint16")
    result["OpenDayOfMonth"] = result["Date"].dt.day.astype("uint16")
    result["OpenWeekOfYear"] = result["Date"].dt.weekofyear.astype("uint16")
    result["OpenDayOfYear"] = result["Date"].dt.dayofyear.astype("uint16")

    # imitate the data that promo2 started by using year and week
    promo2date = []

    for year, week in zip(result["Promo2SinceYear"], result["Promo2SinceWeek"]):
        if ((not np.isnan(year)) & (not np.isnan(week))):
            promo2date.append(datetime.strptime("{0:04} {1:02} 1".format(
                int(year), int(week)), "%G %V %u"))
        else:
            promo2date.append(np.nan)

    # parse the information about the promo2 by comparing the date with
    # promo2startdate, firstly
    duplicate_features.extend(["Promo2"])
    result["InPromo2"] = ((result["Date"] >= pd.Series(promo2date)) &
        (result["Promo2"].apply(lambda x: True if x==1 else False))).apply(
            lambda x: 1 if x else 0)

    # parse information whether the open date is in promointerval
    promointerval = []
    for month, promomonth in zip(result["Date"].apply(datetime.strftime, args=("%b",)),
                            result["PromoInterval"].str.split(",")):

        try:
            if month in promomonth:
                promointerval.append(True)
            else:
                promointerval.append(False)
        except TypeError:
            promointerval.append(False)

    # analysis the information between open date and the promointeral
    result["InPromo2"] = ((result["InPromo2"] == 1) & pd.Series(promointerval)
                            ).apply(lambda x: 1 if x else 0).astype("uint8")

    # caculate months since the competition opened
    result["CompetitionOpenMonths"] = result["OpenMonth"] + (result["OpenYear"] -
        result["CompetitionOpenSinceYear"]) * 12 - result["CompetitionOpenSinceMonth"]

    # convert datatype for save storage
    result["Store"] = result["Store"].astype("uint16")
    result["Open"] = result["Open"].astype("uint16")
    return result, duplicate_features

def extend_features(data):
    """
    Create the new features which will be used to create the model, and create
    the new information data

    Params:
        (DataFrame) data - get the dataframe from the collect data
    Result:
        (DataFrame) result - dataframe store the data information
        (list) new_features - store the features that will bu used to create the model
    """

    # initial the featues
    new_features = []

    # calculate the store information that is parsed from the sales feature and
    # the customers feature
    new_features.extend(["AvgSales", "AvgCustomers", "AvgSalesPerCustomers",
        "MedianCustomers"])

    # grouby the store to calculate
    train_data = data[data["Tag"]=="train"].copy()

    data = data.merge((train_data.groupby("Store")["Sales"].sum()
        / train_data.groupby("Store")["Open"].count()).reset_index(name="AvgSales"),
        how="left", on="Store", validate="m:1")

    data = data.merge((train_data.groupby("Store")["Customers"].sum()
        / train_data.groupby("Store")["Open"].count()).reset_index(name="AvgCustomers"),
        how="left", on="Store", validate="m:1")

    data["AvgSalesPerCustomers"] = data["AvgSales"] / data["AvgCustomers"]

    data = data.merge(train_data.groupby("Store")["Customers"].median().reset_index(name="MedianCustomers"),
        on="Store", how="left", validate="m:1")

    # calculate the number of the schoolholiday in this week, lastweek and next
    # week
    new_features.extend(["HolidayThisWeek", "HolidayNextWeek", "HolidayLastWeek"])

    school_holiday_counts = data.groupby(["Store", "OpenYear", "OpenWeekOfYear"]) \
        ["SchoolHoliday"].sum().reset_index(name="HolidayThisWeek")

    school_holiday_counts["HolidayNextWeek"] = \
        school_holiday_counts["HolidayThisWeek"].shift(-1)
    school_holiday_counts["HolidayLastWeek"] = \
        school_holiday_counts["HolidayThisWeek"].shift()
    # fill the missing values
    school_holiday_counts.fillna(0)
    data = data.merge(school_holiday_counts, on=["Store", "OpenYear",
    "OpenWeekOfYear"], how="left", validate="m:1")

    # calculate mean, median information about the sales and the customers in
    # each every week day
    new_features.extend(["AvgSalesInDayOfWeek", "MedianSalesInDayOfWeek",
        "AvgCustsInDayOfWeek", "MedianCustsInDayOfWeek"])

    train_group = train_data.groupby(["Store", "DayOfWeek"])
    data = data.merge(train_group["Sales"].mean().reset_index(name="AvgSalesInDayOfWeek"),
        how="left", on=["Store", "DayOfWeek"], validate="m:1")

    data = data.merge(train_group["Sales"]. median().reset_index(name="MedianSalesInDayOfWeek"),
        how="left", on=["Store", "DayOfWeek"], validate="m:1")

    data = data.merge(train_group["Customers"].mean().reset_index(name="AvgCustsInDayOfWeek"),
        how="left", on=["Store", "DayOfWeek"], validate="m:1")

    data = data.merge(train_group["Customers"].median().reset_index(name="MedianCustsInDayOfWeek"),
        how="left", on=["Store", "DayOfWeek"], validate="m:1")

    data[["AvgSales", "AvgCustomers", "AvgSalesPerCustomers", "MedianCustomers",
    "HolidayThisWeek", "HolidayNextWeek", "HolidayLastWeek", "AvgSalesInDayOfWeek",
    "MedianSalesInDayOfWeek", "AvgCustsInDayOfWeek", "MedianCustsInDayOfWeek"]] = \
    	data[["AvgSales", "AvgCustomers", "AvgSalesPerCustomers", "MedianCustomers",
    	"HolidayThisWeek", "HolidayNextWeek", "HolidayLastWeek", "AvgSalesInDayOfWeek",
    	"MedianSalesInDayOfWeek", "AvgCustsInDayOfWeek", "MedianCustsInDayOfWeek"]]. \
    	astype("float16")

    return data, new_features