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
