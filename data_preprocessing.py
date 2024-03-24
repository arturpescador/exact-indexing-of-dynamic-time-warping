"""
This file contains the functions to load and process the data.

Author:
- Artur Dandolini Pescador
"""

import numpy as np
import pandas as pd
import os

def load_and_process_data(filepath):
    data = pd.read_csv(filepath)
    del data['Unnamed: 0']
    data['Date'] = pd.to_datetime(data['Date'])
    timeseries = data.set_index(data['Date'])
    del timeseries['Date']
    return timeseries
    
def load_data_UCR(data_dir, dataset_name):
    data_dir = os.path.join(data_dir, dataset_name)
    train_file = os.path.join(data_dir, dataset_name + "_TRAIN")
    test_file = os.path.join(data_dir, dataset_name + "_TEST")
    train_data = np.loadtxt(train_file, delimiter=',')
    test_data = np.loadtxt(test_file, delimiter=',')
    return train_data, test_data

def normalize_timeseries(timeseries):
    return (timeseries - timeseries.mean()) / timeseries.std()