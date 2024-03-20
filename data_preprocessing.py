"""
This file contains the functions to load and process the data.

Author:
- Artur Dandolini Pescador
"""

import numpy as np
import pandas as pd

def load_and_process_data(filepath):
    data = pd.read_csv(filepath)
    del data['Unnamed: 0']
    data['Date'] = pd.to_datetime(data['Date'])
    timeseries = data.set_index(data['Date'])
    del timeseries['Date']
    return timeseries

def normalize_timeseries(timeseries):
    return (timeseries - timeseries.mean()) / timeseries.std()