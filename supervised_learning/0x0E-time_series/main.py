#!/usr/bin/env python3
""" Main File """
import numpy as np
import tensorflow as tf
preprocess = __import__('preprocess_data').preprocess_data
load_data = __import__('forecast_btc').load_data
compile_and_fit = __import__('forecast_btc').compile_and_fit

file_name = './bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
train_df, val_df, test_df = preprocess(file_name)

train_res, val_res, test_res = load_data(24, 24, 1, train_df, val_df, test_df)

history = compile_and_fit(lstm_model, train_res, val_res)
