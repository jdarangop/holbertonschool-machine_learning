#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.loc[df["Timestamp"] >= 1483228800]

df = df.drop(columns=['Weighted_Price'])
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date')
df['Close'].fillna(method='ffill', inplace=True)
df['High'].fillna(df['Close'].shift(1), inplace=True)
df['Low'].fillna(df['Close'].shift(1), inplace=True)
df['Open'].fillna(df['Close'].shift(1), inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

df_final = pd.DataFrame()
df_final['High'] = df['High'].resample('D').max()
df_final['Low'] = df['Low'].resample('D').min()
df_final['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
df_final['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()


def first(array):
    """ Get the first value in a array. """
    return array[0]


def last(array):
    """ Get the last value in a array. """
    return array[-1]


df_final['Open'] = df['Open'].resample('D').apply(first)
df_final['Close'] = df['Close'].resample('D').apply(last)

df.plot()
# print(df.head())
