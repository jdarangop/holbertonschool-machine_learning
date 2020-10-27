#!/usr/bin/env python3
""" From Dictionary """
import pandas as pd


df = pd.DataFrame({'First': pd.Series([0.0, 0.5, 1.0, 1.5]),
                   'Second': pd.Series(['one', 'two', 'three', 'four'])})
df.index = ['A', 'B', 'C', 'D']
