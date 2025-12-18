import os
import pandas as pd
import numpy as np
import scipy.io as sio

def df_to_mat(df: pd.DataFrame):
    structured_array = np.empty(len(df), dtype=[
        (col, 'O') for col in df.columns
    ])
    for i, row in df.iterrows():
        structured_array[i] = tuple(row[col] for col in df.columns)
    return structured_array