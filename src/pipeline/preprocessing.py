"""
Author: Kei
Date: January, 2022
"""
import pandas as pd


def get_cleaned_data(path):
    """
    Load and clean the data from a given path
    Args:
        path: str, path to the data
    Returns:
        X_df: pandas dataframe, features from the data
        y_df: pandas dataframe, labels from the data
    """

    data_df = pd.read_csv(path)

    data_df = data_df.drop_duplicates()

    # # change column names to use _ instead of -
    # columns = data_df.columns
    # columns = [col.replace('-', '_') for col in columns]
    # data_df.columns = columns

    # # make all characters to be lowercase in string columns
    # data_df = data_df.applymap(lambda s: s.lower() if isinstance(s, str) else s)

    # map label salary to numbers
    data_df['salary'] = data_df['salary'].map({'>50K': 1, '<=50K': 0})

    X_df = data_df
    y_df = X_df.pop('salary')

    return X_df, y_df

