"""
Functions needed to preprocess data.
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
        data_df: pandas dataframe, cleaned data
    """

    data_df = pd.read_csv(path)

    data_df = data_df.drop_duplicates()

    # Change column names to use '_' instead of '-' to use them as variables.
    columns = data_df.columns
    columns = [col.replace('-', '_') for col in columns]
    data_df.columns = columns

    # # make all characters to be lowercase in string columns
    # data_df = data_df.applymap(lambda s: s.lower() if isinstance(s, str) else s)

    # Map label salary to numbers.
    data_df['salary'] = data_df['salary'].map({'>50K': 1, '<=50K': 0})

    return data_df
