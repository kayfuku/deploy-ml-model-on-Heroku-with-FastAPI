"""
Test functions for dataset
Author: Kei
Date: January, 2022
"""
import pandas as pd
import numpy as np

def test_column_presence_and_type(data):
    """
    Test if all expected columns exist
    Args:
        data: Pandas DataFrame, data to be tested
    """
    columns = {
        'age': pd.api.types.is_integer_dtype,
        'workclass': pd.api.types.is_string_dtype,
        'fnlgt': pd.api.types.is_integer_dtype,
        'education': pd.api.types.is_string_dtype,
        'education_num': pd.api.types.is_integer_dtype,
        'marital_status': pd.api.types.is_string_dtype,
        'occupation': pd.api.types.is_string_dtype,
        'relationship': pd.api.types.is_string_dtype,
        'race': pd.api.types.is_string_dtype,
        'sex': pd.api.types.is_string_dtype,
        'capital_gain': pd.api.types.is_integer_dtype,
        'capital_loss': pd.api.types.is_integer_dtype,
        'hours_per_week': pd.api.types.is_integer_dtype,
        'native_country': pd.api.types.is_string_dtype,
        'salary': pd.api.types.is_integer_dtype,
    }

    # Check if every element in columns.keys() is in data.columns.values.
    assert set(data.columns.values).issuperset(set(columns.keys()))

    # Check that the columns are of the right dtype.
    for col, type_verification_func in columns.items():
        assert type_verification_func(data[col]), f"Column {col} failed test {type_verification_func}"


def test_education_num_column(data):
    """
    Test if education num column values are within correct range.
    Args:
        data: Pandas DataFrame, data to be tested
    """
    bools = data['education_num'].between(1, 17)
    assert np.sum(~bools) == 0


def test_marital_status_column(data):
    """
    Test if marital status column values are with correct categories.
    Args:
        data: Pandas DataFrame, data to be tested
    """
    categories = ['Married-civ-spouse', 'Never-married', 'Divorced',
        'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']

