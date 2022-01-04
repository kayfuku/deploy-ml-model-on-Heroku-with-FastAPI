"""
Conftest data used with pytest.
Author: Kei
Date: January, 2022
"""
import os
import pytest
# import great_expectations as ge

import config
from pipeline.preprocess import get_cleaned_data


@pytest.fixture(scope='session')
def data():
    """
    Prepare data for tests
    Returns:
        data_df: Pandas DataFrame, data to be tested
    """
    if not os.path.exists(config.DATA_PATH):
        pytest.fail(f"Data not found at path: {config.DATA_PATH}")

    data_df = get_cleaned_data(config.DATA_PATH)
    # data_df = ge.from_pandas(data_df)

    return data_df
