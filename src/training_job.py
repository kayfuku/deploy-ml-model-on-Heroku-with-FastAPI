"""
Author: Kei
Date: January, 2022
"""
import sys
import joblib
import logging
from sklearn.model_selection import train_test_split

import config
import pipeline


logging.basicConfig(level=logging.INFO)

def run():
    """
    Main entry point
    """
    logging.info("Loading and getting clean data")
    X, y = pipeline.preprocessing.get_cleaned_data(config.DATA_PATH)




if __name__ == "__main__":
    run()
