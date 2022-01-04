"""
Author: Kei
Date: January, 2022
"""
import sys
import joblib
import logging
from sklearn.model_selection import train_test_split

import config
from pipeline.preprocess import get_cleaned_data
from pipeline.train import train
from pipeline.evaluate import evaluate


logging.basicConfig(level=logging.INFO)

def run():
    """
    Main entry point of pipeline
    """
    logging.info("Loading and getting clean data..")
    data_df = get_cleaned_data(config.DATA_PATH)

    logging.info("Splitting data to train and test..")
    X_df = data_df
    y_df = X_df.pop('salary')
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=y_df)

    logging.info("Training and validating model..")
    best_model = train(
        config.MODEL,
        X_train,
        y_train,
        config.PARAM_GRID,
        config.FEATURES
    )

    logging.info("Testing model..")
    evaluate(best_model, X_test, y_test, "test")

    logging.info("Saving model..")
    joblib.dump(best_model, config.MODEL_PATH)


if __name__ == "__main__":
    run()
