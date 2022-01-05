"""
Author: Kei
Date: January, 2022
"""
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils import estimator_html_repr

import config
from pipeline.preprocess import get_cleaned_data
from pipeline.train import train
from pipeline.evaluate import evaluate
from pipeline.slice import evaluate_slices


logging.basicConfig(level=logging.INFO)


def run():
    """
    Main entry point of pipeline
    """
    # Load and get data.
    logging.info("Loading and getting clean data..")
    data_df = get_cleaned_data(config.DATA_PATH)

    # Split data.
    logging.info("Splitting data to train and test..")
    X_df = data_df
    y_df = X_df.pop('salary')
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y_df)

    # Train model.
    logging.info("Training and validating model..")
    best_model = train(
        config.MODEL,
        X_train,
        y_train,
        config.PARAM_GRID,
        config.FEATURES
    )

    # Evaluate on train data.
    evaluate(best_model, X_train, y_train, "train", config.EVAL_PATH)

    logging.info("Evaluating model on slices of data..")
    for col in config.SLICE_COLUMNS:
        evaluate_slices(best_model, col, X_train, y_train,
                        "train", config.EVAL_SLICE_PATH)

    # Evaluate on test data.
    logging.info("Testing model..")
    evaluate(best_model, X_test, y_test, "test", config.EVAL_PATH)

    logging.info("Evaluating model on slices of data..")
    for col in config.SLICE_COLUMNS:
        evaluate_slices(best_model, col, X_test, y_test,
                        "test", config.EVAL_SLICE_PATH)

    # Save model.
    logging.info("Saving model..")
    joblib.dump(best_model, config.MODEL_PATH)

    # Save pipeline as html format.
    with open(config.MODEL_PATH + '.html', 'w') as f:
        f.write(estimator_html_repr(best_model))


if __name__ == "__main__":
    run()
