"""
Functions needed to evaluate model.
Author: Kei
Date: January, 2022
"""
import sys
import logging
from sklearn.metrics import fbeta_score, precision_score, recall_score, classification_report
import config

logging.basicConfig(level=logging.INFO)


def compute_metrics(y_true, y_pred):
    """
    Compute precision, recall, and f1 scores
    Args:
        y_true: array, array of true labels
        y_pred: array, array of predicted labels
    Returns:
        f1: float, f1 score
        precision: float, precision score
        recall: float, recall score
    """
    f1 = fbeta_score(y_true, y_pred, beta=1)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    # print(classification_report(y_true, y_pred))

    return f1, precision, recall


def evaluate(model_pipe, X, y, split):
    """
    Evaluate a model on given data split and save the results.
    Args:
        model_pipe: sklearn model or pipeline
        X: pandas dataframe, features
        y: pandas series, labels
        split: str, train or test split
        file: file object
    """
    logging.info("Running inference..")
    y_pred = model_pipe.predict(X)

    logging.info("Evaluating model..")
    f1, precision, recall = compute_metrics(y, y_pred)

    logging.info(f"Evaluating on {split} data..")
    logging.info("F1: {:.3f}, Precision: {:.3f}, Recall: {:.3f}".format(
        f1, precision, recall))

    # Save the result in a file.
    file = config.EVAL_PATH + '_' + split + '.txt'
    with open(file, 'w') as f:
        print(f"Evaluation on {split} data", file=f)
        print("F1: {:.3f}, Precision: {:.3f}, Recall: {:.3f}".format(
            f1, precision, recall), file=f)

