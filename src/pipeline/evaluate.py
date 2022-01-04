"""
Functions needed to evaluate model.
Author: Kei
Date: January, 2022
"""
import logging
from sklearn.metrics import precision_score, recall_score, fbeta_score
# from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)


def compute_metrics(y_true, y_pred):
    """
    Compute precision, recall, and f1 scores
    Args:
        y_true: array, array of true labels
        y_pred: array, array of predicted labels
    Returns:
        precision: float, precision score
        recall: float, recall score
        f1: float, f1 score
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = fbeta_score(y_true, y_pred, beta=1)
    # print(classification_report(y_true, y_pred))

    return precision, recall, f1


def evaluate(model_pipe, X, y, split, file):
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

    logging.info(f"Evaluating on {split} data..")
    precision, recall, f1 = compute_metrics(y, y_pred)

    logging.info("Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
        precision, recall, f1))

    # Save the result in a file.
    file = file + '_' + split + '.txt'
    with open(file, 'w') as f:
        print(f"Evaluation on {split} data", file=f)
        print("Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
            precision, recall, f1), file=f)
