"""
Functions needed to evaluate model on a slice of data.
Author: Kei inspired by and learned a lot from Ibrahim
Date: January, 2022
"""
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pipeline.evaluate import compute_metrics

sns.set()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def compute_slice_metrics(column, X, y_true, y_pred):
    """
    Calculate metrics on a slice of data for a given column
    Args:
        column: str, Column name representing a feature
        X: pandas dataframe, features
        y_true: true labels
        y_pred: predicted labels
    Returns:
        Pandas dataframe with metrics for each category
    """
    df = pd.concat([X[column].copy(), y_true], axis=1)
    df['salary_pred'] = y_pred

    metrics = []
    for category in df[column].unique():
        precision, recall, f1 = compute_metrics(
            df[df[column] == category]['salary'],
            df[df[column] == category]['salary_pred']
        )
        metrics.append([category, precision, recall, f1])
        # print(f"[INFO] {categ}: Precision = {prec:.3f}, Recall = {rec:.3f}, F1 = {f1:.3f}")

    return pd.DataFrame(
        metrics,
        columns=[
            'Category',
            'Precision',
            'Recall',
            'F1'])


def evaluate_slices(model_pipe, column, X, y, split, file):
    """
    Evaluate model on a slice of data for a given column and data split and
    save the results to a file.
    Args:
        model_pipe: sklearn model or pipeline
        column: str, column name used for slicing
        X: pandas dataframe, features
        y: pandas series, labels
        split: str, train or test split
        file: file object
    """
    logging.info(f"Evaluating {column} on slice of {split} data..")

    y_pred = model_pipe.predict(X)
    slice_df = compute_slice_metrics(column, X, y, y_pred)

    file = file + '_slice_metrics_' + column + '_' + split
    plot_slice_metrics(
        slice_df,
        f"{column} column for {split} data",
        file
    )

    # Save the result in a file.
    file = file + '.txt'
    with open(file, 'w') as f:
        print(f"Model evaluation on {column} slice of {split} data", file=f)
        print(slice_df.to_string(index=False), file=f)
        print("", file=f)


def plot_slice_metrics(df, title, save_path=None):
    """
    Plot slice metrics in a bar plot using the dataframe from
    compute_slice_metrics function
    Args:
        df: pandas dataframe, dataframe of metrics for each category
        title: str, plot title
        save_path: str, optional, the plot save path. Defaults to None.
    """
    df = df.melt(id_vars=['Category'], value_vars=[
                 'Precision', 'Recall', 'F1'])

    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x='variable', y='value', hue='Category', data=df)

    ax.set(title=title)
    ax.legend(loc='lower right')
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

    for bars in ax.containers:
        labels = [f'{bar.get_height():.3f}' for bar in bars]
        ax.bar_label(bars, labels=labels, label_type='edge')
        if save_path:
            plt.savefig(f'{save_path}.png')
