"""
Functions needed to train model.
Author: Kei
Date: January, 2022
"""
import logging

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV

from pipeline.evaluate import evaluate

logging.basicConfig(level=logging.INFO)


def create_model_pipeline(model, features):
    """
    Create a model pipeline with given model and features.
    Args:
        model: sklearn model, model algorithm
        features: dict, features
    Returns:
        model_pipe: model pipeline
    """
    assert isinstance(model, (RandomForestClassifier, LogisticRegression)), \
        "Model should be RandomForestClassifier or LogisticRegression"

    categorical_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )

    numeric_preproc = StandardScaler()

    features_preproc = ColumnTransformer([
        ('drop', 'drop', features['drop']),
        ('categorical', categorical_preproc, features['categorical']),
        ('numeric', numeric_preproc, features['numeric'])
    ],
        remainder='passthrough'
    )

    # model pipeline
    model_pipe = Pipeline([
        ('features_preprocessor', features_preproc),
        ('model', model)
    ])

    return model_pipe


def perform_grid_search(model, X_train, y_train, param_grid):
    """
    Perform gridsearch on a model to choose best parameters.
    Args:
        model: sklearn model or pipeline
        X_train: pandas dataframe, train features
        y_train: pandas dataframe, train labels
        param_grid: dict, parameters for grid search
    Returns:
        best_model: sklearn model or pipeline, best model based on grid search
    """
    g_search = GridSearchCV(
        model,
        param_grid,
        scoring='f1',
        cv=5,
        error_score='raise',
        n_jobs=-1,
        verbose=2
    )

    g_search.fit(X_train, y_train)

    return g_search.best_estimator_


def train(model, X_train, y_train, param_grid, features):
    """
    Train a model with given data and model pipeline.
    Args:
        model: sklearn model, model algorithm
        X_train: pandas dataframe, train features
        y_train: pandas dataframe, train labels
        param_grid: dict, parameters for grid search
        features: dict, features
    Returns:
        best_model: best model based on grid search
    """
    # Create a model pipeline.
    logging.info("Creating model pipeline..")
    model_pipe = create_model_pipeline(model, features)

    # Perform grid search.
    logging.info("Performing grid search..")
    best_model = perform_grid_search(model_pipe, X_train, y_train, param_grid)

    return best_model
