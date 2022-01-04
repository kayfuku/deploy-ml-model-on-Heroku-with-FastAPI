"""
Author: Kei
Date: January, 2022
"""
import os
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 41
TEST_SIZE = 0.2
DATA_FILE = 'edited_census.csv'

MODEL = RandomForestClassifier(
    random_state=RANDOM_STATE,
)
# MODEL = LogisticRegression(
#     random_state=RANDOM_STATE
# )

PARAM_GRID = None
if isinstance(MODEL, RandomForestClassifier):
    # To tell the GridSearchCV which part of pipeline the parameters are used in,
    # we need '{name of part of pipeline}__' before parameter name.
    PARAM_GRID = {
        'model__n_estimators': [5],
        'model__max_depth': [10]
        # 'model__n_estimators': list(range(50, 151, 25)),
        # 'model__max_depth': list(range(2, 11, 2)),
        # 'model__min_samples_leaf': list(range(1, 51, 5)),
    }
elif isinstance(MODEL, LogisticRegression):
    PARAM_GRID = {
        'model__C': np.linspace(0.1, 10, 3)
    }

FEATURES = {
    'categorical': [
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'workclass',
        'native_country'
    ],
    'numeric': [
        'age',
        'fnlgt',
        'capital_gain',
        'capital_loss',
        'hours_per_week'
    ],
    'drop': ['education']
}

__MAIN_DIR = Path(__file__).parent.parent.absolute()

EVAL_FILE = 'eval_' + MODEL.__class__.__name__
MODEL_FILE = 'pipe_' + MODEL.__class__.__name__

DATA_PATH = os.path.join(__MAIN_DIR, 'data', DATA_FILE)
EVAL_PATH = os.path.join(__MAIN_DIR, 'eval', EVAL_FILE)
MODEL_PATH = os.path.join(__MAIN_DIR, 'models', MODEL_FILE)

