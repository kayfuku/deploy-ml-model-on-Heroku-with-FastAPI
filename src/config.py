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


__MAIN_DIR = Path(__file__).parent.parent.absolute()

model = RandomForestClassifier(
    random_state=RANDOM_STATE,
)
# model = LogisticRegression(
#     random_state=RANDOM_STATE
# )

PARAM_GRID = None
if isinstance(model, RandomForestClassifier):
    PARAM_GRID = {
        'model__n_estimators': list(range(50, 151, 25)),
        'model__max_depth': list(range(2, 11, 2)),
        'model__min_samples_leaf': list(range(1, 51, 5)),
    }
elif isinstance(model, LogisticRegression):
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

__DATA_FILE = 'cleaned_census.csv'


DATA_DIR = os.path.join(__MAIN_DIR, 'data', __DATA_FILE)

