"""
FastAPI routing functions
Author: Kei
Date: January, 2022
"""
import os
from fastapi.param_functions import Body
import yaml
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body

from config import MODEL_PATH, EXAMPLES_PATH
from app.schemas import Person

app = FastAPI(
    title="Udacity project 3",
    description="deploy ML model with FastAPI on Heroku",
    version="1.0",
)

model = joblib.load(MODEL_PATH)
with open(EXAMPLES_PATH) as f:
    examples = yaml.safe_load(f)


@app.get('/')
async def greetings():
    return "Hello, welcome!"


@app.post('/predict')
async def predict(person: Person = Body(..., examples=examples)):
    person = person.dict()
    features = np.array(
        [person[f] for f in examples['features_columns']]).reshape(1, -1)
    df = pd.DataFrame(features, columns=examples['features_columns'])

    pred_label = model.predict(df)
    pred_probs = model.predict_proba(df)[:, 1]
    pred = '>50k' if pred_label == 1 else '<=50k'

    return {'label': pred_label, 'prob': pred_probs, 'salary': pred}
