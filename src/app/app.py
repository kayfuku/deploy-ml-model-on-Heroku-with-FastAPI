"""
FastAPI routing functions
Author: Kei
Date: January, 2022
"""
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

# print(examples)


@app.get('/')
async def greetings():
    return "Hello, welcome!"


@app.post('/predict')
async def predict(person: Person = Body(..., example=examples['post_examples'])):
    print("predict() called.")
    person = person.dict()
    print(person)
    features = np.array(
        [person[f] for f in examples['features_columns']]).reshape(1, -1)
    df = pd.DataFrame(features, columns=examples['features_columns'])

    pred_label = model.predict(df)[0]
    # print(pred_label)
    pred_probs = model.predict_proba(df)[0][pred_label]
    # print(pred_probs)
    pred = '>50k' if pred_label == 1 else '<=50k'
    print(pred)
    ret = {
        "label": str(pred_label),
        "prob": str(pred_probs),
        "salary": str(pred)
    }

    return ret
