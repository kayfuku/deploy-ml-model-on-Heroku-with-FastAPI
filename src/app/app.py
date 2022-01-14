"""
FastAPI routing functions
Author: Kei
Date: January, 2022
"""
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body

from config import __MAIN_DIR, MODEL_PATH, EXAMPLES_PATH
from app.schemas import Person

# print("app.py start!!")

# To use dvc on Heroku
# Check if the host is Heroku server and this project is dvc initialized.
if "DYNO" in os.environ and os.path.isdir(os.path.join(__MAIN_DIR, '.dvc')):
    print(".dvc detected!")
    # os.system("rm -r " + os.path.join(__MAIN_DIR, '.dvc/tmp/lock'))
    os.system("dvc config core.no_scm true")
    # os.system("dvc remote add -f -d s3-bucket s3://bucket-demo-fastapi")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r " +
              os.path.join(__MAIN_DIR, '.dvc') + " " +
              os.path.join(__MAIN_DIR, '.apt/usr/lib/dvc'))

app = FastAPI(
    title="Udacity project 3",
    description="deploy ML model with FastAPI on Heroku",
    version="1.0",
)

model = joblib.load(MODEL_PATH)
with open(EXAMPLES_PATH) as f:
    examples = yaml.safe_load(f)

# print(examples)


@ app.get('/')
async def greetings():
    return "Hello, welcome!"


@ app.post('/predict')
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
    print("pred:", pred)
    ret = {
        "label": str(pred_label),
        "prob": str(pred_probs),
        "salary": str(pred)
    }
    print("predicted label:", ret['label'])

    return ret
