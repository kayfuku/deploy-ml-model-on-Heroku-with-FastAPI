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

from config import MODEL_PATH, EXAMPLES_DIR
from app.schemas import Person

app = FastAPI(
    title="Udacity project 3",
    description="deploy ML model with FastAPI on Heroku",
    version="1.0",
)

model = joblib.load(MODEL_PATH)


@app.get("/")
async def greetings():
    return "Hello, welcome!"


@app.post("/predict")
async def predict(person: Person = Body()):
    person = person.dict()
