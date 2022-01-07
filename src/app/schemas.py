"""
Functions needed to train model.
Author: Kei
Date: January, 2022
"""
from typing import Optional
from pydantic import BaseModel


class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: Optional[str] = None
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
