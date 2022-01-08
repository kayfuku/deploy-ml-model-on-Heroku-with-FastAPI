"""
Test FastAPI APIs.
Author: Kei
Date: January, 2022
"""
from app.app import app
from fastapi.testclient import TestClient
from http import HTTPStatus
import pytest

client = TestClient(app)


def test_greetings():
    """
    Test greetings function, GET /
    """
    response = client.get('/')
    assert response.status_code == HTTPStatus.OK
    assert response.json() == "Hello, welcome!"


def test_predict():
    """
    Test predict function, POST /predict
    """
    data = {
        'age': 40,
        'fnlgt': 1000,
        'education_num': 14,
        'capital_gain': 200,
        'capital_loss': 0,
        'hours_per_week': 40,
    }
    response = client.post('/predict', json=data)
    # print(response.json())
    assert response.status_code == HTTPStatus.OK
    assert int(response.json()['label']) == 0 or \
        int(response.json()['label']) == 1
    assert float(response.json()['prob']) >= 0 and \
        float(response.json()['label']) <= 1
