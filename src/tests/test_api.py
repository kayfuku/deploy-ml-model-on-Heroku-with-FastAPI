"""
Test FastAPI APIs.
Author: Kei
Date: January, 2022
"""
from app.app import app
from fastapi.testclient import TestClient
from http import HTTPStatus

client = TestClient(app)


def test_greetings():
    """
    Test greetings function, GET /
    """
    response = client.get('/')
    assert response.status_code == HTTPStatus.OK
    assert response.json() == "Hello, welcome!!"


def test_predict_label_1():
    """
    Test predict function, POST /predict, with data to predict label 1
    """
    data = {'age': 41,
            'workclass': 'Private',
            'fnlgt': 153031,
            'education': 'Some-college',
            'education_num': 10,
            'marital_status': 'Married-civ-spouse',
            'occupation': 'Sales',
            'relationship': 'Husband',
            'race': 'White',
            'sex': 'Male',
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 65,
            'native_country': 'United-States',
            }

    response = client.post('/predict', json=data)
    # print(response.json())
    assert response.status_code == HTTPStatus.OK  # 200
    assert int(response.json()['label']) == 1


def test_predict_label_0():
    """
    Test predict function, POST /predict, with data to predict label 0
    """
    data = {'age': 26,
            'workclass': 'Private',
            'fnlgt': 108019,
            'education': 'HS-grad',
            'education_num': 9,
            'marital_status': 'Never-married',
            'occupation': 'Craft-repair',
            'relationship': 'Own-child',
            'race': 'White',
            'sex': 'Male',
            'capital_gain': 3325,
            'capital_loss': 0,
            'hours_per_week': 40,
            'native_country': 'United-States'
            }

    response = client.post('/predict', json=data)
    # print(response.json())
    assert response.status_code == HTTPStatus.OK  # 200
    assert int(response.json()['label']) == 0


def test_invalid_predict():
    """
    Test predict function with invalid request, POST /predict
    """
    data = {
        'age': 40,
    }
    response = client.post('/predict', json=data)
    # print(response.json())
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY  # 422
