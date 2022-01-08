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
    Test GET /
    """
    response = client.get('/')
    assert response.status_code == HTTPStatus.OK
    assert response.json() == "Hello, welcome!"
