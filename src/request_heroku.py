import requests

data = {
    "age": 40,
    "workclass": "state-gov",
    "fnlgt": 1000,
    "education": "bachelors",
    "education_num": 14,
    "marital_status": "never-married",
    "occupation": "adm-clerical",
    "relationship": "not-in-family",
    "race": "white",
    "sex": "male",
    "capital_gain": 2500,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "united-states",
}

# GET request
response = requests.get(
    'https://deploy-ml-model-with-github.herokuapp.com/')
print(response.status_code)
print(response.json())

# POST request
response = requests.post(
    'https://deploy-ml-model-with-github.herokuapp.com/predict/',
    auth=('user', 'pass'),
    json=data)
print(response.status_code)
print(response.json())
