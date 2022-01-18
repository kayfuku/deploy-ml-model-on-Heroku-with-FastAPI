#Deploying-a-ML-Model-on-Heroku-with-FastAPI
The third project ML DevOps Engineer Nanodegree by Udacity. Instructions are available in udacity's repository

##Description
This project is part of Unit 4: Deploying a Scalable ML Pipeline in Production. The problem is to build a machine learning application that predicts an employer's annual income more than $50K using the census income dataset from UCI. The application is deployed using FastAPI, with CI and CD using Github Actions and Heroku respectively.

##Prerequisites
Python and Jupyter Notebook are required
AWS account with S3 bucket
Github account to use Github Actions for CI
Heroku account for CD
Linux environment may be needed within windows through WSL
In addition to the following CLI tools
  
AWS CLI
Heroku CLI
Dependencies
This project dependencies is available in the requirements.txt file.

##Installation
Use the package manager pip to install the dependencies from the requirements.txt. Its recommended to install it in a separate virtual environment.

'''pip install -r requirements.txt  
  
##Project Structure
deploy-ml-model-on-Heroku-with-FastAPI
├── Aptfile
├── Procfile
├── README.md
├── data
│   ├── census.csv
│   ├── census.csv.dvc
│   ├── edited_census.csv
│   └── edited_census.csv.dvc
├── eval
│   ├── eval_RandomForestClassifier_test.txt
│   ├── eval_RandomForestClassifier_train.txt
│   ├── slice_output_RandomForestClassifier_slice_metrics_race_test.png
│   ├── slice_output_RandomForestClassifier_slice_metrics_race_test.txt
│   ├── slice_output_RandomForestClassifier_slice_metrics_race_train.png
│   ├── slice_output_RandomForestClassifier_slice_metrics_race_train.txt
│   ├── slice_output_RandomForestClassifier_slice_metrics_sex_test.png
│   ├── slice_output_RandomForestClassifier_slice_metrics_sex_test.txt
│   ├── slice_output_RandomForestClassifier_slice_metrics_sex_train.png
│   └── slice_output_RandomForestClassifier_slice_metrics_sex_train.txt
├── model_card.md
├── models
│   ├── pipe_RandomForestClassifier
│   ├── pipe_RandomForestClassifier.dvc
│   ├── pipe_RandomForestClassifier.html
│   └── pipe_RandomForestClassifier.jpg
├── notebooks
│   └── EDA.ipynb
├── requirements.txt
├── screenshots
│   ├── continuous_deployment.png
│   ├── continuous_integration.png
│   ├── dvcdag.png
│   ├── example.png
│   ├── live_get.png
│   ├── live_post.png
│   └── model_architecture.jpg
└── src
    ├── __pycache__
    │   └── config.cpython-38.pyc
    ├── app
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-38.pyc
    │   │   ├── app.cpython-38.pyc
    │   │   └── schemas.cpython-38.pyc
    │   ├── app.py
    │   ├── examples.yaml
    │   └── schemas.py
    ├── config.py
    ├── pipeline
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-38.pyc
    │   │   ├── evaluate.cpython-38.pyc
    │   │   ├── preprocess.cpython-38.pyc
    │   │   ├── preprocessing.cpython-38.pyc
    │   │   ├── slice.cpython-38.pyc
    │   │   └── train.cpython-38.pyc
    │   ├── evaluate.py
    │   ├── preprocess.py
    │   ├── slice.py
    │   └── train.py
    ├── request_heroku.py
    ├── tests
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-38.pyc
    │   │   ├── conftest.cpython-38-pytest-6.2.5.pyc
    │   │   ├── test_api.cpython-38-pytest-6.2.5.pyc
    │   │   ├── test_api.cpython-38.pyc
    │   │   └── test_data.cpython-38-pytest-6.2.5.pyc
    │   ├── conftest.py
    │   ├── test_api.py
    │   └── test_data.py
    └── training_job.py

##Usage
The config file contains MODEL variable with a choice of either LogisticRegression or RandomForestClassifier. Each model with a set of parameters for the grid search PARAM_GRID. You can your own model with the parameters needed. The SLICE_COLUMNS variable holds the columns for slice evaluation.

##1. Start training

cd src
python training_job.py
This saves a seralized model, generates evaluation metrics, slice evaluation metrics and figures,

2- Start FastAPI app

cd src
uvicorn app.api:app --reload
3- FastAPI app documentation to test the API from the browser

http://127.0.0.1:8000/docs


4- Testing the project

cd src
pytest -vv
5- Showing tracked files with DVC

dvc dag


6- CI using github action will be triggered upon pushing to github

git push
7- CD is enabled from within Heroku app settings



8- Starting the app on Heroku



9- Test deployment on Heroku, demo post request

python request_heroku.py
