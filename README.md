# Deploying-a-ML-Model-on-Heroku-with-FastAPI
The third project ML DevOps Engineer Nanodegree by Udacity. Instructions are available in udacity's repository

## Description
This project is part of Unit 4: Deploying a Scalable ML Pipeline in Production. The problem is to build a machine learning application that predicts an employer's annual income more than $50K using the census income dataset from UCI. The application is deployed using FastAPI, with CI and CD using Github Actions and Heroku respectively.

## Prerequisites
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

## Installation
Use the package manager pip to install the dependencies from the requirements.txt. Its recommended to install it in a separate virtual environment.

```pip install -r requirements.txt```

## Usage
The config file contains MODEL variable with a choice of either LogisticRegression or RandomForestClassifier. Each model with a set of parameters for the grid search PARAM_GRID. You can your own model with the parameters needed. The SLICE_COLUMNS variable holds the columns for slice evaluation.

## 1. Start training
```
cd src
python training_job.py
```
This saves a seralized model, generates evaluation metrics, slice evaluation metrics and figures,

## 2. Start FastAPI app
```
cd src
uvicorn app.api:app --reload
```
## 3. FastAPI app documentation to test the API from the browser

http://127.0.0.1:8000/docs
<img src='/screenshots/fastapi_docs.png' width=400>


## 4. Testing the project
```
cd src
pytest -vv
```
## 5. Showing tracked files with DVC
```
dvc dag
```
<img src='/screenshots/dvcdag.png' width=400>

## 6. CI using github action will be triggered upon pushing to github
```
git push
```
## 7. CD is enabled from within Heroku app settings

<img src='/screenshots/continuous_deployment.png' width=400>

## 8. Root API endpoint

http://127.0.0.1:8000/
<img src='/screenshots/live_get.png' width=400>

## 9. Test deployment on Heroku, demo post request
```
python request_heroku.py
```
