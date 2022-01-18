# Deploying an ML Model on Heroku with FastAPI
The third project ML DevOps Engineer Nanodegree by Udacity.  

## Overview
This project is part of the Udacity course, "Machine Learning DevOps Engineer". Instructions are available in udacity's [repository](https://github.com/udacity/nd0821-c3-starter-code/tree/master/starter). The project is to build a machine learning application that predicts whether a person's annual income is more than $50K using the census income dataset from UCI. The application is deployed using FastAPI, with CI/CD using Github Actions and Heroku.
  
<img src='/screenshots/overview.jpg' width=800>
  
  
## Prerequisites
Python and Jupyter Notebook   
AWS account with S3 bucket  
Github account to use Github Actions for CI  
Heroku account for CD  
Linux environment may be needed within windows through WSL  
In addition to the following CLI tools  
  
AWS CLI  
Heroku CLI  
Dependencies  
This project dependencies are available in the requirements.txt file.  
  
## Installation
Use the package manager pip to install the dependencies from the requirements.txt. It's recommended to install it in a separate virtual environment.

```pip install -r requirements.txt```

## Usage
The config file contains MODEL variable of RandomForestClassifier. Each model with a set of parameters for the grid search PARAM_GRID. You can use your own model with the parameters needed. The SLICE_COLUMNS variable holds the columns for slice evaluation.

## 1. Starting training
```
cd src
python training_job.py
```
This saves a seralized model, generates evaluation results and slice evaluation results,

## 2. Starting FastAPI app
```
cd src
uvicorn app.api:app --reload
```
## 3. FastAPI app documentation

http://127.0.0.1:8000/docs  
<img src='/screenshots/fastapi_docs.jpg' width=800>


## 4. Testing the data and the model
```
cd src
pytest -vv
```
## 5. Showing tracked files with DVC
```
dvc dag
```
<img src='/screenshots/dvcdag.png' width=800>

## 6. CI using GitHub Actions will be triggered upon pushing to GitHub
```
git push
```
## 7. CD is enabled from within Heroku app settings

<img src='/screenshots/continuous_deployment.png' width=800>

## 8. Root API endpoint

http://127.0.0.1:8000/  
<img src='/screenshots/live_get.png' width=800>

## 9. Testing deployment on Heroku, demo post request
```
python request_heroku.py
```
<img src='/screenshots/live_post.png' width=800>

