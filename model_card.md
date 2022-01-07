# Model Card

## Model Details
• This model is trained to predict whether a person makes over 50K a year.  
• Random Forest will be used to develop the model.  
• Kei created the model. This project is a part of Udacity Machine Learning DevOps Engineer nanodegree program.  
  
<img src='/screenshots/model_architecture.jpg' width=561, height=276>
  
## Intended Use  
• Intended to be used to determine whether a person makes over 50K a year or not.  
  
## Training Data  
• ![Census Income Data Set] from UCI (https://archive.ics.uci.edu/ml/datasets/census+income)  
• Features: 'age', 'workclass', 'fnlgt', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex','capital-gain', 'capital-loss', 'hours-per-week', 'native-country'  
• Target: 'salary'  
• For categrorical data, missing values were imputed by most frequent value, and OneHotEncoder was used to encode.  
• For numeric data, StandardScaler was applied.  
• Random Forest algorithm was used, which is an ensemble of decision trees with bagging method.  
  
## Evaluation Data  
• I used 5 folds cross validation with some combinations of parameters grid search.  
  
## Metrics
• The model was evaluated based on F1 score, which uses Precision and Recall.  
  
## Ethical Considerations
• The dataset is open sourced on UCI machine learning repository for educational purposes.  
  
## Caveats and Recommendations  
• Further error analysis and exploring hyperparameters and model architectures should improve the inference performance.  
  
  
  
