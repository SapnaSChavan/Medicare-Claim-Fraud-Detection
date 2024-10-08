import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer

# Machine Learning Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
from sklearn.metrics import make_scorer

from sklearn.metrics import classification_report, accuracy_score
import random
random.seed(100)
import pickle
import time
import pyodbc
print(pyodbc.drivers())


data = pd.read_csv(r'../data/interim/model_data_new.csv')
print(data.head())

del data['Unnamed: 0']
print(data.columns)

imp_cols = ['BeneID', 'ClaimID', 'Provider', 'InscClaimAmtReimbursed',
       'DeductibleAmtPaid', 'ClaimPeriod', 'TimeInHptal', 'Diagnosis Count',
       'Procedures Count', 'SamePhysician', 'OPD_Flag', 'PotentialFraud',
        'Gender', 'Race', 'RenalDiseaseIndicator', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
       'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
       'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
       'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
       'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
       'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
       'ChronicCond_stroke', 'IPAnnualReimbursementAmt',
       'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
       'OPAnnualDeductibleAmt',  'Age',
       'ChronicDisease_Count', 'Total_Claims_Per_Bene',
       'Avg_Reimbursement_Per_Bene', 'Age_At_Claim',
       'Multiple_Chronic_Conditions', 'Claim_To_Deductible_Ratio',
       'Total_Annual_Reimbursement', 'Avg_Reimbursement_By_Provider',
       'Provider_Claim_Frequency', 'High_Risk_Provider']

data['PotentialFraud'].value_counts()

data = data[imp_cols]
data.shape

data.select_dtypes(include='object').columns

### One-hot encoding
cat_cols = ['SamePhysician', 'OPD_Flag', 'Gender',
       'Race', 'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke']

data_encoded = pd.get_dummies(data, columns = cat_cols,drop_first=True)
data_encoded.shape

data_encoded.head(2)
del data_encoded['BeneID']

data_encoded['PotentialFraud'].value_counts(normalize=True)

# X = data_encoded.drop(columns='PotentialFraud')
X = data_encoded[['ClaimID', 'Provider','Total_Claims_Per_Bene',
    'TimeInHptal',
    'Provider_Claim_Frequency',
    'ChronicCond_stroke_Yes',
    'DeductibleAmtPaid',
    'NoOfMonths_PartBCov',
    'NoOfMonths_PartACov',
    'OPD_Flag_Yes',
    'Diagnosis Count',
    'ChronicDisease_Count', 'Age']]
y = data_encoded['PotentialFraud']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42, stratify=y)
print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)


#########################################################################################################
def drop_columns(X):
    """
    Drops specified columns from the input DataFrame.

    Parameters:
    X (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: DataFrame after dropping the specified columns.
    """
    return X.drop(['ClaimID', 'Provider'], axis=1)

drop_columns_transformer = FunctionTransformer(drop_columns)


# Define the pipeline
pipeline = Pipeline([
    ('drop_columns', drop_columns_transformer),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=0.01, penalty='l2', solver='liblinear'))
])
# Train the model
start_time = time.time()
pipeline.fit(X_train, y_train)
end_time = time.time()
print('Execution time: ', end_time - start_time)

# Predictions
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Metrics for Training Data
print("\nLogistic Regression Training Report:")
print(classification_report(y_train, y_pred_train))
print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train)}")
print(f"Training Precision: {precision_score(y_train, y_pred_train)}")
print(f"Training Recall: {recall_score(y_train, y_pred_train)}")
print(f"Training F1 Score: {f1_score(y_train, y_pred_train)}")

# Metrics for Test Data
print("\nLogistic Regression Test Report:")
print(classification_report(y_test, y_pred_test))
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")
print(f"Test Precision: {precision_score(y_test, y_pred_test)}")
print(f"Test Recall: {recall_score(y_test, y_pred_test)}")
print(f"Test F1 Score: {f1_score(y_test, y_pred_test)}")

# Feature Importance
importances = pipeline.named_steps['classifier'].coef_[0]
feature_names = X.drop(['ClaimID', 'Provider'], axis=1).columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importance_df)

# Save the pipeline
model_filename = 'fraud_claim_logistic_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(pipeline, file)
print(f"Model pipeline saved as {model_filename}")



