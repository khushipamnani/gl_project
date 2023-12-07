#Importing Libraries
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump, load
import sys
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

#GetData
def get_data(file_path):
    data = pd.read_csv(file_path)
    del data['Id']
    return data
#Splitting Data
def data_split(data):
    le = preprocessing.LabelEncoder()
    le.fit(data['Species'])
    y = le.transform(data['Species'])
    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test
#Training
def train_model(X_train, y_train):
    logit = LogisticRegression(max_iter=1000)
    logit = logit.fit(X_train, y_train)
    return logit
#Saving Model
def save_model(logit):
    dump(logit, 'Model/my_model.joblib')
    print("Data saved")
#Creating Metrics
def create_metrics(X_test, y_test, logit):
    clf_report = classification_report(y_test, logit.predict(X_test))
    auc = roc_auc_score(y_test, logit.predict_proba(X_test), multi_class='ovr')
    return {'auc': auc, 'clf_report': clf_report}
#Saving Metrics
def save_metrics(metrics):
    with open('Model/metrics.json', 'w') as out_file:
        json.dump(metrics, out_file, sort_keys=True, indent = 4, ensure_ascii=False)

if __name__ == "__main__":
    file_path = 'Datasets/Iris.csv'
    data = get_data(file_path)
    X_train = data_split(data)[0]
    X_test = data_split(data)[1]
    y_train = data_split(data)[2]
    y_test = data_split(data)[3]
    logit = train_model(X_train, y_train)
    metrics = create_metrics(X_test, y_test, logit)
    save_model(logit)
    save_metrics(metrics)