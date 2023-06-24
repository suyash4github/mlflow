import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score

import argparse

def get_data():
    URL='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    try:
        df=pd.read_csv(URL,sep=';')
        return df
    except Exception as e:
        raise e
def evaluate(y_true,y_pred):
    '''mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_true,y_pred)'''
    accuracy=accuracy_score(y_true,y_pred)
    return accuracy

def main(n_estimators,max_depth):
    df=get_data()
    train,test=train_test_split(df)
    X_train=train.drop(['quality'],axis=1)
    X_test=test.drop(['quality'],axis=1)
    y_train=train['quality']
    y_test=test['quality']


    '''lr=ElasticNet()
    lr.fit(X_train,y_train)
    pred=lr.predict(X_test)
    mae,mse,rmse,r2=evaluate(y_test,pred)
    print(f'mae={mae},mse={mse},rmse={rmse},r2={r2}')'''
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(X_train,y_train)
    pred=rf.predict(X_test)
    acc=evaluate(y_test,pred)
    print(acc)
if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n",default=50,type=int)
    args.add_argument("--max_depth","-m",default=5,type=int)
    parse_args=args.parse_args()
    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
    except Exception as e:
        raise e
