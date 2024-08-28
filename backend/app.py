import streamlit as st
from fastapi import FastAPI, Depends
import numpy as np
import pandas as pd
from sqlalchemy.orm import session
from typing import Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import pickle

from config import SessionLocal, engine
from db_models import ClientDB
from schemas import Client

app = FastAPI()


def get_session():
    with SessionLocal() as session:
        return session


def get_data(db: session) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    table = db.query(ClientDB)
    df = pd.read_sql(table.statement, engine)

    df = df.dropna(subset='TARGET')

    X = df.drop(['TARGET', 'AGREEMENT_RK'], axis=1)
    y = df['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


@app.get('/fit_lr')
def fit_model_lr(db: session = Depends(get_session)) -> float:
    X_train, X_test, y_train, y_test = get_data(db)

    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GridSearchCV(
            LogisticRegression(),
            param_grid={'C': np.arange(0.1, 10, 0.1)},
            n_jobs=-1,
            cv=5,
            scoring='roc_auc'
        ))
    ])

    pipe_lr.fit(pd.DataFrame(X_train, columns=X_train.columns), y_train)

    with open('model_lr.pickle', 'wb') as f:
        pickle.dump(pipe_lr, f)

    y_pred = pipe_lr.predict(X_test)

    score = roc_auc_score(y_test, y_pred)

    return round(score, 3)


@app.get('/fit_svm')
def fit_model_lr(db: session = Depends(get_session)) -> float:
    X_train, X_test, y_train, y_test = get_data(db)

    pipe_cvm = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GridSearchCV(
            SVC(),
            param_grid={'C': np.arange(0.1, 5, 0.5),
                        'kernel': ['linear', 'rbf']},
            cv=2,
            scoring='roc_auc',
        ))
    ])

    pipe_cvm.fit(pd.DataFrame(X_train, columns=X_train.columns), y_train)

    with open('model_svm.pickle', 'wb') as f:
        pickle.dump(pipe_cvm, f)

    y_pred = pipe_cvm.predict(X_test)

    score = roc_auc_score(y_test, y_pred)

    return round(score, 3)


@app.post('/predict_lr')
def predict_model_lr(client: Client) -> int:
    with open('model_lr.pickle', 'rb') as f:
        model = pickle.load(f)

    data = np.array([[client.AGE,
                      client.SOCSTATUS_WORK_FL,
                      client.SOCSTATUS_PENS_FL,
                      client.GENDER,
                      client.CHILD_TOTAL,
                      client.DEPENDANTS,
                      client.PERSONAL_INCOME,
                      client.LOAN_NUM_TOTAL,
                      client.LOAN_NUM_CLOSED]])

    prediction = model.predict(data)

    return prediction


@app.post('/predict_svm')
def predict_model_lr(client: Client) -> int:
    with open('model_svm.pickle', 'rb') as f:
        model = pickle.load(f)
    data = np.array([[client.AGE,
                      client.SOCSTATUS_WORK_FL,
                      client.SOCSTATUS_PENS_FL,
                      client.GENDER,
                      client.CHILD_TOTAL,
                      client.DEPENDANTS,
                      client.PERSONAL_INCOME,
                      client.LOAN_NUM_TOTAL,
                      client.LOAN_NUM_CLOSED]])

    prediction = model.predict(data)

    return prediction
