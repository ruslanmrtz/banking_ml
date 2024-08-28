import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

warnings.simplefilter("ignore")

# Загружаем модель
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)


def get_prediction(data: np.array) -> tuple[int, tuple]:
    proba = model.predict_proba(data)
    pred = int(proba[:, 1] > 0.12)

    return pred, proba


def get_threshold(df: pd.DataFrame, threshold: float) -> tuple[np.array, np.array]:
    df = df.dropna(subset='TARGET')

    X = df.drop(['TARGET', 'AGREEMENT_RK'], axis=1)
    y = df['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    classes = proba > threshold

    return y_test, classes


def write_metrics(y_test: np.array, classes: np.array) -> None:
    st.code(f'Accuracy: {accuracy_score(y_test, classes):.2f}')
    st.code(f'Precision: {precision_score(y_test, classes, zero_division=0):.2f}')
    st.code(f'Recall: {recall_score(y_test, classes):.2f}')
    st.code(f'F1: {f1_score(y_test, classes):.2f}')


