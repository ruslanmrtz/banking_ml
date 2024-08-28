import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns

def get_na_dubls(df: pd.DataFrame) -> None:
    na = df.isnull().sum().sum()
    st.markdown(f'Общее число пропущенных значений: **{na}**')

    dup = df.duplicated().sum()
    st.markdown(f'Общее число дубликатов: **{dup}**')


def numerical_distribution(df: pd.DataFrame) -> None:
    sns.set(style="whitegrid", palette="flare")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(x=df['AGE'], ax=ax1)
    sns.histplot(x=df['CHILD_TOTAL'], ax=ax2, bins=10)
    ax1.set_title('Возраст')
    ax2.set_title('Количество детей')

    fig2, ax3 = plt.subplots(1, 1, figsize=(15, 5))
    sns.histplot(x=df['PERSONAL_INCOME'], ax=ax3, bins=30)
    ax3.set_title('Доход')

    st.pyplot(fig)
    st.pyplot(fig2)


def correlation(df: pd.DataFrame) -> None:

    corr_df = df.corr()
    fig, ax = plt.subplots()
    ax.set_title('Матрица корреляций')
    sns.heatmap(corr_df, ax=ax)

    st.pyplot(fig)


def boxplots(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    sns.boxplot(x=df['AGE'], y=df['GENDER'].map({1: 'Мужчина', 0: 'Женщина'}), hue=df['TARGET'])

    st.pyplot(fig)


def relation_target(df: pd.DataFrame) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    sns.barplot(data=df, x='TARGET', y='AGE', ax=ax1)
    ax1.set_title('Возраст')
    sns.barplot(data=df, x='TARGET', y='PERSONAL_INCOME', ax=ax2)
    ax2.set_title('Доход')
    sns.barplot(data=df, x='TARGET', y='LOAN_NUM_CLOSED', ax=ax3)
    ax3.set_title('Количество закрытых кредитов')

    st.pyplot(fig)


def describe_data(df: pd.DataFrame) -> None:

    desc = df.describe()

    st.dataframe(desc)

