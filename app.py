import streamlit as st
from PIL import Image
import pandas as pd
import eda

st.title('Информацию о клиентах банка $')

img = Image.open('photos/bank.jpg')
st.image(img)

df = pd.read_csv('data/data.csv')

with st.expander('Число пропусков, дубликатов в данных'):
    eda.get_na_dubls(df)

with st.expander('Графики распределений числовых признаков'):
    eda.numerical_distribution(df)

with st.expander('Анализ распределения выбросов'):
    eda.boxplots(df)

with st.expander('Матрица корреляций'):
    eda.correlation(df)

with st.expander('Графики зависимостей целевой переменной и признаков'):
    eda.relation_target(df)

with st.expander('Числовые характеристики распределения числовых столбцов'):
    eda.describe_data(df[['AGE', 'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME',
                          'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']])

with st.expander('Числовые характеристики категориальных столбцов'):
    categorial = df[['TARGET', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
                     'GENDER']]

    binary_stats = pd.DataFrame()

    binary_stats['Количество 1'] = categorial.sum()
    binary_stats['Количество 0'] = categorial.shape[0] - categorial.sum()
    binary_stats['Процент 1'] = categorial.mean() * 100
    binary_stats['Процент 0'] = (1 - categorial.mean()) * 100
    binary_stats['Среднее'] = categorial.mean()

    st.dataframe(binary_stats)




