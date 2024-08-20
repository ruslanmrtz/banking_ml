import streamlit as st
from PIL import Image
import pandas as pd
import eda
import model
import numpy as np

st.title('КлиентАналитик $')
st.header('Оценка и прогнозирование склонности клиентов к отклику на предложения банка')


# img = Image.open('photos/bank_1.jpg')
# st.image(img)

df = pd.read_csv('data/data.csv')

inf, pred, vizual = st.tabs(["Информация о клиентах банка", "Прогноз по данным клиента", "Визуализация результатов модели"])


with inf:
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


with pred:
    st.subheader('Настройте данные о клиенте')

    age = st.slider('Возраст', 20, 70)

    col1, col2 = st.columns(2)
    with col1:
        children = st.number_input("Количество детей", min_value=0, max_value=10, step=1)
    with col2:
        inv = st.number_input("Количество иждивенцев", min_value=0, max_value=10, step=1)

    gender = st.radio('Пол',
                      ['Мужской', 'Женский'])

    col1, col2 = st.columns(2)
    with col1:
        loans = st.number_input("Количество кредитов", min_value=1, max_value=11, step=1)
    with col2:
        closed_loans = st.number_input("Количество закрытых кредитов", min_value=0, max_value=11, step=1)

    col1, col2 = st.columns(2)
    with col1:
        job = st.radio('Социальный статус относительно работы',
                      ['Работает', 'Не работает'])
    with col2:
        pens = st.radio('Социальный статус относительно пенсии',
                        ['Не пенсионер', 'Пенсионер'])

    income = st.slider('Доход', 0, 250000, value=25000)

    binary_dict = {
        'Мужской': 1,
        'Женский': 0,
        'Работает': 1,
        'Не работает': 0,
        'Пенсионер': 1,
        'Не пенсионер': 0
    }

    data = np.array([[age, binary_dict[job], binary_dict[pens],
                     binary_dict[gender], children, inv, income,
                     loans, closed_loans]])

    pred = model.get_prediction(data)
    if pred[0]:
        st.success('Клиент откликнется на маркетинговую компанию!')
    else:
        st.error('Клиент не откликнется на маркетинговую компанию!')


with vizual:
    threshold = st.slider('Выберите порог', min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    y_test, classes = model.get_threshold(df, threshold)

    model.write_metrics(y_test, classes)





