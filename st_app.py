import streamlit as st
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title('Модель бинарной классификации Spaceship Titanic')

with st.sidebar:
    st.subheader('Параметры пользователя')
    st.divider()
    flag = st.checkbox('Отобразить непредобработанные данные')
if flag:
    st.divider()
    st.warning('Непредобработанные данные:')
    data = pd.read_csv('data/train.csv')
    st.dataframe(data)

st.divider()
st.success('Предобработанные данные:')
df = pd.read_csv('data/train_full.csv')
df_pred = pd.read_csv('data/test_full.csv')

df_pred_PassengerId = pd.read_csv('data/test.csv')
Id = df_pred_PassengerId['PassengerId']
st.dataframe(df)

with st.sidebar:
    st.divider()
    flag_1 = st.checkbox('Отобразить данные для предсказания')
if flag_1:
    st.divider()
    st.success('Данные для предсказания:')
    df_pred = pd.read_csv('data/test_full.csv')
    st.dataframe(df_pred)

category_col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP',
                'Number', 'Deck', 'Side', 'HasServices'
                ]

for i in category_col:
    df[i] = df[i].astype('category')
    df_pred[i] = df_pred[i].astype('category')

X = df.drop(columns='Transported')
y = df['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.divider()
st.subheader('Обучение модели')
flag_2 = st.checkbox('Ввести параметры вручную:')
if flag_2:
    max_depth = st.slider('max_depth', 1, 17, 3)
    learning_rate = st.slider('learning_rate', 0.01, 0.3, 0.1)
    n_estimators = st.slider('n_estimators', 100, 1000, 100, 10)
    params = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators
    }
else:
    st.write('Лучшие параметры поиска optuna ')
    params = {
        'learning_rate': 0.030018077829513434,
        'n_estimators': 802,
        'lambda_l1': 3.0098282141559296,
        'lambda_l2': 4.011035680904194e-08,
        'max_depth': 11,
        'colsample_bytree': 0.5717499737816155,
        'subsample': 0.8885252066982045,
        'min_child_samples': 21,
        'feature_fraction': 0.6090781828286976,
        'bagging_fraction': 0.6518437070058158
    }
    st.json(params)

flag_3 = st.button("Обучить модель", type="primary")
if flag_3:
    model = LGBMClassifier(**params, verbosity=-1, boosting_type="gbdt", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pred = model.predict(df_pred)
    output = pd.DataFrame({'PassengerId': Id, 'Transported': pred})
    output['Transported'] = output['Transported'].astype(bool)
    score = accuracy_score(y_test, y_pred)
    if score >= 0.8:
        st.balloons()
        st.write(f'accuracy_score на тестовых данных: {score}')
        st.subheader('Наши предсказания')
        st.dataframe(output)
    else:
        st.write(f'accuracy_score на тестовых данных: {score}')
        st.subheader('Наши предсказания')
        st.dataframe(output)
