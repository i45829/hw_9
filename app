import streamlit as st
import pandas as pd
import pickle

# Загрузка модели
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Прогноз стоимости недвижимости')

total_square = st.number_input('Площадь')
rooms = st.number_input('Количество комнат')
floor = st.number_input('Этаж')

if st.button('Сделать прогноз'):

    dt = pd.DataFrame({
        
        'total_square': [total_square],
        'rooms': [rooms],
        'floor': [floor]
    })

    sc = scaler.transform(dt)

    prediction = model.predict(sc)

    st.success(f'Прогнозируемая стоимость: {prediction[0]:.2f}')
