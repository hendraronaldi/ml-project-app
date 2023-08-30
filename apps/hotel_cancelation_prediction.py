import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from os import path
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler


class HotelCancelation:
    def __init__(self, repo):
        self.repo = repo
        self.load_model()
        self.load_scaler()

    def load_model(self):
        if not path.exists('apps/lgbm_imp15.pkl'):
            m = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/master/lgbm_imp15.pkl?raw=true')
            open('apps/lgbm_imp15.pkl', 'wb').write(m.content)
        self.model = joblib.load('apps/lgbm_imp15.pkl')

    def load_scaler(self):
        if not path.exists('apps/scaler_imp15.pkl'):
            scl = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/master/scaler_imp15.pkl?raw=true')
            open('apps/scaler_imp15.pkl', 'wb').write(scl.content)
        self.scaler = joblib.load('apps/scaler_imp15.pkl')

    def info(self):
        st.write('Classify whether the hotel booking will be canceled or not')
        st.write('This is an application from my final project group Shift Academy Batch 9 Data Science Bootcamp')
        st.write("""The presentation [here](https://github.com/hendraronaldi/machine_learning_projects/blob/main/Shift%20Academy%20DS%20Bootcamp%20Batch%209/Shift%20Academy%20DS%20Bootcamp%20Batch%209%20Final%20Project%20Kelompok%203.pptx)""")
        st.write("""Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)""")
        st.write("""Notebook can be found [here](https://github.com/hendraronaldi/machine_learning_projects/blob/main/Shift%20Academy%20DS%20Bootcamp%20Batch%209/Shift%20Academy%20DS%20Bootcamp%20Batch%209%20Final%20Project%20Kelompok%203.ipynb)""")

    def input_features(self):
        country = st.selectbox('Country', ('PRT', 'Other'))
        agent = st.selectbox('Agent', ('9', '240', 'Other'))
        assigned_room_type = st.selectbox('Assigned Room Type', ('I', 'Other'))

        return pd.DataFrame({
            'adr': [st.number_input('Average Daily Rate', min_value=0.00, max_value=5500.00, step=0.01)],
            'lead_time': [st.number_input('Lead Time', min_value=0, max_value=750, step=1)],
            'stays_in_week_nights': [st.number_input('stays_in_week_nights', min_value=0, max_value=50, step=1)],
            'total_of_special_requests': [
                st.number_input('total_of_special_requests', min_value=0, max_value=5, step=1)],
            'stays_in_weekend_nights': [st.number_input('stays_in_weekend_nights', min_value=0, max_value=20, step=1)],
            'booking_changes': [st.number_input('booking_changes', min_value=0, max_value=25, step=1)],
            'adults': [st.number_input('Number of Adults', min_value=0, max_value=60, step=1)],
            'previous_bookings_not_canceled': [
                st.number_input('previous_bookings_not_canceled', min_value=0, max_value=75, step=1)],
            'previous_cancellations': [st.number_input('previous_cancellations', min_value=0, max_value=30, step=1)],
            'required_car_parking_spaces': [
                st.number_input('required_car_parking_spaces', min_value=0, max_value=10, step=1)],
            'children': [st.number_input('Number of children', min_value=0, max_value=15, step=1)],
            'country_PRT': [1] if country == 'PRT' else [0],
            'agent_9': [1] if agent == '9' else [0],
            'agent_240': [1] if agent == '14' else [0],
            'assigned_room_type_I': [1] if assigned_room_type == 'I' else [0]
        })

    def predict(self):
        self.info()
        pred_df = self.scaler.transform((self.input_features()))
        self.show(self.model.predict(pred_df))

    def show(self, predictions):
        st.subheader('Prediction')
        result_text = st.empty()
        if predictions == 1:
            result_text.error("Cancel")
        else:
            result_text.success("Not Cancel, OK")
