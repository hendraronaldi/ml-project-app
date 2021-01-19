import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from os import path

import tensorflow as tf

class CarPurchasePrediction:
	def __init__(self, repo):
		self.repo = repo
		self.load_dataset()
		self.load_model()
		self.load_scaler()
		self.load_encoder()

	def load_dataset(self):
		self.dataset = pd.read_csv(f'https://raw.githubusercontent.com/hendraronaldi/{self.repo}/master/Car_Purchasing_Data.csv', encoding='ISO-8859-1')
		self.countries = sorted(self.dataset.Country.unique())

	def load_model(self):
		if not path.exists('apps/cpp_model.h5'):
			m = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/master/model.h5?raw=true')
			open('apps/cpp_model.h5', 'wb').write(m.content)
		self.model =  tf.keras.models.load_model('apps/cpp_model.h5')

	def load_scaler(self):
		if not path.exists('apps/cpp_scaler_x.pkl'):
			scx = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/master/scaler_x.pkl?raw=true')
			open('apps/cpp_scaler_x.pkl', 'wb').write(scx.content)

		if not path.exists('apps/cpp_scaler_y.pkl'):
			scy = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/master/scaler_y.pkl?raw=true')
			open('apps/cpp_scaler_y.pkl', 'wb').write(scy.content)

		self.scaler_x = joblib.load('apps/cpp_scaler_x.pkl')
		self.scaler_y = joblib.load('apps/cpp_scaler_y.pkl')

	def load_encoder(self):
		if not path.exists('apps/cpp_encoder.pkl'):
			enc = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/master/encoder.pkl?raw=true')
			open('apps/cpp_encoder.pkl', 'wb').write(enc.content)
		self.encoder = joblib.load('apps/cpp_encoder.pkl')

	def input_features(self):
		st.subheader('Input Parameters')
		country = st.selectbox('Country', self.countries)
		gender = st.selectbox('Gender', (0, 1))
		age = st.number_input('Age')
		annual_salary = st.number_input('Annual Salary')
		credit_card = st.number_input('Credit Card Debt')
		net_worth = st.number_input('Net Worth')

		return pd.DataFrame({
			'Country': [country],
			'Gender': [gender],
			'Age': [age],
			'Annual Salary': [annual_salary],
			'Credit Card Debt': [credit_card],
			'Net Worth': [net_worth]
			})

	def predict(self):
		pred_df = self.input_features()
		pred_df['Country'] = self.encoder.transform(pred_df['Country'])
		X = self.scaler_x.transform(pred_df)

		self.show(self.scaler_y.inverse_transform(self.model.predict(X))[0][0])

	def show(self, predictions):
		st.subheader('Car Purchase Amount Prediction')
		st.write(predictions)
