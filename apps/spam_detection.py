import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from os import path

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SpamDetection:
	def __init__(self, repo):
		self.repo = repo
		self.load_model()
		self.load_tokenizer()

	def load_model(self):
		if not path.exists('apps/sd_model.h5'):
			m = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/master/model.h5?raw=true')
			open('apps/sd_model.h5', 'wb').write(m.content)
		self.model =  tf.keras.models.load_model('apps/sd_model.h5')

	def load_tokenizer(self):
		if not path.exists('apps/sd_tokenizer.pickle'):
			tok = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/master/tokenizer.pickle?raw=true')
			open('apps/sd_tokenizer.pickle', 'wb').write(tok.content)

		with open('apps/sd_tokenizer.pickle', 'rb') as handle:
			self.tokenizer = pickle.load(handle)

	def input_features(self):
		st.subheader('Input Parameters')
		threshold = 0.5
		text = st.text_area('Text to analyze', "", height=5)
		return text, threshold

	def predict(self):
		text, threshold = self.input_features()
		text_tok = self.tokenizer.texts_to_sequences([text])
		text_pad = pad_sequences(text_tok, maxlen=189) #189 is maxlen in training
		predictions = self.model.predict(text_pad)[0][0]
		self.show(predictions, threshold)

	def show(self, predictions, threshold):
		st.header('Prediction')
		if predictions < threshold:
			st.write('Not a spam')
		else:
			st.write('Spam detected!!')
		st.header('Score')
		st.write(predictions)
