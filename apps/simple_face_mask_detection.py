import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import joblib
from os import path

class FaceMaskDetection:
	def __init__(self, repo):
		self.repo = repo
		self.load_model()
		self.load_scaler()
		self.load_encoder()

	def load_model(self):
		if not path.exists('apps/fmd_model.h5'):
			m = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/main/model.h5?raw=true')
			open('apps/fmd_model.h5', 'wb').write(m.content)
		self.model =  tf.keras.models.load_model('apps/fmd_model.h5')

	def load_scaler(self):
		if not path.exists('apps/fmd_scaler.pkl'):
			sc = requests.get(f'https://github.com/hendraronaldi/{self.repo}/blob/main/scaler.pkl?raw=true')
			open('apps/fmd_scaler.pkl', 'wb').write(sc.content)
		self.scaler = joblib.load('apps/fmd_scaler.pkl')

	def load_encoder(self):
		ptm = PretrainedModel(
		    include_top=False,
		    weights='imagenet',
		    input_shape=[200, 200] + [3]
		)
		dx = Flatten()(ptm.output)
		dm = Model(inputs=ptm.input, outputs=dx)
		self.encoder = dm

	def input_features(self):
		st.subheader('Input Parameters')
		frame = None
		threshold = st.slider('Threshold', 0.0, 1.0, 0.5)
		uploaded_img = st.file_uploader("Upload Image")
		if uploaded_img is not None:
			try:
				file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
				frame = cv2.imdecode(file_bytes, 1)

				if frame.shape[0] > 500:
					st.image(frame, channels="BGR", width=500)
				else:
					st.image(frame, channels="BGR")
			except:
				st.subheader("Please upload image")

		return frame, threshold

	def predict(self):
		frame, threshold = self.input_features()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (200, 200))
		img = image.img_to_array(frame)
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
		img = preprocess_input(img)

		feat_test = self.encoder.predict(img)
		feat_test = self.scaler.transform(feat_test)
		predictions = 1-self.model.predict(feat_test)[0][0]
		self.show(predictions, threshold)

	def show(self, predictions, threshold):
		st.header('Face Mask Prediction')
		if predictions < threshold:
			st.subheader('Not using face mask!!!')
		else:
			st.subheader('Using face mask, good')
		st.write("***")
		st.header('Prediction Score')
		st.subheader(predictions)