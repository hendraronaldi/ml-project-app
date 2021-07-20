import cv2
import av
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
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

def face_mask_detection():
	WEBRTC_CLIENT_SETTINGS = ClientSettings(
		rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
		media_stream_constraints={
			"video": True,
		},
	)

	class FaceMaskDetection(VideoProcessorBase):
		predictions: float
		def __init__(self):
			self.predictions = None
			self.load_model()
			self.load_scaler()
			self.load_encoder()

		def load_model(self):
			if not path.exists('apps/fmd_model.h5'):
				m = requests.get(f'https://github.com/hendraronaldi/simple-face-mask-detection/blob/main/model.h5?raw=true')
				open('apps/fmd_model.h5', 'wb').write(m.content)
			self.model =  tf.keras.models.load_model('apps/fmd_model.h5')

		def load_scaler(self):
			if not path.exists('apps/fmd_scaler.pkl'):
				sc = requests.get(f'https://github.com/hendraronaldi/simple-face-mask-detection/blob/main/scaler.pkl?raw=true')
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

		def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
			try:
				frame = frame.to_ndarray(format="bgr24")
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frame = cv2.resize(frame, (200, 200))
				img = image.img_to_array(frame)
				img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
				img = preprocess_input(img)

				feat_test = self.encoder.predict(img)
				feat_test = self.scaler.transform(feat_test)
				self.predictions = 1-self.model.predict(feat_test)[0][0]
			except Exception as e:
				print(e)

	webrtc_ctx = webrtc_streamer(
        key="face-mask-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=FaceMaskDetection,
        async_processing=True,
    )

	if webrtc_ctx.state.playing:
		st.header('Face Mask Prediction')
		mask_text = st.empty()
		st.header('Prediction Score')
		pred_text = st.empty()
		while True:
			if webrtc_ctx.video_processor:
				pred = webrtc_ctx.video_processor.predictions
				pred_text.subheader(pred)
				if pred != None:
					if pred < 0.5:
						mask_text.warning('Not using face mask!!!')
					else:
						mask_text.success('Using face mask, Good Job')