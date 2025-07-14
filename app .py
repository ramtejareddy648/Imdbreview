import streamlit as st
from tensorflow.keras.models import load_model
from keras.utils import pad_sequences
import tensorflow as tf
from tensorflow.keras.datasets import imdb

model=load_model('/content/drive/MyDrive/projectreview/modelimdb.h5')

word_index=imdb.get_word_index()


def preprocess_text(text):
  words=text.lower().split()
  encoded_review=[word_index.get(word, 2) + 3  for word in words]
  padd_review=pad_sequences([encoded_review],maxlen=500)
  return padd_review


def predictions(review):
  padd_num=preprocess_text(review)
  prediction=model.predict(padd_num)

  sentiment='positive' if prediction[0][0]> 0.5  else 'negative'

  return sentiment,prediction[0][0]

st.title('IMDB movie Rating')
st.write('write your review here to know positive or negative')

user_input=st.text_area('enter Review')

if st.button('Classify'):
  sentiment,pred_score=predictions(user_input)
  st.write(f'review is :{user_input}')
  st.write(f'sentiment is :{sentiment}')
  st.write(f'pred_score is :{pred_score}')
else:
  st.write('enter your review')
