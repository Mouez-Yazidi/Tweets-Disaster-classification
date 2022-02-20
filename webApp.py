#import packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
plt.style.use('seaborn')

def main():
    Menu=['Overview','home','About']
    choice=st.sidebar.selectbox('Menu',Menu)
    if choice=='home':
        st.title('Natural Language Processing with Disaster Tweets')
        st.image('https://digitalfireflymarketing.com/wp-content/uploads/2013/10/pablo-17-2.png')
        st.markdown("""
        <style>
        .big-font {
            font-size:48px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="big-font">Enter Tweet here :</p>', unsafe_allow_html=True)
        text = st.text_area(label='',height=200)
        if st.button('Predic class'):
            model = load_model('model.h5')
            tokenizer = Tokenizer(split=' ')
            tokenizer.fit_on_texts(text)
            X = tokenizer.texts_to_sequences(text)
            pred = model.predict(X)
            st.success(f"The news item is {pred}")

if __name__=='__main__':
    main()
