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
import nltk 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords,wordnet
import re
plt.style.use('seaborn')

def remove_Stopwords(text):
    stopW=stopwords.words('english') #get the english stopwords
    return " ".join([i for i in text.split() if i not in stopW])

def get_wordnet_pos(word):
   # """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(text):
    # 1. Init Lemmatizer
    lemmatizer = WordNetLemmatizer()
    # 2. Lemmatize text with the appropriate POS tag
    return " ".join([lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in text.split()])

def clean_text(text):
    # convert catacter to lowercase
    text=text.lower()
    #remove URLS
    text =re.sub(r"http\S+", "", text)
    #remove ponctuation
    text=re.sub(r"[^\w\s]", "", text)
    #remove 
    text = re.sub(r'/n',"",text)
    #remove degits
    text = re.sub(r'\d+',"",text)
    #remove multiple spaces
    text = re.sub(r'\s+'," ",text)
    
    #text = re.sub(r'\s+[a-zA-Z]\s+'," ",text)
    #remove stop Words
    text = remove_Stopwords(text)
    # text normalization
    text = lemmatize(text)
    #remove single caracter
    text = re.sub(r'\s+[a-zA-Z]\s+'," ",text)
    return text
def text_to_seq(text):
    text = clean_text(text)
    l=[]
    l.append(text)
    tokenizer = Tokenizer(split=' ')
    tokenizer.fit_on_texts(l)
    X = tokenizer.texts_to_sequences(l)
    X=pad_sequences(X,25,padding='post')
    return X,l
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
            X,text1 = text_to_seq(text)
            pred = model.predict(X)
            pred = np.where(pred >0.5,1,0)
            if pred ==1:
                st.image('https://c.tenor.com/J4AkSzHfCMUAAAAC/emergency-animated.gif')
                st.success("This tweet is Disaster tweet")
            else : 
                st.success("This tweet is Normal tweet")

if __name__=='__main__':
    main()
