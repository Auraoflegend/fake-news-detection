import streamlit as st
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



news =pd.read_csv(r'C:\Users\adity\OneDrive\Desktop\fake news detection\news_dataset.csv')



        

def wordopt(content):
    st= re.sub('[^a-zA-Z]',' ',str(content))
    st = st.lower()
    return st

news['content']=news['content'].apply(wordopt)



x= news['content']
y= news ['label']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)




vector = TfidfVectorizer()
xv_train = vector.fit_transform(x_train)
xv_test = vector.transform(x_test)

model = LogisticRegression()
model.fit(xv_train,y_train)
#website
st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)

    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    
    else:
        st.write('The News Is Real')
    








