import streamlit as st
import pickle
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
ps = PorterStemmer()


###################################
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model (1).pkl', 'rb'))
################################################

st.title("Spam Detection Service")
input_msg = st.text_area("Copy and Paste the Message")
# 1. Preprocess input message
# 2. Vectorize
# 3. Predict
# 4. Display
if st.button('Predict'):
    transformed_msg = transform_text(input_msg)
    vec_input = tfidf.transform([transformed_msg])
    result = model.predict(vec_input)[0]
    if result == 0:
        st.header("Not a Spam Message!!!")
    else:
        st.header("Spam Message !!!")
