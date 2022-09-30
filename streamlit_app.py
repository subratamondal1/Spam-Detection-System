import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer()
porterStemmer = PorterStemmer()

# Text Preprocessing Helper Function


def text_preprocessing(text):
    text = text.lower()  # return lowercase str | Lower Case
    # returns list of tokenize words | Tokenization
    text = nltk.word_tokenize(text)
    store = []

    for i in text:
        if i.isalnum():  # Removing Special Characters
            # Storing only alphanumeric characters | Removing Special Characters
            store.append(i)

    text = store[:]
    store.clear()

    for i in text:
        # Removing stop words and punctuation
        if i not in stopwords.words("english") and i not in string.punctuation:
            store.append(i)

    text = store[:]
    store.clear()

    for i in text:
        store.append(porterStemmer.stem(i))  # Stemming

    text = store

    return " ".join(text)


# Loading tfidfVectorizer
tfidfVectorizer = pickle.load(open("tfidfVectorizer.pkl", "rb"))

# Loading Model
multinomialNB = pickle.load(open("multinomialNB.pkl", "rb"))

st.title("Spam Detection System")

input_text = st.text_area("Write or copy paste any text")

if st.button("Predict"):

    # Spam Detection will be done via these 4 steps
    # Step 1. Preprocess
    transformed_text = text_preprocessing(input_text)

    # 2. Vectorize
    vector_input = tfidfVectorizer.transform([transformed_text])

    # 3. Predict
    result = multinomialNB.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
