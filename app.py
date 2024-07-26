import pandas as pd
import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
texts = data['v2'].tolist()
labels = data['v1'].apply(lambda x: 1 if x == 'spam' else 0).tolist()

# Convert text data to num
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

#  model use
model = MultinomialNB()
model.fit(X, y)

# UI
st.title("Spam Detection App")
input_text = st.text_area("Enter your message here:")
predict_button = st.button("Predict")

if predict_button:
    if input_text:
        vector_input = vectorizer.transform([input_text])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.write("Prediction: SPAM")
        else:
            st.write("Prediction: NOT SPAM")
    else:
        st.write("Please enter a message to classify.")