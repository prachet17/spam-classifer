# Save this as spam_classifier.py and run using: python spam_classifier.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load the data
raw_mail_data = pd.read_csv(r"C:\Users\sahoo\OneDrive\Desktop\New folder\spam.csv")

# Replace null values with empty strings
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Label encoding: spam as 0 and ham as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate the data into texts (X) and labels (Y)
X = mail_data['Message']
Y = mail_data['Category']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Transform the text data to feature vectors that can be used as input to the Logistic Regression model
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert Y_train and Y_test values to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train the Logistic Regression model with the training data
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Function to predict spam or ham
def predict_mail(email_content):
    input_data_features = feature_extraction.transform([email_content])
    prediction = model.predict(input_data_features)
    if prediction[0] == 1:
        result = 'Ham mail'
    else:
        result = 'Spam mail'
    return result

# Create the GUI
def create_gui():
    def on_predict():
        email_content = email_text.get("1.0", tk.END).strip()
        if email_content:
            result = predict_mail(email_content)
            messagebox.showinfo("Prediction Result", f'Prediction: {result}')
        else:
            messagebox.showwarning("Input Error", "Please enter email content.")

    root = tk.Tk()
    root.title("Spam/Ham Email Classifier")

    tk.Label(root, text="Enter email content:").pack(pady=10)
    email_text = tk.Text(root, height=10, width=50)
    email_text.pack(pady=10)

    predict_button = tk.Button(root, text="Predict", command=on_predict)
    predict_button.pack(pady=10)

    root.mainloop()

# Run the GUI
create_gui()