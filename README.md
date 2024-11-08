# Credit-card-fraud-detection-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
@st.cache(allow_output_mutation=True)
def load_data(file):
return pd.read_csv(file)
def train_model(data):
# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)
# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, stratify=y, random_state=2
)
# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)
return model, train_acc, test_acc
def parse_transaction_string(transaction_string, feature_names):
values = transaction_string.split(",")
transaction = {}
for i in range(len(values)):
transaction[feature_names[i]] = float(values[i])
return transaction
st.title("Credit Card Fraud Detection")
file = st.file_uploader(
"Upload a CSV file containing credit card transaction data:")
if file is not None:
data = load_data(file)
st.write("Data shape:", data.shape)
st.dataframe(data)
model, train_acc, test_acc = train_model(data)
st.write("Training accuracy:", train_acc)
st.write("Test accuracy:", test_acc)
# allow user to input transaction features and get a prediction
st.subheader("Check a transaction")
feature_names = data.drop(columns="Class", axis=1).columns
data= pd.read_csv("creditcard.csv")
df = pd.DataFrame(data)
# Filter the DataFrame to only include rows where the "Class" column is 1
filtered_df = df[df['Class'] == 1]
# Display the filtered DataFrame in Streamlit
st.write("Rows where Transaction are Fraud")
st.write(filtered_df)
