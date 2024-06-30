import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pickle.load(open('df.pkl','rb'))
lr = pickle.load(open('lr.pkl','rb'))

# Define a dictionary to map numerical labels to type names
type_map = {'CASH_OUT': 1, 'PAYMENT': 2, 'CASH_IN': 3, 'TRANSFER': 4, 'DEBIT': 5}

# Invert the dictionary to map type names back to numerical labels (optional)
# reverse_type_map = {v: k for k, v in type_map.items()}  # Uncomment if needed

df['type_name'] = df['type'].map(type_map)  # Use mapping to create 'type_name' column

X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
y = df['isFraud']

z = sorted(df['type'].unique())

st.title('Transaction Fraud Detection')

# Use the 'type_name' column for the selectbox
selected_type = st.selectbox(label='Type:',options=z)

# Use the selected type name to get the corresponding numerical label (optional)
# selected_type_label = reverse_type_map.get(selected_type)  # Uncomment if needed

amount = st.number_input(label='Amount:')

oldbalanceOrg = st.number_input(label='Amount of Old Balance:')

newbalanceOrig = st.number_input(label='Amount of New Balance:')

btn = st.button('Predict')

if btn:
    # Use the selected type name or numerical label (depending on your preference)
    query = pd.DataFrame(data=np.array([[selected_type, float(amount), float(oldbalanceOrg), float(newbalanceOrig)]]), columns=X.columns)
    st.title('Prediction is:'+str(lr.predict(query)))
