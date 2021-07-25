import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title ("Penguins Classifier")

st.write("""
# Penguin Prediction App (Multi-Layered Perceptron / MLP)

### Created By : [DeanTevin](https://deantevin.github.io/)

This app predicts the **Palmer Penguin** species!

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        temp = st.sidebar.slider('temp',32.1,59.6,43.9)
        rh = st.sidebar.slider('rh',32.1,59.6,43.9)
        pressure = st.sidebar.slider('pressure', 32.1,59.6,43.9)
        ws = st.sidebar.slider('ws', 13.1,21.5,17.2)
        wd = st.sidebar.slider('wd', 172.0,231.0,201.0)
        ch = st.sidebar.slider('ch', 2700.0,6300.0,4207.0)
        solrad = st.sidebar.slider('solrad', 2700.0,6300.0,4207.0)
        data = {'temp': temp,
                'rh': rh,
                'pressure': pressure,
                'ws': ws,
                'wd': wd,
                'ch': ch,
                'solrad': solrad}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('testingdata.csv')
penguins = penguins_raw.drop(columns=['status','unit','cycle'])
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sex','island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('data.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['suspect','good'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)