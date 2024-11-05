
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Load data
penguins = sns.load_dataset("penguins").dropna()

# Sidebar for user inputs
st.sidebar.header("Penguin Features")
bill_length = st.sidebar.slider("Bill Length (mm)", float(penguins.bill_length_mm.min()), float(penguins.bill_length_mm.max()))
bill_depth = st.sidebar.slider("Bill Depth (mm)", float(penguins.bill_depth_mm.min()), float(penguins.bill_depth_mm.max()))
flipper_length = st.sidebar.slider("Flipper Length (mm)", float(penguins.flipper_length_mm.min()), float(penguins.flipper_length_mm.max()))
body_mass = st.sidebar.slider("Body Mass (g)", float(penguins.body_mass_g.min()), float(penguins.body_mass_g.max()))

# Train model
X = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = penguins['species']
clf = RandomForestClassifier()
clf.fit(X, y)

# Predict
input_features = [[bill_length, bill_depth, flipper_length, body_mass]]
prediction = clf.predict(input_features)

# Display prediction
st.write("Predicted Species:", prediction[0])
