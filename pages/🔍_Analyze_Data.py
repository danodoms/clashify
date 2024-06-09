import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from library.feature_selection import cramers_v_analysis, chi2_feature_significance
from library.model import tabulize_model_results, balance_data
from library.bin import label_encode_all
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from library.describe import *

st.title("Data Analysis Tool")
st.write("Upload your dataset and perform data analysis in seconds!")


uploaded_file = st.file_uploader("Choose a CSV file")
# bytes_data = uploaded_file.read()
# st.write("filename:", uploaded_file.name)
# st.write(bytes_data)

st.divider()

st.header("Description")
st.dataframe(pd.read_csv(uploaded_file))

# create a function that counts missing vlaues in a dataframe
def missing_values_table(df):
    total_null = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])
    return missing_values


st.dataframe(missing_values_table(pd.read_csv(uploaded_file)))