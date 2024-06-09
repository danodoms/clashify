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



st.set_page_config(
    page_title="Online Food Order Dataset 🍔",
    layout="centered",
    page_icon="🍔" )

# %% Live Reload
from liveReload import live_reload
# live_reload()

st.sidebar.title("Online Food Order Dataset")
# st.sidebar.markdown("Use the options below to filter the data or change the visualization settings.")
st.sidebar.divider()
st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("app.py", label="Map", icon="🗺️")
# st.sidebar.page_link("pages/🔍_Analyze_Data.py", label="Map", icon="🔍")




st.title("Online Food Order Dataset")
st.write("The dataset contains information collected from an online food ordering platform over a period of time. It encompasses various attributes related to Occupation, Family Size, Feedback etc..")
# create a link
st.markdown("**Source:** [Kaggle](https://www.kaggle.com/datasets/sudarshan24byte/online-food-dataset/data)")

st.divider()


# col1, col2 = st.columns([2, 1])

# Load dataset
# @st.cache_data
# def load_data():
#     data = pd.read_csv('onlinefoods.csv')
#     return data


# data = load_data()
data = pd.read_csv('onlinefoods.csv')
st.write("## Dataset")
st.write(data)



# st.write("## Dataset Summary")
# st.write(data.describe())






# Display the map
st.write("## Map")
st.write("Data was collected from the following locations")
st.map(data[['latitude', 'longitude']])












# Outlier Detection
# st.write("## Outlier Detection")

# column_name = st.selectbox('Select column for outlier detection', data.columns)
# selected_data = data[column_name]

# def detect_outliers_iqr(data):
#     Q1 = data.quantile(0.05)
#     Q3 = data.quantile(0.95)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = data[(data < lower_bound) | (data > upper_bound)]
#     return outliers

# outliers_iqr = detect_outliers_iqr(selected_data)

# st.write("Outliers detected using IQR method:")
# st.write(outliers_iqr)

# plt.figure(figsize=(10, 5))
# sns.boxplot(selected_data)
# plt.title(f'Box plot of {column_name}')
# st.pyplot(plt)

# Binning and Label Encoding
# st.write("## Binning and Label Encoding")

binned_age = pd.cut(data["Age"],
            bins=[17, 22, 26, 29, 34],
            labels=['1', '2', '3', '4'])

binned_pinCode = pd.cut(data["Pin code"],
            bins=[560000, 560025, 560050, 560075, 560100, 560125],
            labels=['1', '2', '3', '4', '5'])

data = label_encode_all(data, exclude_columns=['Age', 'Pin code'])

data['Age'] = binned_age
data['Pin code'] = binned_pinCode

# st.write("New Data after Binning and Label Encoding")
# st.write(data.head())

# # Feature Selection
# st.write("## Feature Selection")
# st.write("### X vs X Feature Selection Using Cramer's V")

fig, ax = plt.subplots(figsize=(10, 8))
cramers_v_analysis(data, target_column='Feedback', threshold=0.6, top_n=10, plot=True)
# st.pyplot(fig)

# st.write("### X vs Y Feature Selection using Chi Square Test")
chi2_feature_significance(data, "Feedback", significance_threshold=0.05)

# Model Comparison
# st.write("## Accuracy Comparison")
# st.write("### Initial Model Creation with SMOTE for Comparison")

x = data.drop(columns=['Feedback'], axis=1)
y = data['Feedback']

x_smote, y_smote = balance_data(x, y)
x_smote_featureSelection = x_smote.drop(['Family size', 'Gender', 'Pin code'], axis=1)

# Interactive Feature Selection and Model Evaluation
st.write("## Interactive Model Testing")

# Interactive feature selection
st.write("### Decision Tree vs Random Forest vs SVM")
selected_features = st.multiselect(
    'Select features to include in the model',
    options=x.columns,
    default=x.columns.tolist()
)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machines': SVC(random_state=42)
}

if st.button('Evaluate'):
    # Split data
    x_selected = x_smote[selected_features]
    x_train, x_test, y_train, y_test = train_test_split(x_selected, y_smote, test_size=0.2, random_state=42)
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

    # Display results
    col1, col2, col3 = st.columns(3)
    col1.metric("Decision Tree Accuracy", f"{results['Decision Tree']*100:.2f}%")
    col2.metric("Random Forest Accuracy", f"{results['Random Forest']*100:.2f}%")
    col3.metric("SVM Accuracy", f"{results['Support Vector Machines']*100:.2f}%")

# def display_model_results(model, model_name, x, y):
#     st.write(f"### {model_name}")
#     results = tabulize_model_results(model, model_name, x, y)
#     st.write(results)

# decision_tree_model = DecisionTreeClassifier(random_state=42)
# display_model_results(decision_tree_model, "Decision Tree", x_smote, y_smote)
# display_model_results(decision_tree_model, "Decision Tree (feature selection)", x_smote_featureSelection, y_smote)

# random_forest_model = RandomForestClassifier(random_state=42)
# display_model_results(random_forest_model, "Random Forest", x_smote, y_smote)
# display_model_results(random_forest_model, "Random Forest with feature selection", x_smote_featureSelection, y_smote)

# svm_model = SVC(random_state=42)
# display_model_results(svm_model, "Support Vector Machines", x_smote, y_smote)
# display_model_results(svm_model, "Support Vector Machines with feature selection", x_smote_featureSelection, y_smote)
