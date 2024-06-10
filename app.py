import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydeck as pdk

# Custom functions (assumed to be provided)
from library.model import balance_data

# Set page configuration
st.set_page_config(
    page_title="Online Food Order Dataset 🍔",
    layout="centered",
    page_icon="🍔"
)

# Sidebar navigation
st.sidebar.title("Online Food Order Dataset")
st.sidebar.write("Dominador G. Dano Jr.")
st.sidebar.divider()
st.sidebar.subheader("Navigation")

sections = ["Home", "Dataset", "Map", "Column Visualizations", "Age Metrics", "Feedback Metrics", "Interactive Model Testing"]
selection = st.sidebar.radio("Go to", sections)

# Load dataset
data = pd.read_csv('onlinefoods.csv')

# Sidebar filters and interactivity
st.sidebar.subheader("Filters")

# Date range selector
date_column = 'Order Date'  # Example date column name
if date_column in data.columns:
    min_date = data[date_column].min()
    max_date = data[date_column].max()
    start_date, end_date = st.sidebar.date_input("Select date range", [min_date, max_date])
    data = data[(data[date_column] >= start_date) & (data[date_column] <= end_date)]

# Numeric filters
for column in data.select_dtypes(include=[np.number]).columns:
    min_val = data[column].min()
    max_val = data[column].max()
    selected_range = st.sidebar.slider(f"Filter {column}", float(min_val), float(max_val), (float(min_val), float(max_val)))
    data = data[(data[column] >= selected_range[0]) & (data[column] <= selected_range[1])]

# Boolean filters (example)
if 'Delivery On Time' in data.columns:
    delivery_on_time = st.sidebar.checkbox("Only show orders delivered on time", value=True)
    if delivery_on_time:
        data = data[data['Delivery On Time']]


# Home Section
if selection == "Home":
    st.title("Online Food Order Dataset")
    st.write("""
    The dataset contains information collected from an online food ordering platform over a period of time.
    It encompasses various attributes related to Occupation, Family Size, Feedback, etc.
    Explore the data and gain insights through the different sections provided.
    """)
    st.markdown("**Source:** [Kaggle](https://www.kaggle.com/datasets/sudarshan24byte/online-food-dataset/data)")
    st.divider()

# Dataset Section
if selection == "Dataset":
    st.title("Dataset")
    st.write("## Dataset")
    st.write("""
    This section displays the raw dataset containing information about online food orders. 
    You can scroll through the table to see all the data points collected.
    """)
    st.write(data)

# Map Section
if selection == "Map":
    st.title("Map")
    st.write("## Top View")
    st.write("""
    This map shows the locations where data was collected. 
    It provides a geographic distribution of the orders.
    """)
    st.map(data[['latitude', 'longitude']])

    # Define the initial view state for the map
    view_state = pdk.ViewState(
        latitude=data['latitude'].mean(),
        longitude=data['longitude'].mean(),
        zoom=11,
        pitch=50,
    )

    # Define the layer for the map
    layer = pdk.Layer(
        'HexagonLayer',
        data=data[['latitude', 'longitude']],
        get_position='[longitude, latitude]',
        radius=200,
        elevation_scale=4,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
    )

    # Define the deck
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{position}\nElevation: {elevationValue}"}
    )

    st.write("## With distribution")

    # Render the map
    st.pydeck_chart(r)

# Column Visualizations Section
if selection == "Column Visualizations":
    st.title("Column Visualizations")
    st.write("""
    This section provides visualizations for each column in the dataset. 
    It includes different types of charts and plots to help you understand the distribution and characteristics of the data.
    """)

    chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Area Chart"])

    # Numeric columns
    st.write("### Numeric Columns")
    for column in data.select_dtypes(include=[np.number]).columns:
        st.write(f"#### {column}")
        if chart_type == "Line Chart" or data[column].nunique() > 20:
            st.line_chart(data[column])
        elif chart_type == "Area Chart":
            st.area_chart(data[column])
        else:
            st.bar_chart(data[column].value_counts())

    # Categorical columns
    st.write("### Categorical Columns")
    for column in data.select_dtypes(include=[object]).columns:
        st.write(f"#### {column}")
        st.bar_chart(data[column].value_counts())

    # Specific visualizations for some columns
    st.write("### Specific Visualizations")
    
    # Gender distribution
    st.write("#### Gender Distribution")
    st.bar_chart(data['Gender'].value_counts())
    
    # Age distribution
    st.write("#### Age Distribution")
    st.bar_chart(data['Age'].value_counts().sort_index())
    
    # Family size distribution
    st.write("#### Family Size Distribution")
    st.area_chart(data['Family size'].value_counts().sort_index())
    
    # Feedback distribution
    if 'Feedback' in data.columns:
        st.write("#### Feedback Distribution")
        feedback_counts = data['Feedback'].value_counts()
        st.write(feedback_counts.plot.pie(autopct='%1.1f%%').get_figure())
    else:
        st.write("#### Feedback Distribution")
        st.write("Feedback column is not available in the dataset.")

# Age Metrics Section
if selection == "Age Metrics":
    st.title("Age Metrics")
    st.write("""
    This section provides metrics related to the age of the customers. 
    It displays the average age, minimum age, and maximum age of the customers in the dataset.
    """)
    average_age = data['Age'].mean()
    min_age = data['Age'].min()
    max_age = data['Age'].max()
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Age", f"{average_age:.2f}")
    col2.metric("Min Age", f"{min_age}")
    col3.metric("Max Age", f"{max_age}")

# Feedback Metrics Section
if selection == "Feedback Metrics":
    st.title("Feedback Metrics")
    st.write("""
    This section provides metrics related to customer feedback. 
    It displays the count of each type of feedback received.
    """)
    if 'Feedback' in data.columns:
        feedback_types = data['Feedback'].value_counts()
        for feedback_type, count in feedback_types.items():
            st.metric(f"{feedback_type} Feedback Count", count)
    else:
        st.write("Feedback column is not available in the dataset.")

# Interactive Model Testing Section
if selection == "Interactive Model Testing":
    st.title("Interactive Model Testing")
    st.write("""
    In this section, you can test different machine learning models on the dataset. 
    Select the features you want to include in the model and see the accuracy of Decision Tree, Random Forest, and SVM classifiers.
    """)
    
    # Exclude 'Feedback' from selectable features
    selectable_features = data.columns.drop('Feedback')
    
    selected_features = st.multiselect(
        'Select features to include in the model',
        options=selectable_features,
        default=selectable_features.tolist()
    )

    # Encode categorical columns
    data_encoded = data.copy()
    label_encoders = {}
    for column in data_encoded.select_dtypes(include=[object]).columns:
        le = LabelEncoder()
        data_encoded[column] = le.fit_transform(data_encoded[column])
        label_encoders[column] = le

    if 'Feedback' in data_encoded.columns:
        # Define models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Support Vector Machines': SVC(random_state=42)
        }

        if st.button('Evaluate'):
            x = data_encoded.drop(columns=['Feedback'], axis=1)
            y = data_encoded['Feedback']
            x_smote, y_smote = balance_data(x, y)
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
    else:
        st.write("Feedback column is not available in the dataset.")
