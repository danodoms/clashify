def balance_data(x, y):
    from imblearn.over_sampling import SMOTE
    import pandas as pd
    
    x = pd.DataFrame(x).round().astype(int)
    
    smote = SMOTE(random_state=42)
    x_smote, y_smote = smote.fit_resample(x, y)
    return x_smote, y_smote



def train_model(x, y, model):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return X_train, X_test, y_train, y_test, y_pred



def evaluate_model(y_test, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, mean_squared_error, mean_absolute_error, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # %% Confusion Matrix
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.title("Confusion Matrix")
    # plt.show()
    
    # Classification Report
    # print(classification_report(y_test, y_pred))
    
    # %% Confusion Matrix
    return accuracy, f1, precision, mse, mae



def tabulize_model_results(model, model_name, x_smote, y_smote):
    import pandas as pd
    results = {}
    
    X_train, X_test, y_train, y_test, y_pred = train_model(x_smote, y_smote, model)
    results[model_name] = evaluate_model(y_test, y_pred)
    
    # Create and return the metrics table
    metrics_df = pd.DataFrame(results).T
    metrics_df.columns = ["Accuracy", "F1-score", "Precision", "Mean Squared Error", "Mean Absolute Error"]
    return metrics_df