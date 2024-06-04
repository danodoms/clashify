import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency



def cramers_v_analysis(data, target_column, threshold=0.6, top_n=5, plot=True, figsize=None):
    """
    Perform Cramér's V analysis on a DataFrame.

    Parameters:
    data (DataFrame): Input DataFrame.
    target_column (str): Name of the target column.
    threshold (float): Threshold for high association.
    top_n (int): Number of top features to list.
    plot (bool): Whether to plot the heatmap.
    figsize (tuple): Size of the figure (width, height).

    Returns:
    dict: Dictionary containing the top features with high Cramér's V scores.
    """

    # Function to calculate Cramér's V
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        return (chi2 / (n * (min(confusion_matrix.shape) - 1))) ** 0.5

    # Calculate Cramér's V for all pairs of features
    cramers_v_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != target_column and col2 != target_column:
                cramers_v_matrix.loc[col1, col2] = cramers_v(data[col1], data[col2])

    # Plot heatmap
    if plot:
        if figsize:
            plt.figure(figsize=figsize)
        sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm', linewidths=0.5, vmin=0, vmax=1)
        plt.title("Cramér's V Heatmap")
        plt.show()

    # Drop highly associated features
    to_drop = set()
    for col1 in cramers_v_matrix.columns:
        for col2 in cramers_v_matrix.columns:
            if col1 != col2 and cramers_v_matrix.loc[col1, col2] > threshold:
                to_drop.add(col2)

    # Extract top features with high Cramér's V scores
    top_features = cramers_v_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(top_n).to_dict()

    # Remove target column and dropped columns
    remaining_features = data.drop(columns=to_drop).drop(columns=target_column)

    return top_features, remaining_features

# Example usage:
# data = pd.read_csv('data/your_data.csv')
# top_features, remaining_features = cramers_v_analysis(data, 'target_column', threshold=0.25, top_n=5, plot=True, figsize=(12, 10))






def chi2_feature_significance(data, target, significance_threshold=0.05):
    """
    Calculate chi-square feature significance for each feature in the dataset.

    Parameters:
    data (DataFrame): Input DataFrame containing features and target variable.
    target (str): Name of the target variable column.
    significance_threshold (float): Threshold for significance.

    Returns:
    DataFrame: DataFrame containing feature significance and p-values, sorted by significance value.
    """

    # Split data into features (X) and target variable (Y)
    X = data.drop(target, axis=1)
    Y = data[target]

    # Perform chi-square test for each feature
    feature_significance = {}
    for col in X:
        contingency_table = pd.crosstab(X[col], Y)
        chi2, p_val, _, _ = chi2_contingency(contingency_table)
        significance = "Significant" if p_val < significance_threshold else "Not Significant"
        feature_significance[col] = {"Significance": significance, "p-value": p_val}

    # Create DataFrame to store chi-square test results
    chi2_df = pd.DataFrame(feature_significance).T
    chi2_df.index.name = 'Feature'
    chi2_df.reset_index(inplace=True)

    # Sort DataFrame by significance value
    chi2_df_sorted = chi2_df.sort_values(by='p-value', ascending=True)

    return chi2_df_sorted

# Example usage:
# chi2_results = chi2_feature_significance(df_encoded, 'Feedback', significance_threshold=0.05)
# print(chi2_results)

