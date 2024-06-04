import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encode_all(df, exclude_columns=[]):
    """
    Apply label encoding to all columns of the DataFrame except the specified columns.

    Parameters:
    df (DataFrame): Input DataFrame.
    exclude_columns (list): List of columns to exclude from label encoding.

    Returns:
    DataFrame: DataFrame with specified columns label encoded.
    """
    df_encoded = df.copy()  # Create a copy to avoid modifying the original DataFrame
    label_encoder = LabelEncoder()
    
    for column in df_encoded.columns:
        if column not in exclude_columns:
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column].astype(str))
    
    return df_encoded