import pandas as pd

def count_null_values(df: pd.DataFrame) -> int:
    """
    Count the total number of null values in the entire DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame to check for null values.
    
    Returns:
    int: Total count of null values in the DataFrame.
    """
    total_nulls = df.isnull().sum().sum()
    return total_nulls

def identify_null_columns(df: pd.DataFrame) -> pd.Series:
    """
    Identify columns with null values and show the count of null values for each column.
    
    Args:
    df (pd.DataFrame): The DataFrame to check for null values.
    
    Returns:
    pd.Series: A series with column names as index and count of null values as values.
    """
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    return null_columns

def analyze_nulls(df: pd.DataFrame):
    """
    Analyze null values in the DataFrame by counting total null values and identifying
    columns with null values along with their counts.
    
    Args:
    df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    dict: A dictionary containing the total null count and a series with null counts per column.
    """
    total_null_count = count_null_values(df)
    null_columns_count = identify_null_columns(df)
    
    return {
        'total_null_count': total_null_count,
        'null_columns_count': null_columns_count
    }