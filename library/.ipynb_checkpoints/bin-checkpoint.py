import pandas as pd
from sklearn.preprocessing import LabelEncoder

def bin_numeric_columns(df, binning_info):
    """
    Convert numeric columns into categories by binning and replace the original columns with the new binned columns.

    Parameters:
    df (DataFrame): Input DataFrame.
    binning_info (dict): Dictionary where keys are column names to bin and values are tuples of (bins, labels).

    Returns:
    DataFrame: DataFrame with original numeric columns replaced by binned ones.
    """
    df_binned = df.copy()  # Create a copy to avoid modifying the original DataFrame
    
    for column, (bins, labels) in binning_info.items():
        # Bin the numeric column
        df_binned[column] = pd.cut(df_binned[column], bins=bins, labels=labels)
    
    return df_binned

def label_encode_columns(df, exclude_columns=[]):
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
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column])
    
    return df_encoded


# Example usage
if __name__ == "__main__":
    # Sample DataFrame based on provided data
    # data = pd.read_csv("data/TravelInsurancePrediction.csv")
    data = pd.read_csv("data/onlinefoods.csv")
    
    df = pd.DataFrame(data)
    
    # Define binning information
    binning_info = {
        # 'AnnualIncome': ([0, 500000, 1000000, 1500000], ['Low', 'Medium', 'High']),
        'Age': ([17, 22, 26, 29, 34], ['1', '2', '3', '4']),
        'Pin code':([560000, 560025, 560050, 560075, 560100, 560125], ['1', '2', '3', '4', '5']),
        # 'latitude':([12.8, 12.9, 13.0, 13.1, 13.2], ['1', '2', '3', '4']),
        # 'longitude':([77.4, 77.5, 77.6, 77.7, 77.8], ['1', '2', '3', '4'])
    }
    
    # # Minimum and maximum values
    # min_value = df["longitude"].min()
    # max_value = df["longitude"].max()
    
    # print(f"Minimum value of : {min_value}")
    # print(f"Maximum value of : {max_value}")

    

   
    
    
    # Apply label encoding to all columns
    df_encoded = label_encode_columns(df, ['Age', 'Pin code'])
    
    print("\nDataFrame with All Columns Label Encoded:")
    print(df_encoded)
    
    print(df_encoded.describe())
    desc = df_encoded.describe()
    
    
    # Bin numeric columns and replace original ones
    df_binned = bin_numeric_columns(df_encoded, binning_info)
    
    print("DataFrame with Original Columns Replaced by Binned Ones:")
    print(df_binned)
    
    # Export the encoded DataFrame to a CSV file
    df_binned.to_csv('data/onlinefoods_encoded.csv', index=False)
    