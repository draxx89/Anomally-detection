import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_convert_mixed_types(df):
    mixed_type_columns = {}
    for col in df.columns:
        types_in_col = df[col].apply(type).unique()
        if len(types_in_col) > 1:
            mixed_type_columns[col] = types_in_col
            df[col] = df[col].astype(str)
    return mixed_type_columns, df

def check_mixed(df):
    mixed_type_columns = {}
    for col in df.columns:
        types_in_col = df[col].apply(type).unique()
        if len(types_in_col) > 1:
            mixed_type_columns[col] = types_in_col
    return mixed_type_columns

def convert_to_DateTime_format(time_dimension_column, df):
    try:
        df[time_dimension_column] = pd.to_datetime(df[time_dimension_column], format='%Y%m%d', errors='coerce')
        df['Month'] = df[time_dimension_column].dt.month
        df['Year'] = df[time_dimension_column].dt.year
        return df
    except KeyError:
        st.error(f"Column '{time_dimension_column}' not found in the Dataframe.")
        return None

def group_data(df, time_dimension_column, series_columns, group_by_column, time_dimension='monthly'):
    df[time_dimension_column] = pd.to_datetime(df[time_dimension_column], format='%Y%m%d', errors='coerce')
    if time_dimension == 'daily':
        df['time_dim'] = df[time_dimension_column].dt.to_period('D')
    elif time_dimension == 'weekly':
        df['time_dim'] = df[time_dimension_column].dt.to_period('W')
    elif time_dimension == 'monthly':
        df['time_dim'] = df[time_dimension_column].dt.to_period('M')
    elif time_dimension == 'yearly':
        df['time_dim'] = df[time_dimension_column].dt.to_period('Y')
    else:
        raise ValueError("Invalid time dimension. Choose from 'daily', 'weekly', 'monthly', 'yearly'.")
    grouped_data = df.groupby(['time_dim', group_by_column])[series_columns].sum().reset_index()
    return grouped_data

def standardize_data(grouped_df, series_columns):
    scaler = StandardScaler()
    standard_df = grouped_df.copy()
    standard_df[series_columns] = scaler.fit_transform(standard_df[series_columns])
    return standard_df, scaler

def detect_outliers_zscore(grouped_data, group_by_column, series_column, threshold=2):
    customer_id = grouped_data[group_by_column].iloc[0]
    mean_val = grouped_data[series_column].mean()
    std_val = grouped_data[series_column].std()
    grouped_data['Z-Score'] = (grouped_data[series_column] - mean_val) / std_val
    grouped_data['Outlier'] = grouped_data['Z-Score'].abs() > threshold
    results = grouped_data.copy()
    results['customer_id'] = customer_id
    return results

def process_batch(batch, group_by_column, series_column, threshold):
    local_results = []
    for group in batch:
        try:
            local_results.append(detect_outliers_zscore(group, group_by_column, series_column, threshold))
        except Exception as e:
            logger.error(f"Error processing group: {e}")
    return local_results

def parallel_outlier_detection(customer_groups, detection_method, group_by_column, series_column, threshold=2):
    results = []
    batch_size = 100
    batches = [customer_groups[i:i + batch_size] for i in range(0, len(customer_groups), batch_size)]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_batch, batch, group_by_column, series_column, threshold) for batch in batches]
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as e:
                logger.error(f"Error in future: {e}")
    return pd.concat(results, ignore_index=True)

# Streamlit UI
st.title('Anomaly Detection Dashboard')

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.write("Data loaded successfully!")
    
    mixed_type, df = check_and_convert_mixed_types(df)
    if mixed_type:
        st.write("Columns with mixed data types:")
        for col, types in mixed_type.items():
            st.write(f"{col}: {types}")
    
    time_dimension_column = st.selectbox("Select time dimension column:", df.columns)
    
    df = convert_to_DateTime_format(time_dimension_column, df)
    
    series_columns = st.multiselect("Select series columns to analyze:", df.columns)
    group_by_column = st.selectbox("Select group by column:", df.columns)
    
    if st.button("Process Data"):
        filtered_df = df.dropna(subset=[time_dimension_column])
        data = group_data(filtered_df, time_dimension_column, series_columns, group_by_column)
        standardized_data, scaler = standardize_data(data, series_columns)
        
        customer_groups = [group for _, group in standardized_data.groupby(group_by_column)]
        series_column_for_outlier = st.selectbox("Select series column for outlier detection:", series_columns)
        
        outlier_results = parallel_outlier_detection(customer_groups, 'zscore', group_by_column, series_column_for_outlier)
        
        customers_with_outliers = outlier_results[outlier_results['Outlier'] == 1]['customer_id'].unique()
        filtered_outlier_results = outlier_results[outlier_results['customer_id'].isin(customers_with_outliers)]
        
        merged_table = pd.merge(filtered_outlier_results, data, on=[group_by_column, 'time_dim'], suffixes=('_std', '_actual'))
        
        st.write("Merged results:")
        st.dataframe(merged_table.head())
        
        if st.button("Save Results"):
            save_path = st.text_input("Enter path to save CSV file:", "merged_outlier_results.csv")
            merged_table.to_csv(save_path, index=False)
            st.success(f"Results saved to {save_path}")
