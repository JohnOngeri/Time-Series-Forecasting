"""
Data utilities for air quality forecasting project.
Handles data loading, preprocessing, and sequence generation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_data(train_path='train.csv', test_path='test.csv'):
    """
    Load training and test datasets.
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to test data
        
    Returns:
        tuple: (train_df, test_df)
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Convert datetime columns
    train['datetime'] = pd.to_datetime(train['datetime'])
    test['datetime'] = pd.to_datetime(test['datetime'])
    
    # Set datetime as index
    train.set_index('datetime', inplace=True)
    test.set_index('datetime', inplace=True)
    
    return train, test

def get_data_summary(df, name="Dataset"):
    """
    Generate comprehensive data summary statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        name (str): Name for the dataset
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'name': name,
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'date_range': (df.index.min(), df.index.max()) if hasattr(df.index, 'min') else None
    }
    return summary

def handle_missing_values(df, strategy='mean', target_col='pm2.5'):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Imputation strategy ('mean', 'median', 'forward_fill', 'interpolate')
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    if strategy == 'mean':
        # Use mean imputation for numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        
    elif strategy == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        
    elif strategy == 'forward_fill':
        df_clean = df_clean.fillna(method='ffill')
        
    elif strategy == 'interpolate':
        df_clean = df_clean.interpolate(method='time')
        
    # Final cleanup - fill any remaining NaN with mean
    df_clean = df_clean.fillna(df_clean.mean())
    
    return df_clean

def create_time_features(df):
    """
    Create time-based features from datetime index.
    
    Args:
        df (pd.DataFrame): Input dataframe with datetime index
        
    Returns:
        pd.DataFrame: DataFrame with additional time features
    """
    df_features = df.copy()
    
    # Extract time components
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['day_of_year'] = df_features.index.dayofyear
    
    # Cyclical encoding for time features
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    return df_features

def create_lag_features(df, target_col='pm2.5', lags=[1, 2, 3, 6, 12, 24]):
    """
    Create lag features for time series forecasting.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        lags (list): List of lag periods
        
    Returns:
        pd.DataFrame: DataFrame with lag features
    """
    df_lag = df.copy()
    
    for lag in lags:
        df_lag[f'{target_col}_lag_{lag}'] = df_lag[target_col].shift(lag)
    
    return df_lag

def create_rolling_features(df, target_col='pm2.5', windows=[3, 6, 12, 24]):
    """
    Create rolling window features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        windows (list): List of window sizes
        
    Returns:
        pd.DataFrame: DataFrame with rolling features
    """
    df_roll = df.copy()
    
    for window in windows:
        df_roll[f'{target_col}_rolling_mean_{window}'] = df_roll[target_col].rolling(window=window).mean()
        df_roll[f'{target_col}_rolling_std_{window}'] = df_roll[target_col].rolling(window=window).std()
        df_roll[f'{target_col}_rolling_min_{window}'] = df_roll[target_col].rolling(window=window).min()
        df_roll[f'{target_col}_rolling_max_{window}'] = df_roll[target_col].rolling(window=window).max()
    
    return df_roll

def scale_features(X_train, X_val=None, X_test=None, scaler_type='standard'):
    """
    Scale features using specified scaler.
    
    Args:
        X_train (np.array): Training features
        X_val (np.array): Validation features (optional)
        X_test (np.array): Test features (optional)
        scaler_type (str): Type of scaler ('standard', 'minmax')
        
    Returns:
        tuple: Scaled arrays and fitted scaler
    """
    # Clean data: replace inf and extreme values
    def clean_data(X):
        # Ensure X is numeric and convert to float64
        X_clean = np.array(X, dtype=np.float64)
        
        # Replace infinity with NaN
        X_clean[np.isinf(X_clean)] = np.nan
        
        # Replace extreme values (beyond 6 standard deviations)
        for col in range(X_clean.shape[1]):
            col_data = X_clean[:, col]
            if not np.all(np.isnan(col_data)):
                mean_val = np.nanmean(col_data)
                std_val = np.nanstd(col_data)
                if std_val > 0:
                    threshold = 6 * std_val
                    mask = np.abs(col_data - mean_val) > threshold
                    X_clean[mask, col] = np.nan
        
        # Fill NaN with column mean
        for col in range(X_clean.shape[1]):
            col_data = X_clean[:, col]
            if np.any(np.isnan(col_data)):
                mean_val = np.nanmean(col_data)
                if np.isnan(mean_val):
                    mean_val = 0
                X_clean[np.isnan(col_data), col] = mean_val
        
        return X_clean
    
    X_train_clean = clean_data(X_train)
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    X_train_scaled = scaler.fit_transform(X_train_clean)
    
    results = [X_train_scaled, scaler]
    
    if X_val is not None:
        X_val_clean = clean_data(X_val)
        X_val_scaled = scaler.transform(X_val_clean)
        results.append(X_val_scaled)
    
    if X_test is not None:
        X_test_clean = clean_data(X_test)
        X_test_scaled = scaler.transform(X_test_clean)
        results.append(X_test_scaled)
    
    return tuple(results)

def create_sequences(data, target, window_size=24, forecast_horizon=1):
    """
    Create sequences for time series forecasting.
    
    Args:
        data (np.array): Input features
        target (np.array): Target values
        window_size (int): Number of time steps to look back
        forecast_horizon (int): Number of steps to forecast ahead
        
    Returns:
        tuple: (X_sequences, y_sequences)
    """
    X, y = [], []
    
    for i in range(window_size, len(data) - forecast_horizon + 1):
        X.append(data[i-window_size:i])
        y.append(target[i:i+forecast_horizon])
    
    return np.array(X), np.array(y)

def time_series_split(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split time series data chronologically.
    
    Args:
        df (pd.DataFrame): Input dataframe
        train_ratio (float): Proportion for training
        val_ratio (float): Proportion for validation
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

def prepare_data_for_modeling(train_df, test_df, target_col='pm2.5', 
                            window_size=24, scaler_type='standard',
                            include_time_features=True, include_lags=True, 
                            include_rolling=True):
    """
    Complete data preparation pipeline for modeling.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        target_col (str): Target column name
        window_size (int): Sequence window size
        scaler_type (str): Scaler type
        include_time_features (bool): Whether to include time features
        include_lags (bool): Whether to include lag features
        include_rolling (bool): Whether to include rolling features
        
    Returns:
        dict: Prepared data dictionary
    """
    # Handle missing values
    train_clean = handle_missing_values(train_df, strategy='interpolate', target_col=target_col)
    test_clean = handle_missing_values(test_df, strategy='interpolate', target_col=target_col)
    
    # Add time features
    if include_time_features:
        train_clean = create_time_features(train_clean)
        test_clean = create_time_features(test_clean)
    
    # Add lag features
    if include_lags and target_col in train_clean.columns:
        train_clean = create_lag_features(train_clean, target_col)
        # For test set, we need to be careful with lags
        combined = pd.concat([train_clean, test_clean])
        combined = create_lag_features(combined, target_col)
        test_clean = combined.iloc[len(train_clean):]
    
    # Add rolling features
    if include_rolling and target_col in train_clean.columns:
        train_clean = create_rolling_features(train_clean, target_col)
        # For test set, use combined approach
        combined = pd.concat([train_clean, test_clean])
        combined = create_rolling_features(combined, target_col)
        test_clean = combined.iloc[len(train_clean):]
    
    # Remove rows with NaN (due to lag/rolling features)
    train_clean = train_clean.dropna()
    test_clean = test_clean.dropna()
    
    # Split features and target
    feature_cols = [col for col in train_clean.columns if col != target_col and col != 'No']
    
    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values if target_col in train_clean.columns else None
    X_test = test_clean[feature_cols].values
    
    # Time series split for validation
    if y_train is not None:
        train_split, val_split, _ = time_series_split(train_clean)
        X_train_split = train_split[feature_cols].values
        y_train_split = train_split[target_col].values
        X_val = val_split[feature_cols].values
        y_val = val_split[target_col].values
    else:
        X_train_split, y_train_split = X_train, y_train
        X_val, y_val = None, None
    
    # Scale features
    if X_val is not None:
        X_train_scaled, scaler, X_val_scaled, X_test_scaled = scale_features(
            X_train_split, X_val, X_test, scaler_type
        )
    else:
        X_train_scaled, scaler, X_test_scaled = scale_features(
            X_train_split, None, X_test, scaler_type
        )
        X_val_scaled = None
    
    # Create sequences
    if y_train_split is not None:
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_split, window_size)
        if X_val_scaled is not None and y_val is not None:
            X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, window_size)
        else:
            X_val_seq, y_val_seq = None, None
    else:
        X_train_seq, y_train_seq = None, None
        X_val_seq, y_val_seq = None, None
    
    # For test sequences, we need to handle the case where we don't have targets
    X_test_seq = []
    for i in range(window_size, len(X_test_scaled) + 1):
        X_test_seq.append(X_test_scaled[i-window_size:i])
    X_test_seq = np.array(X_test_seq) if X_test_seq else None
    
    return {
        'X_train': X_train_seq,
        'y_train': y_train_seq,
        'X_val': X_val_seq,
        'y_val': y_val_seq,
        'X_test': X_test_seq,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'train_clean': train_clean,
        'test_clean': test_clean
    }
