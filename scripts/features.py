"""
Feature engineering utilities for air quality forecasting.
"""

import pandas as pd
import numpy as np
from scipy import stats

def create_interaction_features(df, feature_pairs=None):
    """
    Create interaction features between weather variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_pairs (list): List of feature pairs to interact
        
    Returns:
        pd.DataFrame: DataFrame with interaction features
    """
    df_interact = df.copy()
    
    if feature_pairs is None:
        # Default weather interactions
        feature_pairs = [
            ('TEMP', 'DEWP'),  # Temperature-humidity interaction
            ('TEMP', 'PRES'),  # Temperature-pressure interaction
            ('Iws', 'PRES'),   # Wind-pressure interaction
        ]
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df_interact.columns and feat2 in df_interact.columns:
            df_interact[f'{feat1}_{feat2}_interact'] = df_interact[feat1] * df_interact[feat2]
    
    return df_interact

def create_pollution_ratios(df):
    """
    Create ratios between different pollution indicators.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with ratio features
    """
    df_ratios = df.copy()
    
    # Solar radiation ratios
    if 'Is' in df_ratios.columns and 'Ir' in df_ratios.columns:
        df_ratios['solar_ratio'] = df_ratios['Is'] / (df_ratios['Ir'] + 1e-8)
    
    # Wind components
    if 'Iws' in df_ratios.columns:
        df_ratios['wind_squared'] = df_ratios['Iws'] ** 2
        df_ratios['wind_log'] = np.log1p(df_ratios['Iws'])
    
    return df_ratios

def create_seasonal_features(df):
    """
    Create seasonal and periodic features.
    
    Args:
        df (pd.DataFrame): Input dataframe with datetime index
        
    Returns:
        pd.DataFrame: DataFrame with seasonal features
    """
    df_seasonal = df.copy()
    
    # Season encoding
    month = df_seasonal.index.month
    df_seasonal['season'] = ((month % 12 + 3) // 3).map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'})
    
    # One-hot encode seasons
    season_dummies = pd.get_dummies(df_seasonal['season'], prefix='season')
    df_seasonal = pd.concat([df_seasonal, season_dummies], axis=1)
    df_seasonal.drop('season', axis=1, inplace=True)
    
    # Weekend indicator
    df_seasonal['is_weekend'] = (df_seasonal.index.dayofweek >= 5).astype(int)
    
    # Rush hour indicators
    hour = df_seasonal.index.hour
    df_seasonal['is_morning_rush'] = ((hour >= 7) & (hour <= 9)).astype(int)
    df_seasonal['is_evening_rush'] = ((hour >= 17) & (hour <= 19)).astype(int)
    
    return df_seasonal

def create_statistical_features(df, target_col='pm2.5', windows=[6, 12, 24, 48]):
    """
    Create statistical features over different time windows.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        windows (list): List of window sizes
        
    Returns:
        pd.DataFrame: DataFrame with statistical features
    """
    df_stats = df.copy()
    
    if target_col not in df_stats.columns:
        return df_stats
    
    for window in windows:
        # Rolling statistics
        rolling = df_stats[target_col].rolling(window=window, min_periods=1)
        
        df_stats[f'{target_col}_rolling_mean_{window}'] = rolling.mean()
        df_stats[f'{target_col}_rolling_std_{window}'] = rolling.std()
        df_stats[f'{target_col}_rolling_min_{window}'] = rolling.min()
        df_stats[f'{target_col}_rolling_max_{window}'] = rolling.max()
        df_stats[f'{target_col}_rolling_median_{window}'] = rolling.median()
        df_stats[f'{target_col}_rolling_skew_{window}'] = rolling.skew()
        
        # Rolling percentiles
        df_stats[f'{target_col}_rolling_q25_{window}'] = rolling.quantile(0.25)
        df_stats[f'{target_col}_rolling_q75_{window}'] = rolling.quantile(0.75)
        
        # Rate of change
        df_stats[f'{target_col}_pct_change_{window}'] = df_stats[target_col].pct_change(periods=window)
        
        # Difference from rolling mean
        df_stats[f'{target_col}_diff_from_mean_{window}'] = (
            df_stats[target_col] - df_stats[f'{target_col}_rolling_mean_{window}']
        )
    
    return df_stats

def create_weather_indices(df):
    """
    Create composite weather indices.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with weather indices
    """
    df_indices = df.copy()
    
    # Heat index approximation (simplified)
    if 'TEMP' in df_indices.columns and 'DEWP' in df_indices.columns:
        # Normalize temperature and dew point for index calculation
        temp_norm = (df_indices['TEMP'] - df_indices['TEMP'].min()) / (df_indices['TEMP'].max() - df_indices['TEMP'].min())
        dewp_norm = (df_indices['DEWP'] - df_indices['DEWP'].min()) / (df_indices['DEWP'].max() - df_indices['DEWP'].min())
        df_indices['heat_index'] = temp_norm + 0.5 * dewp_norm
    
    # Atmospheric stability index
    if 'TEMP' in df_indices.columns and 'PRES' in df_indices.columns:
        df_indices['stability_index'] = df_indices['TEMP'] / (df_indices['PRES'] + 1e-8)
    
    # Wind chill approximation
    if 'TEMP' in df_indices.columns and 'Iws' in df_indices.columns:
        df_indices['wind_chill'] = df_indices['TEMP'] - 0.1 * df_indices['Iws']
    
    return df_indices

def select_features_by_correlation(df, target_col='pm2.5', threshold=0.05):
    """
    Select features based on correlation with target variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        threshold (float): Minimum correlation threshold
        
    Returns:
        list: Selected feature names
    """
    if target_col not in df.columns:
        return [col for col in df.columns if col != 'No']
    
    # Calculate correlations
    correlations = df.corr()[target_col].abs()
    
    # Select features above threshold
    selected_features = correlations[correlations >= threshold].index.tolist()
    
    # Remove target column itself
    if target_col in selected_features:
        selected_features.remove(target_col)
    
    # Remove 'No' column if present
    if 'No' in selected_features:
        selected_features.remove('No')
    
    return selected_features

def create_comprehensive_features(df, target_col='pm2.5'):
    """
    Create comprehensive feature set combining all feature engineering techniques.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: DataFrame with comprehensive features
    """
    df_features = df.copy()
    
    # Time features
    df_features = create_time_features(df_features)
    
    # Interaction features
    df_features = create_interaction_features(df_features)
    
    # Pollution ratios
    df_features = create_pollution_ratios(df_features)
    
    # Seasonal features
    df_features = create_seasonal_features(df_features)
    
    # Statistical features
    df_features = create_statistical_features(df_features, target_col)
    
    # Weather indices
    df_features = create_weather_indices(df_features)
    
    # Lag features
    if target_col in df_features.columns:
        df_features = create_lag_features(df_features, target_col, lags=[1, 2, 3, 6, 12, 24, 48])
    
    return df_features

def create_time_features(df):
    """
    Create time-based features from datetime index.
    (Imported from data_utils for consistency)
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
    (Imported from data_utils for consistency)
    """
    df_lag = df.copy()
    
    for lag in lags:
        df_lag[f'{target_col}_lag_{lag}'] = df_lag[target_col].shift(lag)
    
    return df_lag
