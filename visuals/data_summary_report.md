# Air Quality Dataset Summary Report
Generated on: 2025-09-18 09:03:49

## Dataset Overview
- **Shape**: 30676 rows × 11 columns
- **Date Range**: 2010-01-01 00:00:00 to 2013-07-02 03:00:00
- **Duration**: 1278 days
- **Frequency**: h

## Missing Values
- **pm2.5**: 1921 (6.3%)

## pm2.5 Statistics
- **Mean**: 100.79 μg/m³
- **Median**: 75.00 μg/m³
- **Standard Deviation**: 93.14 μg/m³
- **Range**: 0.00 - 994.00 μg/m³
- **IQR**: 29.00 - 142.00 μg/m³

### Air Quality Categories (WHO Guidelines)
- **Good (≤15)**: 3240 (11.3%)
- **Moderate (15-35)**: 4968 (17.3%)
- **Unhealthy (35-75)**: 6245 (21.7%)
- **Very Unhealthy (>75)**: 14302 (49.7%)

## Feature Correlations with pm2.5
- **Iws**: -0.260
- **cbwd_NW**: -0.231
- **DEWP**: 0.218
- **cbwd_cv**: 0.158
- **cbwd_SE**: 0.119
- **PRES**: -0.108
- **Ir**: -0.052
- **TEMP**: -0.040
- **Is**: 0.022
- **No**: 0.018

## Seasonal Patterns (Monthly Averages)
- **Jan**: 114.12 μg/m³
- **Feb**: 113.64 μg/m³
- **Mar**: 94.40 μg/m³
- **Apr**: 80.64 μg/m³
- **May**: 82.15 μg/m³
- **Jun**: 106.25 μg/m³
- **Jul**: 104.92 μg/m³
- **Aug**: 93.80 μg/m³
- **Sep**: 88.73 μg/m³
- **Oct**: 118.17 μg/m³
- **Nov**: 111.32 μg/m³
- **Dec**: 104.76 μg/m³

## Data Quality Assessment
- **Outliers in pm2.5**: 1043 (3.4%)
- **Duplicate rows**: 0
- **Constant features**: None

## Recommendations for Modeling
1. **Missing Value Treatment**: Use interpolation or forward-fill for time series data
2. **Feature Engineering**: Create lag features, rolling statistics, and time-based features
3. **Outlier Treatment**: Consider capping extreme values or using robust scaling
4. **Temporal Validation**: Use time-based splits for model validation
5. **Seasonality**: Account for monthly and hourly patterns in modeling
