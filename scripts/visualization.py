"""
Visualization utilities for air quality forecasting project.
Creates comprehensive plots for data exploration and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_pm25_trends(df, target_col='pm2.5', save_path=None):
    """
    Plot PM2.5 trends over time with seasonal patterns.
    
    Args:
        df (pd.DataFrame): Input dataframe with datetime index
        target_col (str): Target column name
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PM2.5 Concentration Analysis', fontsize=16, fontweight='bold')
    
    # Time series plot
    axes[0, 0].plot(df.index, df[target_col], alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('PM2.5 Time Series')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('PM2.5 (μg/m³)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Monthly averages
    monthly_avg = df.groupby(df.index.month)[target_col].mean()
    axes[0, 1].bar(monthly_avg.index, monthly_avg.values, color='skyblue', alpha=0.8)
    axes[0, 1].set_title('Average PM2.5 by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Average PM2.5 (μg/m³)')
    axes[0, 1].set_xticks(range(1, 13))
    
    # Hourly patterns
    hourly_avg = df.groupby(df.index.hour)[target_col].mean()
    axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=4)
    axes[1, 0].set_title('Average PM2.5 by Hour of Day')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Average PM2.5 (μg/m³)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(0, 24, 4))
    
    # Day of week patterns
    dow_avg = df.groupby(df.index.dayofweek)[target_col].mean()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 1].bar(range(7), dow_avg.values, color='lightcoral', alpha=0.8)
    axes[1, 1].set_title('Average PM2.5 by Day of Week')
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Average PM2.5 (μg/m³)')
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(dow_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PM2.5 trends plot saved to {save_path}")
    
    plt.show()

def plot_feature_distributions(df, save_path=None):
    """
    Plot distributions of all numeric features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_path (str): Path to save the plot
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'No']
    
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
            axes[i].legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature distributions plot saved to {save_path}")
    
    plt.show()

def plot_correlation_heatmap(df, target_col='pm2.5', save_path=None):
    """
    Plot correlation heatmap focusing on relationships with target variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        save_path (str): Path to save the plot
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'No']
    
    # Calculate correlations
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Full correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax1, fmt='.2f')
    ax1.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Target correlation bar plot
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        colors = ['red' if x < 0 else 'blue' for x in target_corr.values]
        
        bars = ax2.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(target_corr)))
        ax2.set_yticklabels(target_corr.index, fontsize=10)
        ax2.set_xlabel('Correlation with PM2.5')
        ax2.set_title(f'Feature Correlations with {target_col}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=0, color='black', linewidth=0.8)
        
        # Add correlation values on bars
        for i, (bar, val) in enumerate(zip(bars, target_corr.values)):
            ax2.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                    va='center', ha='left' if val >= 0 else 'right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {save_path}")
    
    plt.show()
    
    return corr_matrix

def plot_missing_values(df, save_path=None):
    """
    Visualize missing values pattern in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_path (str): Path to save the plot
    """
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) == 0:
        print("No missing values found in the dataset.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Missing values count
    bars = ax1.bar(range(len(missing_data)), missing_data.values, color='salmon', alpha=0.8)
    ax1.set_title('Missing Values Count by Feature', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Number of Missing Values')
    ax1.set_xticks(range(len(missing_data)))
    ax1.set_xticklabels(missing_data.index, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, missing_data.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(missing_data.values),
                f'{val}', ha='center', va='bottom', fontsize=10)
    
    # Missing values percentage
    missing_pct = (missing_data / len(df)) * 100
    bars2 = ax2.bar(range(len(missing_pct)), missing_pct.values, color='lightcoral', alpha=0.8)
    ax2.set_title('Missing Values Percentage by Feature', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Percentage of Missing Values (%)')
    ax2.set_xticks(range(len(missing_pct)))
    ax2.set_xticklabels(missing_pct.index, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, val in zip(bars2, missing_pct.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(missing_pct.values),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Missing values plot saved to {save_path}")
    
    plt.show()

def plot_weather_relationships(df, target_col='pm2.5', save_path=None):
    """
    Plot relationships between weather variables and PM2.5.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        save_path (str): Path to save the plot
    """
    weather_cols = ['TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir']
    available_cols = [col for col in weather_cols if col in df.columns]
    
    if not available_cols or target_col not in df.columns:
        print("Required weather columns or target column not found.")
        return
    
    n_cols = 3
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'Weather Variables vs {target_col}', fontsize=16, fontweight='bold')
    
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(available_cols):
        if i < len(axes):
            # Scatter plot with trend line
            axes[i].scatter(df[col], df[target_col], alpha=0.5, s=1)
            
            # Add trend line
            z = np.polyfit(df[col].dropna(), df[target_col][df[col].notna()], 1)
            p = np.poly1d(z)
            axes[i].plot(df[col], p(df[col]), "r--", alpha=0.8, linewidth=2)
            
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(target_col)
            axes[i].set_title(f'{col} vs {target_col}')
            axes[i].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = df[col].corr(df[target_col])
            axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
    
    # Hide empty subplots
    for i in range(len(available_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weather relationships plot saved to {save_path}")
    
    plt.show()

def plot_seasonal_patterns(df, target_col='pm2.5', save_path=None):
    """
    Plot seasonal patterns and pollution episodes.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        save_path (str): Path to save the plot
    """
    if target_col not in df.columns:
        print(f"Target column {target_col} not found.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Seasonal Patterns and Pollution Episodes', fontsize=16, fontweight='bold')
    
    # Seasonal boxplot
    df_copy = df.copy()
    df_copy['season'] = df_copy.index.month.map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                                3: 'Spring', 4: 'Spring', 5: 'Spring',
                                                6: 'Summer', 7: 'Summer', 8: 'Summer',
                                                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
    
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
    sns.boxplot(data=df_copy, x='season', y=target_col, order=season_order, ax=axes[0, 0])
    axes[0, 0].set_title('PM2.5 Distribution by Season')
    axes[0, 0].set_ylabel('PM2.5 (μg/m³)')
    
    # Monthly trend with error bars
    monthly_stats = df.groupby(df.index.month)[target_col].agg(['mean', 'std'])
    axes[0, 1].errorbar(monthly_stats.index, monthly_stats['mean'], 
                       yerr=monthly_stats['std'], marker='o', capsize=5, capthick=2)
    axes[0, 1].set_title('Monthly PM2.5 Trends with Standard Deviation')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('PM2.5 (μg/m³)')
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pollution episodes (high PM2.5 days)
    threshold = df[target_col].quantile(0.9)  # 90th percentile
    high_pollution = df[df[target_col] > threshold]
    
    if len(high_pollution) > 0:
        episode_months = high_pollution.groupby(high_pollution.index.month).size()
        axes[1, 0].bar(episode_months.index, episode_months.values, color='red', alpha=0.7)
        axes[1, 0].set_title(f'High Pollution Episodes by Month (PM2.5 > {threshold:.1f})')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Number of Episodes')
        axes[1, 0].set_xticks(range(1, 13))
    
    # Yearly trend (if multiple years available)
    if df.index.year.nunique() > 1:
        yearly_avg = df.groupby(df.index.year)[target_col].mean()
        axes[1, 1].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_title('Annual Average PM2.5 Trend')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Average PM2.5 (μg/m³)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor yearly trend', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[1, 1].set_title('Annual Trend (Insufficient Data)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Seasonal patterns plot saved to {save_path}")
    
    plt.show()

def create_interactive_dashboard(df, target_col='pm2.5', save_path=None):
    """
    Create an interactive dashboard using Plotly.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        save_path (str): Path to save the HTML file
    """
    if target_col not in df.columns:
        print(f"Target column {target_col} not found.")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PM2.5 Time Series', 'Hourly Patterns', 
                       'Monthly Distribution', 'Weather Correlation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Time series
    fig.add_trace(
        go.Scatter(x=df.index, y=df[target_col], mode='lines', name='PM2.5',
                  line=dict(width=1), opacity=0.7),
        row=1, col=1
    )
    
    # Hourly patterns
    hourly_avg = df.groupby(df.index.hour)[target_col].mean()
    fig.add_trace(
        go.Scatter(x=hourly_avg.index, y=hourly_avg.values, mode='lines+markers',
                  name='Hourly Average', line=dict(width=3)),
        row=1, col=2
    )
    
    # Monthly box plot
    months = df.index.month
    fig.add_trace(
        go.Box(x=months, y=df[target_col], name='Monthly Distribution'),
        row=2, col=1
    )
    
    # Weather correlation (example with temperature)
    if 'TEMP' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['TEMP'], y=df[target_col], mode='markers',
                      name='PM2.5 vs Temperature', opacity=0.6),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Air Quality Interactive Dashboard",
        title_x=0.5,
        height=800,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=1, col=2)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_xaxes(title_text="Temperature", row=2, col=2)
    
    fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=1, col=1)
    fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=1, col=2)
    fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=2, col=1)
    fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=2, col=2)
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    
    fig.show()

def generate_data_summary_report(df, target_col='pm2.5', save_path=None):
    """
    Generate a comprehensive data summary report.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        save_path (str): Path to save the report
    """
    report = []
    report.append("# Air Quality Dataset Summary Report\n")
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Basic information
    report.append("## Dataset Overview\n")
    report.append(f"- **Shape**: {df.shape[0]} rows × {df.shape[1]} columns\n")
    report.append(f"- **Date Range**: {df.index.min()} to {df.index.max()}\n")
    report.append(f"- **Duration**: {(df.index.max() - df.index.min()).days} days\n")
    report.append(f"- **Frequency**: {pd.infer_freq(df.index) or 'Irregular'}\n\n")
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    report.append("## Missing Values\n")
    if len(missing_data) > 0:
        for col, count in missing_data.items():
            pct = (count / len(df)) * 100
            report.append(f"- **{col}**: {count} ({pct:.1f}%)\n")
    else:
        report.append("- No missing values found\n")
    report.append("\n")
    
    # Target variable statistics
    if target_col in df.columns:
        report.append(f"## {target_col} Statistics\n")
        stats = df[target_col].describe()
        report.append(f"- **Mean**: {stats['mean']:.2f} μg/m³\n")
        report.append(f"- **Median**: {stats['50%']:.2f} μg/m³\n")
        report.append(f"- **Standard Deviation**: {stats['std']:.2f} μg/m³\n")
        report.append(f"- **Range**: {stats['min']:.2f} - {stats['max']:.2f} μg/m³\n")
        report.append(f"- **IQR**: {stats['25%']:.2f} - {stats['75%']:.2f} μg/m³\n\n")
        
        # Air quality categories (WHO guidelines)
        good = (df[target_col] <= 15).sum()
        moderate = ((df[target_col] > 15) & (df[target_col] <= 35)).sum()
        unhealthy = ((df[target_col] > 35) & (df[target_col] <= 75)).sum()
        very_unhealthy = (df[target_col] > 75).sum()
        
        total = len(df[target_col].dropna())
        report.append("### Air Quality Categories (WHO Guidelines)\n")
        report.append(f"- **Good (≤15)**: {good} ({good/total*100:.1f}%)\n")
        report.append(f"- **Moderate (15-35)**: {moderate} ({moderate/total*100:.1f}%)\n")
        report.append(f"- **Unhealthy (35-75)**: {unhealthy} ({unhealthy/total*100:.1f}%)\n")
        report.append(f"- **Very Unhealthy (>75)**: {very_unhealthy} ({very_unhealthy/total*100:.1f}%)\n\n")
    
    # Feature correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target_col in numeric_cols and len(numeric_cols) > 1:
        corr_with_target = df[numeric_cols].corr()[target_col].drop(target_col)
        corr_with_target = corr_with_target.sort_values(key=abs, ascending=False)
        
        report.append(f"## Feature Correlations with {target_col}\n")
        for feature, corr in corr_with_target.head(10).items():
            report.append(f"- **{feature}**: {corr:.3f}\n")
        report.append("\n")
    
    # Seasonal patterns
    if target_col in df.columns:
        seasonal_avg = df.groupby(df.index.month)[target_col].mean()
        report.append("## Seasonal Patterns (Monthly Averages)\n")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month_num, avg in seasonal_avg.items():
            report.append(f"- **{months[month_num-1]}**: {avg:.2f} μg/m³\n")
        report.append("\n")
    
    # Data quality assessment
    report.append("## Data Quality Assessment\n")
    
    # Check for outliers using IQR method
    if target_col in df.columns:
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[target_col] < (Q1 - 1.5 * IQR)) | 
                   (df[target_col] > (Q3 + 1.5 * IQR))).sum()
        report.append(f"- **Outliers in {target_col}**: {outliers} ({outliers/len(df)*100:.1f}%)\n")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    report.append(f"- **Duplicate rows**: {duplicates}\n")
    
    # Check for constant features
    constant_features = []
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        report.append(f"- **Constant features**: {', '.join(constant_features)}\n")
    else:
        report.append("- **Constant features**: None\n")
    
    report.append("\n")
    
    # Recommendations
    report.append("## Recommendations for Modeling\n")
    report.append("1. **Missing Value Treatment**: ")
    if len(missing_data) > 0:
        report.append("Use interpolation or forward-fill for time series data\n")
    else:
        report.append("No missing values to handle\n")
    
    report.append("2. **Feature Engineering**: Create lag features, rolling statistics, and time-based features\n")
    report.append("3. **Outlier Treatment**: Consider capping extreme values or using robust scaling\n")
    report.append("4. **Temporal Validation**: Use time-based splits for model validation\n")
    report.append("5. **Seasonality**: Account for monthly and hourly patterns in modeling\n")
    
    report_text = ''.join(report)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Data summary report saved to {save_path}")
    
    return report_text

def create_comprehensive_eda(df, target_col='pm2.5', output_dir='visuals/'):
    """
    Create comprehensive exploratory data analysis with all visualizations.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        output_dir (str): Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating comprehensive EDA visualizations...")
    
    # 1. PM2.5 trends
    plot_pm25_trends(df, target_col, save_path=f'{output_dir}/pm25_trends.png')
    
    # 2. Feature distributions
    plot_feature_distributions(df, save_path=f'{output_dir}/feature_distributions.png')
    
    # 3. Correlation analysis
    corr_matrix = plot_correlation_heatmap(df, target_col, save_path=f'{output_dir}/correlation_heatmap.png')
    
    # 4. Missing values
    plot_missing_values(df, save_path=f'{output_dir}/missing_values.png')
    
    # 5. Weather relationships
    plot_weather_relationships(df, target_col, save_path=f'{output_dir}/weather_relationships.png')
    
    # 6. Seasonal patterns
    plot_seasonal_patterns(df, target_col, save_path=f'{output_dir}/seasonal_patterns.png')
    
    # 7. Interactive dashboard
    create_interactive_dashboard(df, target_col, save_path=f'{output_dir}/interactive_dashboard.html')
    
    # 8. Summary report
    report = generate_data_summary_report(df, target_col, save_path=f'{output_dir}/data_summary_report.md')
    
    print(f"EDA complete! All visualizations saved to {output_dir}")
    
    return {
        'correlation_matrix': corr_matrix,
        'summary_report': report
    }
