import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import vaex


def resample_data_vaex(df, freq='D'):
    result = df.groupby(by=vaex.BinnerTime(df['open_time'], resolution=freq), agg={
        'open': vaex.agg.first('open'),
        'high': vaex.agg.max('high'),
        'low': vaex.agg.min('low'),
        'close': vaex.agg.last('close')
    })
    result.rename('open_time', 'date')
    return result


def resample_data(df, freq='H'):
    # Resample the data by freq
    result = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    return result

def preprocess_crypto_data(df):
    # Convert open_time to datetime format if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
        df['open_time'] = pd.to_datetime(df['open_time'])
    
    # Set the open_time column as the index
    df.set_index('open_time', inplace=True)
    
    # Convert all columns to numeric type in case they are not
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'quote_asset_volume', 'number_of_trades', 
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with any missing values
    df.dropna(inplace=True)
    
    # Optional: Create additional features like log returns, if needed for the analysis
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Sort the dataframe by index just in case
    df.sort_index(inplace=True)
    
    return df

def compute_rolling_statistics(df, window=30):
    # Assuming df_preprocessed has a 'return' column
    df['return'] = df['close'].pct_change()

    # Volatility (standard deviation of returns)
    df['volatility'] = df['return'].rolling(window=window).std()

    # Skewness and Kurtosis
    df['skewness'] = df['return'].rolling(window=window).apply(skew)
    df['kurtosis'] = df['return'].rolling(window=window).apply(kurtosis)

    # Autocorrelation
    df['autocorr_1'] = df['return'].rolling(window=window).apply(lambda x: x.autocorr(lag=1))
    df['autocorr_2'] = df['return'].rolling(window=window).apply(lambda x: x.autocorr(lag=2))
    df['autocorr_3'] = df['return'].rolling(window=window).apply(lambda x: x.autocorr(lag=3))

    df.head()

def display_eda_dashboard(df):
    # Print basic statistics
    print("Basic Statistics:")
    print(df.describe())
    
    # Plot time series of the prices
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 20), dpi=80)
    sns.lineplot(data=df['open'], ax=axes[0], color='blue', label='Open Price')
    sns.lineplot(data=df['close'], ax=axes[1], color='green', label='Close Price')
    sns.lineplot(data=df['high'], ax=axes[2], color='red', label='High Price')
    sns.lineplot(data=df['low'], ax=axes[3], color='orange', label='Low Price')
    for i, ax in enumerate(axes, start=1):
        ax.set_title(f'Price Time Series {["Open", "Close", "High", "Low"][i-1]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
    
    plt.tight_layout()
    plt.show()

    # Plot distribution of prices and volume
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), dpi=80)
    sns.histplot(df['open'], bins=50, kde=True, ax=axes[0, 0], color='blue')
    sns.histplot(df['volume'], bins=50, kde=True, ax=axes[0, 1], color='green')
    axes[0, 0].set_title('Distribution of Open Prices')
    axes[0, 1].set_title('Distribution of Volume')
    sns.boxplot(data=df[['open', 'high', 'low', 'close']], ax=axes[1, 0])
    sns.boxplot(data=df['volume'], ax=axes[1, 1])
    axes[1, 0].set_title('Boxplot of Prices')
    axes[1, 1].set_title('Boxplot of Volume')

    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8), dpi=80)
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
