"""
Generate sample sales dataset for testing the forecasting system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2 years of daily data
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Base sales level
base_sales = 100

# Create trend component (15% annual growth)
trend = np.linspace(0, base_sales * 0.15 * 2, len(dates))

# Create weekly seasonality (weekends 40% higher)
weekly_seasonality = []
for date in dates:
    if date.dayofweek in [5, 6]:  # Saturday, Sunday
        weekly_seasonality.append(base_sales * 0.4)
    else:
        weekly_seasonality.append(0)
weekly_seasonality = np.array(weekly_seasonality)

# Create monthly seasonality (end of month peaks)
monthly_seasonality = []
for date in dates:
    if date.day > 25:  # End of month
        monthly_seasonality.append(base_sales * 0.3)
    elif date.day < 5:  # Start of month
        monthly_seasonality.append(base_sales * 0.15)
    else:
        monthly_seasonality.append(0)
monthly_seasonality = np.array(monthly_seasonality)

# Add some random noise
noise = np.random.normal(0, base_sales * 0.1, len(dates))

# Combine all components
sales = base_sales + trend + weekly_seasonality + monthly_seasonality + noise

# Ensure no negative sales
sales = np.maximum(sales, 10)

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'sales': sales.round(0).astype(int)
})

# Add some missing values randomly (2% of data)
missing_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
df.loc[missing_indices, 'sales'] = np.nan

# Save to CSV
df.to_csv('sample_sales.csv', index=False)

print(f"Sample dataset created successfully!")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Total records: {len(df)}")
print(f"Missing values: {df['sales'].isna().sum()}")
print(f"Average sales: {df['sales'].mean():.0f}")
print(f"Saved to: sample_sales.csv")
