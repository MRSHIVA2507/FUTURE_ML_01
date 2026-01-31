"""
Prepare Superstore data for sales forecasting.
This script aggregates the transactional Superstore data into daily sales totals.
"""

import pandas as pd
from datetime import datetime

# Read the Superstore dataset with proper encoding
print("Loading Superstore data...")
try:
    df = pd.read_csv('Sample - Superstore.csv', encoding='utf-8')
except UnicodeDecodeError:
    # Try with different encoding if utf-8 fails
    df = pd.read_csv('Sample - Superstore.csv', encoding='ISO-8859-1')

print(f"Original data: {len(df)} transactions")
print(f"Columns: {df.columns.tolist()}")
print(f"\nDate range: {df['Order Date'].min()} to {df['Order Date'].max()}")

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Aggregate sales by date
print("\nAggregating daily sales...")
daily_sales = df.groupby('Order Date').agg({
    'Sales': 'sum',
    'Quantity': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'  # Number of orders per day
}).reset_index()

# Rename columns for clarity
daily_sales.columns = ['date', 'sales', 'quantity', 'profit', 'num_orders']

# Sort by date
daily_sales = daily_sales.sort_values('date').reset_index(drop=True)

# Round to 2 decimal places
daily_sales['sales'] = daily_sales['sales'].round(2)
daily_sales['profit'] = daily_sales['profit'].round(2)

# Save to CSV
output_file = 'data/superstore_daily_sales.csv'
daily_sales.to_csv(output_file, index=False)

print(f"\n✅ Daily sales data created!")
print(f"Saved to: {output_file}")
print(f"\nAggregated data:")
print(f"- Total days: {len(daily_sales)}")
print(f"- Date range: {daily_sales['date'].min()} to {daily_sales['date'].max()}")
print(f"- Average daily sales: ${daily_sales['sales'].mean():.2f}")
print(f"- Total sales: ${daily_sales['sales'].sum():.2f}")
print(f"- Average orders per day: {daily_sales['num_orders'].mean():.1f}")

# Show preview
print(f"\nFirst 10 rows:")
print(daily_sales.head(10))

print(f"\n📊 You can now upload '{output_file}' to the dashboard!")
