"""
Data Handler Module
Handles data loading, validation, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import streamlit as st


class DataHandler:
    """Handle all data operations for the forecasting system."""
    
    def __init__(self):
        self.df = None
        self.date_column = None
        self.sales_column = None
    
    def load_data(self, uploaded_file) -> pd.DataFrame:
        """
        Load data from uploaded CSV or Excel file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas DataFrame
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                # Try UTF-8 first, then fall back to other encodings
                try:
                    df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    # Reset file pointer and try with ISO-8859-1 encoding
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Please upload a CSV or Excel file")
            
            self.df = df
            return df
        
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect the date column in the dataframe.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Name of the date column or None
        """
        # Check for common date column names
        date_keywords = ['date', 'time', 'day', 'timestamp', 'ds', 'datetime']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in date_keywords):
                return col
        
        # If not found by name, check data types
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
            
            # Try to parse as date
            try:
                pd.to_datetime(df[col].head(10), errors='coerce')
                if df[col].head(10).notna().sum() > 5:
                    return col
            except:
                continue
        
        return None
    
    def detect_sales_column(self, df: pd.DataFrame, exclude_date_col: str = None) -> Optional[str]:
        """
        Auto-detect the sales/demand column in the dataframe.
        
        Args:
            df: pandas DataFrame
            exclude_date_col: Date column to exclude from search
            
        Returns:
            Name of the sales column or None
        """
        # Check for common sales column names
        sales_keywords = ['sales', 'demand', 'quantity', 'revenue', 'amount', 
                         'qty', 'volume', 'units', 'count', 'y']
        
        for col in df.columns:
            if col == exclude_date_col:
                continue
            
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in sales_keywords):
                if pd.api.types.is_numeric_dtype(df[col]):
                    return col
        
        # If not found by name, return first numeric column
        for col in df.columns:
            if col == exclude_date_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
        
        return None
    
    def clean_data(self, df: pd.DataFrame, date_col: str, sales_col: str, 
                   strategy: str = 'forward_fill') -> Tuple[pd.DataFrame, Dict]:
        """
        Clean the dataframe by handling missing values, duplicates, and sorting.
        
        Args:
            df: pandas DataFrame
            date_col: Name of date column
            sales_col: Name of sales column
            strategy: Missing value strategy ('forward_fill', 'mean', 'median', 'drop')
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning report dict)
        """
        report = {
            'original_rows': len(df),
            'missing_dates': df[date_col].isna().sum(),
            'missing_sales': df[sales_col].isna().sum(),
            'duplicates': df.duplicated().sum(),
            'strategy_used': strategy,
            'rows_removed': 0
        }
        
        # Create a copy
        df_clean = df.copy()
        
        # Convert date column to datetime
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        
        # Remove rows with invalid dates
        df_clean = df_clean[df_clean[date_col].notna()]
        
        # Sort by date
        df_clean = df_clean.sort_values(by=date_col).reset_index(drop=True)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=[date_col], keep='first')
        
        # Handle missing sales values
        if strategy == 'forward_fill':
            df_clean[sales_col] = df_clean[sales_col].fillna(method='ffill')
            df_clean[sales_col] = df_clean[sales_col].fillna(method='bfill')
        elif strategy == 'mean':
            mean_value = df_clean[sales_col].mean()
            df_clean[sales_col] = df_clean[sales_col].fillna(mean_value)
        elif strategy == 'median':
            median_value = df_clean[sales_col].median()
            df_clean[sales_col] = df_clean[sales_col].fillna(median_value)
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[sales_col])
        
        # Remove any remaining NaN values
        df_clean = df_clean.dropna(subset=[date_col, sales_col])
        
        # Ensure sales values are positive
        df_clean = df_clean[df_clean[sales_col] >= 0]
        
        report['rows_removed'] = report['original_rows'] - len(df_clean)
        report['final_rows'] = len(df_clean)
        
        return df_clean, report
    
    def generate_data_report(self, df: pd.DataFrame, date_col: str, sales_col: str) -> Dict:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: pandas DataFrame
            date_col: Name of date column
            sales_col: Name of sales column
            
        Returns:
            Dictionary with data statistics
        """
        report = {
            'total_records': len(df),
            'date_range': {
                'start': df[date_col].min(),
                'end': df[date_col].max(),
                'total_days': (df[date_col].max() - df[date_col].min()).days
            },
            'sales_stats': {
                'mean': df[sales_col].mean(),
                'median': df[sales_col].median(),
                'min': df[sales_col].min(),
                'max': df[sales_col].max(),
                'std': df[sales_col].std(),
                'total': df[sales_col].sum()
            },
            'data_quality': {
                'missing_values': df.isna().sum().sum(),
                'duplicates': df.duplicated().sum(),
                'data_quality_score': self._calculate_quality_score(df)
            }
        }
        
        return report
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate a data quality score (0-100).
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Quality score as float
        """
        score = 100.0
        
        # Deduct for missing values
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        score -= missing_pct * 2
        
        # Deduct for duplicates
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        score -= duplicate_pct * 2
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        return round(score, 1)
    
    def explain_cleaning_strategy(self, strategy: str) -> str:
        """
        Provide business-friendly explanation of cleaning strategy.
        
        Args:
            strategy: The cleaning strategy used
            
        Returns:
            Human-readable explanation
        """
        explanations = {
            'forward_fill': """
                **Forward Fill Strategy**: Missing sales values are filled using the most recent known value.
                This works well for time-series data where values don't change drastically day-to-day.
                """,
            'mean': """
                **Mean Strategy**: Missing sales values are replaced with the average of all sales.
                This is a safe approach that maintains the overall trend without introducing extreme values.
                """,
            'median': """
                **Median Strategy**: Missing sales values are replaced with the middle value of all sales.
                This is robust to outliers and works well when your data has occasional spikes or drops.
                """,
            'drop': """
                **Drop Strategy**: Rows with missing sales values are removed entirely.
                Use this when you have plenty of data and want to ensure 100% accuracy in available records.
                """
        }
        
        return explanations.get(strategy, "Strategy explanation not available")
