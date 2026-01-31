"""
Feature Engineer Module
Extracts time-based features and creates lag/rolling features for forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats


class FeatureEngineer:
    """Extract and create features for time-series forecasting."""
    
    def __init__(self):
        self.trend_info = {}
        self.seasonality_info = {}
    
    def extract_time_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Extract temporal features from date column.
        
        Args:
            df: pandas DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with additional time features
        """
        df_features = df.copy()
        
        # Ensure date column is datetime
        df_features[date_col] = pd.to_datetime(df_features[date_col])
        
        # Extract features
        df_features['year'] = df_features[date_col].dt.year
        df_features['month'] = df_features[date_col].dt.month
        df_features['week'] = df_features[date_col].dt.isocalendar().week
        df_features['day'] = df_features[date_col].dt.day
        df_features['day_of_week'] = df_features[date_col].dt.dayofweek
        df_features['day_name'] = df_features[date_col].dt.day_name()
        df_features['month_name'] = df_features[date_col].dt.month_name()
        df_features['quarter'] = df_features[date_col].dt.quarter
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Day of month features
        df_features['is_month_start'] = df_features[date_col].dt.is_month_start.astype(int)
        df_features['is_month_end'] = df_features[date_col].dt.is_month_end.astype(int)
        
        return df_features
    
    def create_lag_features(self, df: pd.DataFrame, sales_col: str, 
                           lags: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """
        Create lag features for time-series prediction.
        
        Args:
            df: pandas DataFrame
            sales_col: Name of sales column
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lag features
        """
        df_lags = df.copy()
        
        for lag in lags:
            df_lags[f'lag_{lag}'] = df_lags[sales_col].shift(lag)
        
        return df_lags
    
    def create_rolling_features(self, df: pd.DataFrame, sales_col: str,
                               windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: pandas DataFrame
            sales_col: Name of sales column
            windows: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with rolling features
        """
        df_rolling = df.copy()
        
        for window in windows:
            # Rolling mean
            df_rolling[f'rolling_mean_{window}'] = df_rolling[sales_col].rolling(
                window=window, min_periods=1
            ).mean()
            
            # Rolling std
            df_rolling[f'rolling_std_{window}'] = df_rolling[sales_col].rolling(
                window=window, min_periods=1
            ).std()
            
            # Rolling min/max
            df_rolling[f'rolling_min_{window}'] = df_rolling[sales_col].rolling(
                window=window, min_periods=1
            ).min()
            df_rolling[f'rolling_max_{window}'] = df_rolling[sales_col].rolling(
                window=window, min_periods=1
            ).max()
        
        return df_rolling
    
    def detect_trend(self, df: pd.DataFrame, date_col: str, sales_col: str) -> Dict:
        """
        Detect and quantify trend in the data.
        
        Args:
            df: pandas DataFrame
            date_col: Name of date column
            sales_col: Name of sales column
            
        Returns:
            Dictionary with trend information
        """
        # Create numeric time variable
        df_trend = df.copy()
        df_trend['time_numeric'] = (df_trend[date_col] - df_trend[date_col].min()).dt.days
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_trend['time_numeric'], df_trend[sales_col]
        )
        
        # Calculate trend metrics
        initial_value = df_trend[sales_col].iloc[:30].mean()
        final_value = df_trend[sales_col].iloc[-30:].mean()
        
        if initial_value > 0:
            percent_change = ((final_value - initial_value) / initial_value) * 100
        else:
            percent_change = 0
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Determine trend strength
        if abs(r_value) > 0.7:
            strength = "strong"
        elif abs(r_value) > 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        
        trend_info = {
            'direction': direction,
            'strength': strength,
            'slope': slope,
            'r_squared': r_value ** 2,
            'percent_change': percent_change,
            'initial_avg': initial_value,
            'final_avg': final_value
        }
        
        self.trend_info = trend_info
        return trend_info
    
    def detect_seasonality(self, df: pd.DataFrame, sales_col: str) -> Dict:
        """
        Detect seasonality patterns in the data.
        
        Args:
            df: pandas DataFrame with time features already extracted
            sales_col: Name of sales column
            
        Returns:
            Dictionary with seasonality information
        """
        seasonality_info = {}
        
        # Check for day-of-week seasonality
        if 'day_of_week' in df.columns:
            dow_avg = df.groupby('day_of_week')[sales_col].mean()
            dow_std = df.groupby('day_of_week')[sales_col].std()
            
            # Calculate coefficient of variation
            dow_cv = (dow_std / dow_avg).mean()
            
            seasonality_info['day_of_week'] = {
                'pattern': dow_avg.to_dict(),
                'strength': 'strong' if dow_cv > 0.3 else 'moderate' if dow_cv > 0.15 else 'weak',
                'peak_day': dow_avg.idxmax(),
                'low_day': dow_avg.idxmin(),
                'variation': dow_cv
            }
        
        # Check for monthly seasonality
        if 'month' in df.columns and df['month'].nunique() >= 6:
            month_avg = df.groupby('month')[sales_col].mean()
            month_std = df.groupby('month')[sales_col].std()
            
            month_cv = (month_std / month_avg).mean()
            
            seasonality_info['monthly'] = {
                'pattern': month_avg.to_dict(),
                'strength': 'strong' if month_cv > 0.3 else 'moderate' if month_cv > 0.15 else 'weak',
                'peak_month': month_avg.idxmax(),
                'low_month': month_avg.idxmin(),
                'variation': month_cv
            }
        
        # Check for weekend effect
        if 'is_weekend' in df.columns:
            weekend_avg = df[df['is_weekend'] == 1][sales_col].mean()
            weekday_avg = df[df['is_weekend'] == 0][sales_col].mean()
            
            if weekday_avg > 0:
                weekend_diff = ((weekend_avg - weekday_avg) / weekday_avg) * 100
            else:
                weekend_diff = 0
            
            seasonality_info['weekend_effect'] = {
                'weekend_avg': weekend_avg,
                'weekday_avg': weekday_avg,
                'percent_difference': weekend_diff,
                'significance': 'high' if abs(weekend_diff) > 20 else 'moderate' if abs(weekend_diff) > 10 else 'low'
            }
        
        self.seasonality_info = seasonality_info
        return seasonality_info
    
    def get_trend_explanation(self) -> str:
        """
        Generate business-friendly explanation of detected trends.
        
        Returns:
            Human-readable trend explanation
        """
        if not self.trend_info:
            return "Trend analysis not yet performed."
        
        direction = self.trend_info['direction']
        strength = self.trend_info['strength']
        percent_change = self.trend_info['percent_change']
        
        if direction == "increasing":
            explanation = f"""
            📈 **Growing Sales**: Your sales show a **{strength} upward trend**.
            
            - Sales have increased by **{abs(percent_change):.1f}%** over the analyzed period
            - Average sales grew from **{self.trend_info['initial_avg']:.0f}** to **{self.trend_info['final_avg']:.0f}** units
            - This indicates positive business momentum
            """
        elif direction == "decreasing":
            explanation = f"""
            📉 **Declining Sales**: Your sales show a **{strength} downward trend**.
            
            - Sales have decreased by **{abs(percent_change):.1f}%** over the analyzed period
            - Average sales dropped from **{self.trend_info['initial_avg']:.0f}** to **{self.trend_info['final_avg']:.0f}** units
            - Consider investigating factors affecting sales performance
            """
        else:
            explanation = f"""
            ➡️ **Stable Sales**: Your sales are relatively **stable** with no strong trend.
            
            - Sales fluctuate around **{self.trend_info['final_avg']:.0f}** units
            - Variation is within **±{abs(percent_change):.1f}%**
            - Predictable pattern is good for inventory planning
            """
        
        return explanation.strip()
    
    def get_seasonality_explanation(self) -> str:
        """
        Generate business-friendly explanation of detected seasonality.
        
        Returns:
            Human-readable seasonality explanation
        """
        if not self.seasonality_info:
            return "Seasonality analysis not yet performed."
        
        explanations = []
        
        # Day of week seasonality
        if 'day_of_week' in self.seasonality_info:
            dow_info = self.seasonality_info['day_of_week']
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            peak_day_name = day_names[dow_info['peak_day']]
            low_day_name = day_names[dow_info['low_day']]
            
            explanations.append(f"""
            📅 **Weekly Pattern** ({dow_info['strength']} seasonality):
            - **Highest sales**: {peak_day_name}
            - **Lowest sales**: {low_day_name}
            - Plan staffing and inventory accordingly
            """)
        
        # Weekend effect
        if 'weekend_effect' in self.seasonality_info:
            we_info = self.seasonality_info['weekend_effect']
            diff = we_info['percent_difference']
            
            if abs(diff) > 10:
                if diff > 0:
                    explanations.append(f"""
                    🎉 **Weekend Boost**: Weekend sales are **{abs(diff):.1f}% higher** than weekdays
                    - Weekend average: {we_info['weekend_avg']:.0f} units
                    - Weekday average: {we_info['weekday_avg']:.0f} units
                    - Consider weekend promotions and adequate staffing
                    """)
                else:
                    explanations.append(f"""
                    💼 **Weekday Focus**: Weekday sales are **{abs(diff):.1f}% higher** than weekends
                    - Weekday average: {we_info['weekday_avg']:.0f} units
                    - Weekend average: {we_info['weekend_avg']:.0f} units
                    - Business-driven demand pattern
                    """)
        
        # Monthly seasonality
        if 'monthly' in self.seasonality_info:
            month_info = self.seasonality_info['monthly']
            month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            peak_month = month_names[month_info['peak_month']]
            low_month = month_names[month_info['low_month']]
            
            explanations.append(f"""
            📆 **Monthly Pattern** ({month_info['strength']} seasonality):
            - **Peak month**: {peak_month}
            - **Slowest month**: {low_month}
            - Plan inventory cycles around these patterns
            """)
        
        if explanations:
            return "\n\n".join(explanations).strip()
        else:
            return "No significant seasonality patterns detected."
