"""
Forecaster Module
Implements Prophet and SARIMA models for sales forecasting.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Import statsmodels for SARIMA
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class SalesForecaster:
    """Train and use forecasting models (Prophet, SARIMA, or Baseline)."""
    
    def __init__(self):
        self.prophet_model = None
        self.sarima_model = None
        self.baseline_model = None
        self.scaler = None
        self.feature_cols = []
        self.model_type = None  # Track which model was successfully trained
    
    def prepare_prophet_data(self, df: pd.DataFrame, date_col: str, 
                            sales_col: str) -> pd.DataFrame:
        """
        Prepare data in Prophet's required format (ds, y).
        
        Args:
            df: pandas DataFrame
            date_col: Name of date column
            sales_col: Name of sales column
            
        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[sales_col]
        })
        
        return prophet_df
    
    def train_sarima_model(self, df: pd.DataFrame, date_col: str, 
                          sales_col: str, order=(1,1,1), seasonal_order=(1,1,1,7)):
        """
        Train SARIMA model using statsmodels.
        
        Args:
            df: pandas DataFrame
            date_col: Name of date column
            sales_col: Name of sales column
            order: ARIMA order (p,d,q)
            seasonal_order: Seasonal order (P,D,Q,s)
            
        Returns:
            Trained SARIMA model
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels not installed. Run: pip install statsmodels")
        
        # Prepare data
        df_sarima = df[[date_col, sales_col]].copy()
        df_sarima[date_col] = pd.to_datetime(df_sarima[date_col])
        df_sarima = df_sarima.sort_values(date_col).set_index(date_col)
        df_sarima = df_sarima.asfreq('D', fill_value=0)  # Daily frequency
        
        # Auto-determine seasonal period
        n_days = len(df_sarima)
        if n_days < 14:
            seasonal_order = (0,0,0,0)  # No seasonality for short series
        elif n_days < 90:
            seasonal_order = (1,1,1,7)  # Weekly seasonality
        else:
            seasonal_order = (1,1,1,7)  # Weekly seasonality
        
        # Train SARIMA
        try:
            model = SARIMAX(
                df_sarima[sales_col],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.sarima_model = model.fit(disp=False, maxiter=200)
            self.model_type = 'sarima'
            return self.sarima_model
            
        except Exception as e:
            raise Exception(f"SARIMA training failed: {str(e)}")
    
    def predict_sarima(self, periods: int = 30) -> pd.DataFrame:
        """
        Generate future predictions using SARIMA.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if self.sarima_model is None:
            raise ValueError("SARIMA model not trained yet")
        
        # Get forecast
        forecast_result = self.sarima_model.get_forecast(steps=periods)
        forecast_df = forecast_result.summary_frame(alpha=0.05)
        
        # Rename columns to match Prophet format
        forecast_df_formatted = pd.DataFrame({
            'ds': forecast_df.index,
            'yhat': forecast_df['mean'],
            'yhat_lower': forecast_df['mean_ci_lower'],
            'yhat_upper': forecast_df['mean_ci_upper']
        })
        
        # Ensure non-negative forecasts
        forecast_df_formatted['yhat'] = forecast_df_formatted['yhat'].clip(lower=0)
        forecast_df_formatted['yhat_lower'] = forecast_df_formatted['yhat_lower'].clip(lower=0)
        forecast_df_formatted['yhat_upper'] = forecast_df_formatted['yhat_upper'].clip(lower=0)
        
        return forecast_df_formatted
    
    def train_prophet_model(self, df: pd.DataFrame, date_col: str, 
                           sales_col: str, **kwargs):
        """
        Train Facebook Prophet model with fallback to SARIMA if Prophet fails.
        
        Args:
            df: pandas DataFrame
            date_col: Name of date column
            sales_col: Name of sales column
            **kwargs: Additional Prophet parameters
            
        Returns:
            Trained model (Prophet or SARIMA)
        """
        # Try Prophet first
        try:
            # Prepare data
            prophet_df = self.prepare_prophet_data(df, date_col, sales_col)
            
            # Determine seasonality based on data size
            n_days = (prophet_df['ds'].max() - prophet_df['ds'].min()).days
            
            # Configure Prophet
            model = Prophet(
                daily_seasonality=True if n_days > 14 else False,
                weekly_seasonality=True if n_days > 21 else False,
                yearly_seasonality=True if n_days > 730 else False,
                seasonality_mode='multiplicative',
                interval_width=0.95,
                changepoint_prior_scale=0.05,
                **kwargs
            )
            
            # Train model
            model.fit(prophet_df)
            
            self.prophet_model = model
            self.model_type = 'prophet'
            return model
            
        except Exception as prophet_error:
            # Prophet failed, try SARIMA as fallback
            warnings.warn(f"Prophet failed: {str(prophet_error)}. Falling back to SARIMA model...")
            
            try:
                return self.train_sarima_model(df, date_col, sales_col)
            except Exception as sarima_error:
                raise Exception(
                    f"Both Prophet and SARIMA failed.\n"
                    f"Prophet error: {str(prophet_error)}\n"
                    f"SARIMA error: {str(sarima_error)}\n"
                    f"Try: pip install statsmodels"
                )
    
    def predict_prophet(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """
        Generate future predictions using the trained model (Prophet or SARIMA).
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if self.model_type == 'prophet':
            if self.prophet_model is None:
                raise ValueError("Prophet model not trained yet")
            
            # Create future dataframe
            future = self.prophet_model.make_future_dataframe(periods=periods, freq=freq)
            
            # Predict
            forecast = self.prophet_model.predict(future)
            
            return forecast
            
        elif self.model_type == 'sarima':
            return self.predict_sarima(periods)
            
        else:
            raise ValueError("No forecasting model trained")
    
    def train_baseline_model(self, df: pd.DataFrame, date_col: str, 
                            sales_col: str) -> Tuple[LinearRegression, StandardScaler]:
        """
        Train simple linear regression baseline model.
        
        Args:
            df: pandas DataFrame with time features
            date_col: Name of date column
            sales_col: Name of sales column
            
        Returns:
            Tuple of (trained model, scaler)
        """
        df_train = df.copy()
        
        # Create time-based features
        df_train['time_numeric'] = (pd.to_datetime(df_train[date_col]) - 
                                     pd.to_datetime(df_train[date_col]).min()).dt.days
        
        # Select features
        feature_cols = ['time_numeric']
        
        # Add time features if available
        for col in ['month', 'day_of_week', 'is_weekend']:
            if col in df_train.columns:
                feature_cols.append(col)
        
        # Add lag features if available
        for col in df_train.columns:
            if 'lag_' in col or 'rolling_mean_' in col:
                feature_cols.append(col)
        
        self.feature_cols = feature_cols
        
        # Prepare data
        X = df_train[feature_cols].fillna(method='bfill').fillna(method='ffill')
        y = df_train[sales_col]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        self.baseline_model = model
        self.scaler = scaler
        
        return model, scaler
    
    def predict_baseline(self, df: pd.DataFrame, periods: int = 30) -> np.ndarray:
        """
        Generate predictions using baseline model.
        
        Args:
            df: pandas DataFrame with features
            periods: Number of periods to forecast
            
        Returns:
            Array of predictions
        """
        if self.baseline_model is None or self.scaler is None:
            raise ValueError("Baseline model not trained yet")
        
        # Get last known values
        last_date = pd.to_datetime(df.iloc[-1]['date'] if 'date' in df.columns else df.index[-1])
        last_time_numeric = df['time_numeric'].iloc[-1] if 'time_numeric' in df.columns else len(df) - 1
        
        # Create future features
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
        future_df = pd.DataFrame({'date': future_dates})
        
        future_df['time_numeric'] = range(last_time_numeric + 1, last_time_numeric + periods + 1)
        future_df['month'] = future_df['date'].dt.month
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Use available features
        available_features = [col for col in self.feature_cols if col in future_df.columns]
        X_future = future_df[available_features]
        
        # Add missing features with last known values
        for col in self.feature_cols:
            if col not in X_future.columns:
                X_future[col] = df[col].iloc[-1] if col in df.columns else 0
        
        # Ensure correct order
        X_future = X_future[self.feature_cols]
        
        # Scale and predict
        X_future_scaled = self.scaler.transform(X_future)
        predictions = self.baseline_model.predict(X_future_scaled)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def save_models(self, model_dir: str = 'models'):
        """
        Save trained models to disk.
        
        Args:
            model_dir: Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        if self.prophet_model is not None:
            prophet_path = os.path.join(model_dir, 'prophet_model.pkl')
            with open(prophet_path, 'wb') as f:
                joblib.dump(self.prophet_model, f)
        
        if self.sarima_model is not None:
            sarima_path = os.path.join(model_dir, 'sarima_model.pkl')
            joblib.dump(self.sarima_model, sarima_path)
        
        if self.baseline_model is not None:
            baseline_path = os.path.join(model_dir, 'baseline_model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            
            joblib.dump(self.baseline_model, baseline_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    
    def load_models(self, model_dir: str = 'models'):
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
        """
        prophet_path = os.path.join(model_dir, 'prophet_model.pkl')
        if os.path.exists(prophet_path):
            with open(prophet_path, 'rb') as f:
                self.prophet_model = joblib.load(f)
        
        sarima_path = os.path.join(model_dir, 'sarima_model.pkl')
        if os.path.exists(sarima_path):
            self.sarima_model = joblib.load(sarima_path)
        
        baseline_path = os.path.join(model_dir, 'baseline_model.pkl')
        if os.path.exists(baseline_path):
            self.baseline_model = joblib.load(baseline_path)
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            self.feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
    
    def get_model_info(self) -> Dict:
        """
        Get information about trained models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': self.model_type,
            'prophet_trained': self.prophet_model is not None,
            'sarima_trained': self.sarima_model is not None,
            'baseline_trained': self.baseline_model is not None,
        }
        
        if self.prophet_model is not None:
            info['prophet_params'] = {
                'seasonality_mode': self.prophet_model.seasonality_mode,
                'changepoint_prior_scale': self.prophet_model.changepoint_prior_scale,
                'interval_width': self.prophet_model.interval_width
            }
        
        if self.baseline_model is not None:
            info['baseline_features'] = self.feature_cols
        
        return info
