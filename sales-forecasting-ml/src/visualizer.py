"""
Visualizer Module
Create interactive and professional visualizations using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional


class ForecastVisualizer:
    """Create professional visualizations for forecasting results."""
    
    def __init__(self):
        # Professional color scheme
        self.colors = {
            'historical': '#1f77b4',  # Deep blue
            'forecast': '#2ca02c',    # Green
            'confidence': 'rgba(44, 160, 44, 0.2)',  # Light green
            'trend': '#ff7f0e',       # Orange
            'baseline': '#d62728',    # Red
            'accent': '#9467bd'       # Purple
        }
    
    def plot_historical_vs_forecast(self, historical_df: pd.DataFrame, 
                                    forecast_df: pd.DataFrame,
                                    date_col: str, sales_col: str,
                                    show_confidence: bool = True) -> go.Figure:
        """
        Create interactive plot comparing historical data with forecast.
        
        Args:
            historical_df: Historical data DataFrame
            forecast_df: Forecast DataFrame (Prophet format with ds, yhat, yhat_lower, yhat_upper)
            date_col: Name of date column in historical data
            sales_col: Name of sales column in historical data
            show_confidence: Whether to show confidence intervals
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_df[date_col],
            y=historical_df[sales_col],
            mode='lines',
            name='Historical Sales',
            line=dict(color=self.colors['historical'], width=2),
            hovertemplate='<b>Date</b>: %{x}<br><b>Sales</b>: %{y:.0f}<extra></extra>'
        ))
        
        # Forecast
        forecast_future = forecast_df[forecast_df['ds'] > historical_df[date_col].max()]
        
        if show_confidence and 'yhat_lower' in forecast_df.columns:
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat_lower'],
                mode='lines',
                fill='tonexty',
                fillcolor=self.colors['confidence'],
                line=dict(width=0),
                name='95% Confidence',
                hovertemplate='<b>Range</b>: %{y:.0f}<extra></extra>'
            ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color=self.colors['forecast'], width=2, dash='dash'),
            marker=dict(size=4),
            hovertemplate='<b>Date</b>: %{x}<br><b>Forecast</b>: %{y:.0f}<extra></extra>'
        ))
        
        # Layout
        fig.update_layout(
            title={
                'text': '📈 Sales Forecast: Historical vs Predicted',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title='Date',
            yaxis_title='Sales Volume',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_trend_decomposition(self, df: pd.DataFrame, date_col: str, 
                                sales_col: str) -> go.Figure:
        """
        Create trend decomposition visualization.
        
        Args:
            df: DataFrame with sales data
            date_col: Name of date column
            sales_col: Name of sales column
            
        Returns:
            Plotly figure object
        """
        # Calculate rolling average for trend
        df_trend = df.copy()
        df_trend['trend'] = df_trend[sales_col].rolling(window=30, center=True).mean()
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Original Sales Data', 'Trend (30-day Moving Average)'),
                           vertical_spacing=0.15)
        
        # Original data
        fig.add_trace(
            go.Scatter(x=df_trend[date_col], y=df_trend[sales_col],
                      mode='lines', name='Sales',
                      line=dict(color=self.colors['historical'], width=1.5)),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=df_trend[date_col], y=df_trend['trend'],
                      mode='lines', name='Trend',
                      line=dict(color=self.colors['trend'], width=3)),
            row=2, col=1
        )
        
        fig.update_layout(
            title={
                'text': '📊 Sales Trend Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sales", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        
        return fig
    
    def plot_seasonality(self, df: pd.DataFrame, sales_col: str, 
                        pattern: str = 'day_of_week') -> go.Figure:
        """
        Create seasonality pattern visualization.
        
        Args:
            df: DataFrame with time features
            sales_col: Name of sales column
            pattern: 'day_of_week' or 'monthly'
            
        Returns:
            Plotly figure object
        """
        if pattern == 'day_of_week' and 'day_of_week' in df.columns:
            # Day of week pattern
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            avg_by_day = df.groupby('day_of_week')[sales_col].agg(['mean', 'std']).reset_index()
            avg_by_day['day_name'] = avg_by_day['day_of_week'].apply(lambda x: day_names[x])
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=avg_by_day['day_name'],
                y=avg_by_day['mean'],
                error_y=dict(type='data', array=avg_by_day['std']),
                marker_color=self.colors['accent'],
                name='Average Sales'
            ))
            
            title = '📅 Weekly Sales Pattern'
            xaxis_title = 'Day of Week'
            
        elif pattern == 'monthly' and 'month' in df.columns:
            # Monthly pattern
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            avg_by_month = df.groupby('month')[sales_col].agg(['mean', 'std']).reset_index()
            avg_by_month['month_name'] = avg_by_month['month'].apply(lambda x: month_names[x-1])
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=avg_by_month['month_name'],
                y=avg_by_month['mean'],
                error_y=dict(type='data', array=avg_by_month['std']),
                marker_color=self.colors['forecast'],
                name='Average Sales'
            ))
            
            title = '📆 Monthly Sales Pattern'
            xaxis_title = 'Month'
        else:
            # Default: create empty figure
            fig = go.Figure()
            title = 'Seasonality Pattern Not Available'
            xaxis_title = ''
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title=xaxis_title,
            yaxis_title='Average Sales',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_forecast_table(self, forecast_df: pd.DataFrame, 
                           periods: int = 14) -> go.Figure:
        """
        Create a formatted table of forecast results.
        
        Args:
            forecast_df: Forecast DataFrame from Prophet
            periods: Number of future periods to show
            
        Returns:
            Plotly figure object with table
        """
        # Get future forecast only
        future_forecast = forecast_df.tail(periods)
        
        # Format data
        dates = future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        forecasts = [f"{val:.0f}" for val in future_forecast['yhat'].tolist()]
        
        if 'yhat_lower' in future_forecast.columns:
            lower = [f"{val:.0f}" for val in future_forecast['yhat_lower'].tolist()]
            upper = [f"{val:.0f}" for val in future_forecast['yhat_upper'].tolist()]
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['<b>Date</b>', '<b>Forecast</b>', '<b>Lower Bound</b>', '<b>Upper Bound</b>'],
                    fill_color=self.colors['accent'],
                    align='center',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[dates, forecasts, lower, upper],
                    fill_color='lavender',
                    align='center',
                    font=dict(size=11)
                )
            )])
        else:
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['<b>Date</b>', '<b>Forecast</b>'],
                    fill_color=self.colors['accent'],
                    align='center',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[dates, forecasts],
                    fill_color='lavender',
                    align='center',
                    font=dict(size=11)
                )
            )])
        
        fig.update_layout(
            title={
                'text': f'📋 Next {periods} Days Forecast',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black'}
            },
            height=400
        )
        
        return fig
    
    def plot_model_comparison(self, dates: pd.Series, actual: pd.Series,
                             prophet_pred: pd.Series, baseline_pred: pd.Series) -> go.Figure:
        """
        Compare predictions from different models.
        
        Args:
            dates: Date series
            actual: Actual values
            prophet_pred: Prophet predictions
            baseline_pred: Baseline predictions
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Actual data
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color=self.colors['historical'], width=2)
        ))
        
        # Prophet predictions
        fig.add_trace(go.Scatter(
            x=dates,
            y=prophet_pred,
            mode='lines',
            name='Prophet Model',
            line=dict(color=self.colors['forecast'], width=2, dash='dash')
        ))
        
        # Baseline predictions
        fig.add_trace(go.Scatter(
            x=dates,
            y=baseline_pred,
            mode='lines',
            name='Baseline Model',
            line=dict(color=self.colors['baseline'], width=2, dash='dot')
        ))
        
        fig.update_layout(
            title={
                'text': '🔄 Model Comparison: Actual vs Predictions',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_summary_metrics_viz(self, metrics: dict) -> go.Figure:
        """
        Create visual summary of key metrics.
        
        Args:
            metrics: Dictionary with evaluation metrics
            
        Returns:
            Plotly figure with metric cards
        """
        fig = go.Figure()
        
        # Create metric cards using annotations
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=0.1, color='white'),
            showlegend=False
        ))
        
        fig.update_layout(
            title={
                'text': '📊 Forecast Quality Metrics',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=300,
            annotations=[
                dict(
                    x=0.2, y=0.5,
                    text=f"<b>MAPE</b><br>{metrics.get('mape', 0):.1f}%",
                    showarrow=False,
                    font=dict(size=16),
                    bgcolor=self.colors['forecast'],
                    bordercolor='white',
                    borderwidth=2
                ),
                dict(
                    x=0.5, y=0.5,
                    text=f"<b>MAE</b><br>{metrics.get('mae', 0):.1f}",
                    showarrow=False,
                    font=dict(size=16),
                    bgcolor=self.colors['accent'],
                    bordercolor='white',
                    borderwidth=2
                ),
                dict(
                    x=0.8, y=0.5,
                    text=f"<b>RMSE</b><br>{metrics.get('rmse', 0):.1f}",
                    showarrow=False,
                    font=dict(size=16),
                    bgcolor=self.colors['trend'],
                    bordercolor='white',
                    borderwidth=2
                )
            ]
        )
        
        return fig
