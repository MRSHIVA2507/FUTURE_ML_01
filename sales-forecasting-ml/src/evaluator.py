"""
Evaluator Module
Calculate forecasting metrics and provide business-friendly explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ForecastEvaluator:
    """Evaluate forecast accuracy with business-friendly metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            MAPE value (as percentage)
        """
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'mae': self.calculate_mae(y_true, y_pred),
            'rmse': self.calculate_rmse(y_true, y_pred),
            'mape': self.calculate_mape(y_true, y_pred),
            'mean_actual': np.mean(y_true),
            'mean_predicted': np.mean(y_pred),
            'n_samples': len(y_true)
        }
        
        self.metrics = metrics
        return metrics
    
    def explain_mae(self, mae: float, mean_sales: float) -> str:
        """
        Provide business explanation of MAE.
        
        Args:
            mae: Mean Absolute Error value
            mean_sales: Average sales value
            
        Returns:
            Business-friendly explanation
        """
        pct_of_mean = (mae / mean_sales * 100) if mean_sales > 0 else 0
        
        explanation = f"""
        **Mean Absolute Error (MAE): {mae:.2f}**
        
        **What it means**: On average, our predictions are off by {mae:.0f} units.
        
        **In context**: This is {pct_of_mean:.1f}% of your average sales ({mean_sales:.0f} units).
        
        **Business impact**: 
        - If you order inventory based on forecasts, expect to be off by ±{mae:.0f} units on average
        - Lower MAE = more accurate predictions = better inventory planning
        """
        
        return explanation.strip()
    
    def explain_rmse(self, rmse: float, mae: float) -> str:
        """
        Provide business explanation of RMSE.
        
        Args:
            rmse: Root Mean Squared Error value
            mae: Mean Absolute Error value (for comparison)
            
        Returns:
            Business-friendly explanation
        """
        explanation = f"""
        **Root Mean Squared Error (RMSE): {rmse:.2f}**
        
        **What it means**: RMSE penalizes larger errors more heavily than smaller ones.
        
        **Comparison**: 
        - RMSE ({rmse:.0f}) vs MAE ({mae:.0f})
        - When RMSE is much higher than MAE, it means you have some large prediction errors
        
        **Business impact**: 
        - Helps identify if your forecast has occasional big misses
        - Important for risk management in inventory and budgeting
        """
        
        return explanation.strip()
    
    def explain_mape(self, mape: float) -> str:
        """
        Provide business explanation of MAPE.
        
        Args:
            mape: Mean Absolute Percentage Error value
            
        Returns:
            Business-friendly explanation
        """
        # Determine quality level
        if mape < 10:
            quality = "Excellent"
            recommendation = "Your forecasts are highly reliable for planning."
        elif mape < 20:
            quality = "Good"
            recommendation = "Your forecasts are reliable for most business decisions."
        elif mape < 30:
            quality = "Acceptable"
            recommendation = "Use forecasts with some caution and buffer inventory."
        else:
            quality = "Needs Improvement"
            recommendation = "Consider collecting more data or investigating unusual patterns."
        
        explanation = f"""
        **Mean Absolute Percentage Error (MAPE): {mape:.1f}%**
        
        **What it means**: Your predictions are typically within {mape:.1f}% of actual sales.
        
        **Quality Assessment**: {quality}
        
        **Business recommendation**: {recommendation}
        
        **Example**: 
        - If actual sales are 100 units, expect forecast to be between {100 - mape:.0f} and {100 + mape:.0f} units
        """
        
        return explanation.strip()
    
    def generate_evaluation_report(self, metrics: Dict = None) -> str:
        """
        Generate comprehensive evaluation report in business language.
        
        Args:
            metrics: Optional metrics dictionary (uses self.metrics if not provided)
            
        Returns:
            Complete evaluation report
        """
        if metrics is None:
            if not self.metrics:
                return "No evaluation metrics available yet."
            metrics = self.metrics
        
        mae = metrics['mae']
        rmse = metrics['rmse']
        mape = metrics['mape']
        mean_actual = metrics['mean_actual']
        
        report = f"""
        ## 📊 Forecast Accuracy Report
        
        ### Summary
        - **Samples Evaluated**: {metrics['n_samples']}
        - **Average Actual Sales**: {mean_actual:.0f} units
        - **Average Predicted Sales**: {metrics['mean_predicted']:.0f} units
        
        ### Detailed Metrics
        
        {self.explain_mape(mape)}
        
        ---
        
        {self.explain_mae(mae, mean_actual)}
        
        ---
        
        {self.explain_rmse(rmse, mae)}
        
        ---
        
        ### Why Forecasts Aren't Perfect
        
        **Remember**: No forecast is 100% accurate, and that's normal! Here's why:
        
        1. **Unpredictable Events**: Sudden market changes, holidays, promotions, weather, etc.
        2. **Random Variation**: Customer behavior naturally varies day-to-day
        3. **Limited Data**: The model can only learn from historical patterns
        4. **External Factors**: Economic changes, competition, trends
        
        **The Goal**: Make decisions with better information, not perfect information.
        
        ### How to Use These Forecasts
        
        ✅ **Do**:
        - Use forecasts for planning inventory levels
        - Schedule staff based on predicted busy/slow periods
        - Set realistic sales targets
        - Identify trends and seasonal patterns
        
        ❌ **Don't**:
        - Treat forecasts as exact guarantees
        - Ignore your business knowledge and intuition
        - Over-stock based solely on upper confidence bounds
        - Make major irreversible decisions without validation
        """
        
        return report.strip()
    
    def compare_models(self, metrics_prophet: Dict, metrics_baseline: Dict) -> str:
        """
        Compare two models and recommend which to use.
        
        Args:
            metrics_prophet: Metrics from Prophet model
            metrics_baseline: Metrics from baseline model
            
        Returns:
            Comparison report
        """
        prophet_mape = metrics_prophet['mape']
        baseline_mape = metrics_baseline['mape']
        
        if prophet_mape < baseline_mape:
            better_model = "Prophet"
            improvement = ((baseline_mape - prophet_mape) / baseline_mape) * 100
            recommendation = f"Prophet is {improvement:.1f}% more accurate"
        else:
            better_model = "Baseline"
            improvement = ((prophet_mape - baseline_mape) / prophet_mape) * 100
            recommendation = f"Baseline is {improvement:.1f}% more accurate"
        
        report = f"""
        ## 🔄 Model Comparison
        
        | Metric | Prophet | Baseline | Winner |
        |--------|---------|----------|--------|
        | MAPE | {prophet_mape:.1f}% | {baseline_mape:.1f}% | {better_model} |
        | MAE | {metrics_prophet['mae']:.2f} | {metrics_baseline['mae']:.2f} | {'Prophet' if metrics_prophet['mae'] < metrics_baseline['mae'] else 'Baseline'} |
        | RMSE | {metrics_prophet['rmse']:.2f} | {metrics_baseline['rmse']:.2f} | {'Prophet' if metrics_prophet['rmse'] < metrics_baseline['rmse'] else 'Baseline'} |
        
        ### Recommendation
        
        **Use {better_model} model** - {recommendation}
        
        **Why?**
        - {'Prophet handles seasonality and trends automatically' if better_model == 'Prophet' else 'Simple linear model works well for your data pattern'}
        - {'More sophisticated for complex patterns' if better_model == 'Prophet' else 'Faster and easier to interpret'}
        - Better accuracy for your specific business case
        """
        
        return report.strip()
    
    def calculate_confidence_score(self, mape: float) -> Tuple[str, int]:
        """
        Calculate a confidence score for the forecast.
        
        Args:
            mape: Mean Absolute Percentage Error
            
        Returns:
            Tuple of (confidence level, score out of 100)
        """
        if mape < 10:
            return ("Very High", 95)
        elif mape < 15:
            return ("High", 85)
        elif mape < 20:
            return ("Good", 75)
        elif mape < 30:
            return ("Moderate", 60)
        elif mape < 40:
            return ("Low", 45)
        else:
            return ("Very Low", 30)
