"""
Insights Generator Module
Generate actionable business insights from forecasts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class InsightsGenerator:
    """Generate business-friendly insights and recommendations."""
    
    def __init__(self):
        self.insights = {}
    
    def analyze_forecast_trend(self, forecast_df: pd.DataFrame) -> Dict:
        """
        Analyze the trend in forecast data.
        
        Args:
            forecast_df: Forecast DataFrame with yhat column
            
        Returns:
            Dictionary with trend insights
        """
        future_forecast = forecast_df.tail(30)  # Next 30 days
        
        avg_forecast = future_forecast['yhat'].mean()
        first_week = future_forecast.head(7)['yhat'].mean()
        last_week = future_forecast.tail(7)['yhat'].mean()
        
        if first_week > 0:
            week_growth = ((last_week - first_week) / first_week) * 100
        else:
            week_growth = 0
        
        insights = {
            'avg_forecast': avg_forecast,
            'first_week_avg': first_week,
            'last_week_avg': last_week,
            'week_over_week_growth': week_growth,
            'total_forecast': future_forecast['yhat'].sum(),
            'min_forecast': future_forecast['yhat'].min(),
            'max_forecast': future_forecast['yhat'].max()
        }
        
        return insights
    
    def generate_inventory_recommendation(self, forecast_df: pd.DataFrame,
                                         current_inventory: float = None) -> str:
        """
        Generate inventory planning recommendations.
        
        Args:
            forecast_df: Forecast DataFrame
            current_inventory: Current inventory level (optional)
            
        Returns:
            Inventory recommendation text
        """
        insights = self.analyze_forecast_trend(forecast_df)
        
        avg_daily = insights['avg_forecast']
        max_daily = insights['max_forecast']
        total_30day = insights['total_forecast']
        
        # Safety stock calculation (1.5x max daily demand)
        safety_stock = max_daily * 1.5
        
        # Reorder point (7 days of average demand)
        reorder_point = avg_daily * 7
        
        recommendation = f"""
        ## 📦 Inventory Planning Recommendations
        
        ### Demand Forecast (Next 30 Days)
        - **Average Daily Demand**: {avg_daily:.0f} units
        - **Peak Daily Demand**: {max_daily:.0f} units
        - **Total 30-Day Demand**: {total_30day:.0f} units
        
        ### Stock Levels to Maintain
        - **Safety Stock**: {safety_stock:.0f} units
          - This buffer protects against unexpected demand spikes
        
        - **Reorder Point**: {reorder_point:.0f} units
          - Order new inventory when stock reaches this level
        
        - **Recommended Order Quantity**: {total_30day:.0f} units
          - Covers next month's demand with safety buffer
        
        ### Action Items
        ✅ **This Week**: 
        - Ensure you have at least {(avg_daily * 14):.0f} units in stock (2 weeks coverage)
        
        ✅ **Monthly Planning**: 
        - Order {total_30day * 1.1:.0f} units to cover demand plus 10% buffer
        
        ⚠️ **Watch Out**: 
        - Peak demand could reach {max_daily:.0f} units/day
        - Don't let inventory drop below {reorder_point:.0f} units
        """
        
        if current_inventory:
            days_coverage = current_inventory / avg_daily
            recommendation += f"""
        
        ### Current Status
        - **Current Inventory**: {current_inventory:.0f} units
        - **Days of Coverage**: {days_coverage:.1f} days
        - **Status**: {'✅ Sufficient' if days_coverage > 14 else '⚠️ Low - Reorder Soon' if days_coverage > 7 else '🚨 Critical - Reorder Immediately'}
        """
        
        return recommendation.strip()
    
    def generate_staffing_recommendation(self, forecast_df: pd.DataFrame,
                                       seasonality_info: Dict = None) -> str:
        """
        Generate staffing recommendations based on forecast.
        
        Args:
            forecast_df: Forecast DataFrame
            seasonality_info: Seasonality patterns from FeatureEngineer
            
        Returns:
            Staffing recommendation text
        """
        insights = self.analyze_forecast_trend(forecast_df)
        
        avg_daily = insights['avg_forecast']
        max_daily = insights['max_forecast']
        
        # Estimate staffing needs (assuming 1 staff can handle 50 units/day)
        units_per_staff = 50
        avg_staff_needed = np.ceil(avg_daily / units_per_staff)
        peak_staff_needed = np.ceil(max_daily / units_per_staff)
        
        recommendation = f"""
        ## 👥 Staffing Recommendations
        
        ### Demand-Based Staffing
        - **Average Daily Demand**: {avg_daily:.0f} units
        - **Peak Daily Demand**: {max_daily:.0f} units
        
        ### Recommended Staff Levels
        - **Regular Staff**: {avg_staff_needed:.0f} employees
          - Sufficient for typical daily operations
        
        - **Peak Hours Staff**: {peak_staff_needed:.0f} employees
          - Need {peak_staff_needed - avg_staff_needed:.0f} additional staff for busy periods
        
        ### Scheduling Strategy
        """
        
        # Add day-of-week insights if available
        if seasonality_info and 'day_of_week' in seasonality_info:
            dow_info = seasonality_info['day_of_week']
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            peak_day = day_names[dow_info['peak_day']]
            low_day = day_names[dow_info['low_day']]
            
            recommendation += f"""
        
        📅 **Day-of-Week Patterns**:
        - **Busiest Day**: {peak_day} - Schedule {peak_staff_needed:.0f} staff
        - **Slowest Day**: {low_day} - Can operate with {max(1, avg_staff_needed - 1):.0f} staff
        """
        
        # Add weekend insights if available
        if seasonality_info and 'weekend_effect' in seasonality_info:
            we_info = seasonality_info['weekend_effect']
            if we_info['significance'] != 'low':
                diff = we_info['percent_difference']
                
                if diff > 0:
                    recommendation += f"""
        
        🎉 **Weekend Surge**: 
        - Weekends are {abs(diff):.0f}% busier
        - Add {np.ceil(avg_staff_needed * 0.3):.0f} extra staff on Saturday/Sunday
        """
                else:
                    recommendation += f"""
        
        💼 **Weekday Focus**: 
        - Weekdays are {abs(diff):.0f}% busier
        - Can reduce weekend staff by {np.ceil(avg_staff_needed * 0.2):.0f}
        """
        
        recommendation += """
        
        ### Cost Optimization
        ✅ **Save Money**: 
        - Use part-time staff for peak hours only
        - Cross-train employees for flexibility
        
        ✅ **Avoid Issues**: 
        - Don't under-staff during forecast peaks
        - Keep 1-2 on-call staff for unexpected rushes
        """
        
        return recommendation.strip()
    
    def generate_budget_recommendation(self, forecast_df: pd.DataFrame,
                                      avg_price_per_unit: float = None) -> str:
        """
        Generate budget and revenue projections.
        
        Args:
            forecast_df: Forecast DataFrame
            avg_price_per_unit: Average price per unit sold
            
        Returns:
            Budget recommendation text
        """
        insights = self.analyze_forecast_trend(forecast_df)
        
        # Get weekly forecasts
        future_30days = forecast_df.tail(30)
        week1 = future_30days.head(7)['yhat'].sum()
        week2 = future_30days.iloc[7:14]['yhat'].sum()
        week3 = future_30days.iloc[14:21]['yhat'].sum()
        week4 = future_30days.iloc[21:28]['yhat'].sum()
        
        recommendation = f"""
        ## 💰 Budget & Revenue Projections
        
        ### 30-Day Sales Forecast
        - **Week 1**: {week1:.0f} units
        - **Week 2**: {week2:.0f} units
        - **Week 3**: {week3:.0f} units
        - **Week 4**: {week4:.0f} units
        - **Total 30 Days**: {insights['total_forecast']:.0f} units
        
        ### Growth Trend
        - **Week-over-Week Change**: {insights['week_over_week_growth']:+.1f}%
        """
        
        if avg_price_per_unit:
            revenue_30day = insights['total_forecast'] * avg_price_per_unit
            revenue_week1 = week1 * avg_price_per_unit
            revenue_week4 = week4 * avg_price_per_unit
            
            recommendation += f"""
        
        ### Revenue Projections (at ${avg_price_per_unit:.2f} per unit)
        - **Week 1 Revenue**: ${revenue_week1:,.2f}
        - **Week 4 Revenue**: ${revenue_week4:,.2f}
        - **30-Day Revenue**: ${revenue_30day:,.2f}
        - **Projected Annual**: ${revenue_30day * 12:,.2f}
        """
        
        recommendation += f"""
        
        ### Budget Allocation Suggestions
        
        📊 **Operating Costs** (Plan for):
        - **Inventory Costs**: Budget for {insights['total_forecast']:.0f} units
        - **Staffing**: Based on {insights['avg_forecast']:.0f} units/day capacity
        - **Buffer**: Keep 10-15% emergency fund for demand spikes
        
        💡 **Investment Opportunities**:
        - {'**Growth Mode**: Sales trending up - consider expanding' if insights['week_over_week_growth'] > 5 else '**Stable Operations**: Steady demand - optimize efficiency'}
        - Forecast shows {'high confidence' if insights['avg_forecast'] > 0 else 'need more data'}
        
        ⚠️ **Risk Management**:
        - Don't over-commit to fixed costs during uncertain periods
        - Maintain cash reserves for {insights['max_forecast']:.0f} units/day peak capacity
        """
        
        return recommendation.strip()
    
    def generate_growth_opportunities(self, forecast_df: pd.DataFrame,
                                     trend_info: Dict = None) -> str:
        """
        Identify growth opportunities from forecast data.
        
        Args:
            forecast_df: Forecast DataFrame
            trend_info: Trend information from FeatureEngineer
            
        Returns:
            Growth opportunity recommendations
        """
        insights = self.analyze_forecast_trend(forecast_df)
        
        growth_rate = insights['week_over_week_growth']
        
        recommendation = f"""
        ## 🚀 Growth Opportunities & Strategic Insights
        
        ### Current Trajectory
        - **Forecast Trend**: {'+' if growth_rate > 0 else ''}{growth_rate:.1f}% over next month
        - **Demand Pattern**: {'Growing' if growth_rate > 5 else 'Declining' if growth_rate < -5 else 'Stable'}
        """
        
        if trend_info:
            recommendation += f"""
        - **Overall Trend**: {trend_info.get('direction', 'N/A').title()} ({trend_info.get('strength', 'N/A')})
        - **Long-term Change**: {trend_info.get('percent_change', 0):+.1f}%
        """
        
        recommendation += """
        
        ### Recommended Actions
        """
        
        if growth_rate > 10:
            recommendation += """
        
        ✨ **Strong Growth Detected!**
        
        1. **Expand Capacity**: 
           - Your demand is growing rapidly
           - Consider increasing inventory by 20-30%
           - Explore bulk purchasing for better margins
        
        2. **Marketing Amplification**:
           - Whatever you're doing is working - do more of it!
           - Invest in customer acquisition now while momentum is high
        
        3. **Operational Scaling**:
           - Hire additional staff proactively
           - Automate processes to handle volume
           - Consider expanding product line
        
        ⚠️ **Watch Out**: Don't let quality slip during rapid growth
        """
        
        elif growth_rate < -10:
            recommendation += """
        
        📉 **Declining Trend Alert**
        
        1. **Investigate Root Causes**:
           - Customer feedback - are you meeting needs?
           - Competition - what are others doing?
           - Market changes - external factors?
        
        2. **Recovery Actions**:
           - Promotional campaigns to boost sales
           - Product refresh or new offerings
           - Customer re-engagement initiatives
        
        3. **Cost Management**:
           - Reduce inventory levels proportionally
           - Optimize staffing to match lower demand
           - Focus on profitability over volume
        
        💡 **Opportunity**: Down periods are great for process improvements
        """
        
        else:
            recommendation += """
        
        ➡️ **Stable Operations**
        
        1. **Optimization Focus**:
           - Perfect time to improve efficiency
           - Reduce costs without cutting capacity
           - Automate repetitive tasks
        
        2. **Customer Retention**:
           - Build loyalty programs
           - Improve customer experience
           - Gather feedback for future improvements
        
        3. **Strategic Planning**:
           - Test new products/services
           - Explore new markets
           - Build reserves for future growth
        
        💡 **Opportunity**: Stability enables strategic thinking
        """
        
        recommendation += f"""
        
        ### Key Performance Indicators to Track
        
        📊 **Monitor These Metrics**:
        1. **Daily Sales** vs Forecast ({insights['avg_forecast']:.0f} units/day target)
        2. **Inventory Turnover** (should match {insights['total_forecast']:.0f} units/month)
        3. **Customer Acquisition Cost** (optimize during {'growth' if growth_rate > 0 else 'stable'} phase)
        4. **Gross Margin** (protect during volume changes)
        
        🎯 **Success Indicators**:
        - Forecast accuracy within 15% of actual
        - No stockouts during peak periods
        - Staff utilization > 80%
        - Customer satisfaction maintained/improved
        """
        
        return recommendation.strip()
    
    def generate_complete_insights(self, forecast_df: pd.DataFrame,
                                   trend_info: Dict = None,
                                   seasonality_info: Dict = None,
                                   avg_price: float = None) -> Dict:
        """
        Generate all insights together.
        
        Args:
            forecast_df: Forecast DataFrame
            trend_info: Trend information
            seasonality_info: Seasonality information
            avg_price: Average price per unit
            
        Returns:
            Dictionary with all insights
        """
        insights = {
            'inventory': self.generate_inventory_recommendation(forecast_df),
            'staffing': self.generate_staffing_recommendation(forecast_df, seasonality_info),
            'budget': self.generate_budget_recommendation(forecast_df, avg_price),
            'growth': self.generate_growth_opportunities(forecast_df, trend_info)
        }
        
        self.insights = insights
        return insights
