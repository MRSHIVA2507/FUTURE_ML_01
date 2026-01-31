"""
Sales Forecasting Dashboard
Interactive Streamlit application for sales/demand forecasting with business insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_handler import DataHandler
from src.feature_engineer import FeatureEngineer
from src.forecaster import SalesForecaster
from src.evaluator import ForecastEvaluator
from src.visualizer import ForecastVisualizer
from src.insights_generator import InsightsGenerator

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Light Dark UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Deep Midnight Background */
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 1rem 2rem;
        position: relative;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Premium Dark Glass Cards */
    .stAlert, .metric-card, div[data-testid="stExpander"] {
        background: rgba(17, 25, 40, 0.75) !important;
        backdrop-filter: blur(16px) cubic-bezier(0.4, 0, 0.2, 1) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        transition: all 0.3s ease;
    }
    
    .stAlert:hover, .metric-card:hover, div[data-testid="stExpander"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Glowing 3D Title */
    h1 {
        background: linear-gradient(to right, #00b4db, #0083b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
        font-size: 3.2rem !important;
        letter-spacing: 1.5px;
        margin-bottom: 1.5rem !important;
        filter: drop-shadow(0 0 15px rgba(0, 180, 219, 0.3));
    }
    
    /* Sleek Section Headers */
    h2 {
        color: #e0e6ed !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1.2rem !important;
        border-left: 5px solid #00b4db;
        padding-left: 15px;
    }
    
    /* High Contrast Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 180, 219, 0.2);
    }
    
    div[data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Neon Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00b4db 0%, #0083b0 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(0, 180, 219, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 180, 219, 0.5) !important;
        filter: brightness(1.1);
    }
    
    /* Dark Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b161b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8f9fa !important;
        -webkit-text-fill-color: #f8f9fa !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    /* Dark Input Fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #00b4db !important;
        box-shadow: 0 0 0 2px rgba(0, 180, 219, 0.2) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* File Uploader */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #00b4db;
        background: rgba(0, 180, 219, 0.05);
    }
    
    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: white;
        background: rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: #00b4db !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(0, 180, 219, 0.3);
    }
    
    /* DataFrames */
    .dataframe {
        background: #1a202c !important;
        color: white !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #e0e6ed !important;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #00b4db !important;
    }
    
    /* Notification Colors */
    .stSuccess {
        background: rgba(0, 200, 83, 0.1) !important;
        border-left: 4px solid #00c853 !important;
        color: #69f0ae !important;
    }
    .stInfo {
        background: rgba(41, 121, 255, 0.1) !important;
        border-left: 4px solid #2979ff !important;
        color: #82b1ff !important;
    }
    .stWarning {
        background: rgba(255, 171, 0, 0.1) !important;
        border-left: 4px solid #ffab00 !important;
        color: #ffd740 !important;
    }
    .stError {
        background: rgba(221, 44, 0, 0.1) !important;
        border-left: 4px solid #dd2c00 !important;
        color: #ff9e80 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #0f2027;
    }
    ::-webkit-scrollbar-thumb {
        background: #2c5364;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00b4db;
    }
    
    /* Text Colors */
    p, li, span {
        color: rgba(255, 255, 255, 0.85) !important;
        line-height: 1.6;
    }
    
    strong {
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Links */
    a {
        color: #00b4db !important;
        transition: color 0.3s ease;
    }
    a:hover {
        color: #0083b0 !important;
        text-shadow: 0 0 8px rgba(0, 180, 219, 0.4);
    }
    
    /* Plots */
    .stPlotlyChart {
        background: rgba(17, 25, 40, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'forecast_generated' not in st.session_state:
    st.session_state.forecast_generated = False

# Title and description
st.title("📈 Sales & Demand Forecasting Dashboard")
st.markdown("""
Welcome to your **AI-powered forecasting assistant**! Upload your sales data and get:
- 📊 **Accurate forecasts** for future sales
- 💡 **Business insights** for inventory, staffing, and budget planning
- 📈 **Trend analysis** to understand your business patterns
- 🎯 **Actionable recommendations** you can implement immediately
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    st.subheader("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Your file should contain at least a date column and a sales/quantity column"
    )
    
    # Forecast settings
    st.subheader("🔮 Forecast Settings")
    forecast_days = st.slider(
        "Forecast Period (days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to forecast into the future"
    )
    
    # Data cleaning strategy
    st.subheader("🧹 Data Cleaning")
    cleaning_strategy = st.selectbox(
        "Missing Value Strategy",
        ['forward_fill', 'mean', 'median', 'drop'],
        help="How to handle missing values in your data"
    )
    
    # Optional: Price per unit for revenue calculations
    st.subheader("💰 Optional Settings")
    include_revenue = st.checkbox("Include revenue projections")
    if include_revenue:
        avg_price = st.number_input(
            "Average Price per Unit ($)",
            min_value=0.01,
            value=10.0,
            step=0.1
        )
    else:
        avg_price = None
    
    st.markdown("---")
    st.markdown("""
    ### 📖 Data Format
    Your file should have:
    - **Date column**: Any date format
    - **Sales column**: Numeric values
    - Optional: Category, Product, etc.
    
    Don't have data? Use our [sample dataset](sample_sales.csv)!
    """)

# Main content
if uploaded_file is not None:
    try:
        # Initialize components
        data_handler = DataHandler()
        feature_engineer = FeatureEngineer()
        forecaster = SalesForecaster()
        evaluator = ForecastEvaluator()
        visualizer = ForecastVisualizer()
        insights_gen = InsightsGenerator()
        
        # Load data
        with st.spinner("📥 Loading your data..."):
            df_raw = data_handler.load_data(uploaded_file)
            st.session_state.df_raw = df_raw
        
        st.success("✅ Data loaded successfully!")
        
        # Detect columns
        date_col = data_handler.detect_date_column(df_raw)
        sales_col = data_handler.detect_sales_column(df_raw, date_col)
        
        if date_col is None or sales_col is None:
            st.error("❌ Could not detect date and sales columns. Please check your data format.")
            st.stop()
        
        # Let user confirm or change column selection
        st.subheader("🔍 Column Detection")
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox(
                "Date Column",
                options=df_raw.columns.tolist(),
                index=df_raw.columns.tolist().index(date_col) if date_col else 0
            )
        
        with col2:
            sales_col = st.selectbox(
                "Sales Column",
                options=df_raw.columns.tolist(),
                index=df_raw.columns.tolist().index(sales_col) if sales_col else 0
            )
        
        # Clean data
        with st.spinner("🧹 Cleaning data..."):
            df_clean, cleaning_report = data_handler.clean_data(
                df_raw, date_col, sales_col, cleaning_strategy
            )
            st.session_state.df_clean = df_clean
        
        # === MULTI-LANDING SECTION ===
        st.markdown("---")
        
        # Create 5 distinct tabs for the key areas
        tab_cleaning, tab_features, tab_forecast, tab_eval, tab_insights = st.tabs([
            "🧹 Data Cleaning", 
            "📅 Feature Engineering", 
            "🔮 Forecasting", 
            "📊 Model Evaluation", 
            "💡 Business Insights"
        ])
        
        # --- TAB 1: DATA CLEANING & PREPARATION ---
        with tab_cleaning:
            st.header("🧹 Data Cleaning & Preparation")
            st.markdown("Review how your data was processed and cleaned for optimal results.")
            
            # Data quality metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{cleaning_report['final_rows']:,}")
            with col2:
                st.metric("Data Quality", f"{data_handler._calculate_quality_score(df_clean):.0f}%")
            with col3:
                date_range_days = (df_clean[date_col].max() - df_clean[date_col].min()).days
                st.metric("Date Range", f"{date_range_days} days")
            with col4:
                st.metric("Avg Sales", f"{df_clean[sales_col].mean():.0f}")
            
            # Show cleaning report
            with st.container():
                st.subheader("Data Processing Summary")
                st.info(f"**Strategy Used**: {cleaning_strategy} - {data_handler.explain_cleaning_strategy(cleaning_strategy)}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Rows", cleaning_report['original_rows'])
                with col2:
                    st.metric("Rows Removed", cleaning_report['rows_removed'])
                with col3:
                    st.metric("Final Rows", cleaning_report['final_rows'])
            
            # Show data preview
            st.subheader("Cleaned Data Preview")
            st.dataframe(df_clean.head(20), use_container_width=True)
            
            # Statistical Summary
            st.subheader("Statistical Analysis")
            data_report = data_handler.generate_data_report(df_clean, date_col, sales_col)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Sales Statistics**")
                st.write(f"- **Mean**: {data_report['sales_stats']['mean']:.0f}")
                st.write(f"- **Median**: {data_report['sales_stats']['median']:.0f}")
                st.write(f"- **Std Dev**: {data_report['sales_stats']['std']:.0f}")
            with col2:
                st.markdown("**Timeline**")
                st.write(f"- **Start**: {data_report['date_range']['start']}")
                st.write(f"- **End**: {data_report['date_range']['end']}")
                st.write(f"- **Duration**: {data_report['date_range']['total_days']} days")

        # --- TAB 2: FEATURE ENGINEERING & ANALYSIS ---
        with tab_features:
            st.header("📅 Time-Based Feature Engineering")
            st.markdown("Analysis of trends, seasonality, and time-based patterns in your data.")
            
            # Feature Engineering Process
            with st.spinner("⚙️ Engineering features..."):
                df_features = feature_engineer.extract_time_features(df_clean, date_col)
                df_features = feature_engineer.create_lag_features(df_features, sales_col)
                df_features = feature_engineer.create_rolling_features(df_features, sales_col)
                
                # Detect patterns
                trend_info = feature_engineer.detect_trend(df_features, date_col, sales_col)
                seasonality_info = feature_engineer.detect_seasonality(df_features, sales_col)
                
                st.session_state.df_features = df_features
                st.session_state.trend_info = trend_info
                st.session_state.seasonality_info = seasonality_info
            
            # Trend Analysis
            st.subheader("📈 Trend Analysis")
            st.markdown(feature_engineer.get_trend_explanation())
            fig_trend = visualizer.plot_trend_decomposition(df_features, date_col, sales_col)
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Seasonality Analysis
            st.subheader("📅 Seasonality Patterns")
            st.markdown(feature_engineer.get_seasonality_explanation())
            
            col1, col2 = st.columns(2)
            with col1:
                if 'day_of_week' in seasonality_info:
                    st.markdown("**Weekly Patterns**")
                    fig_dow = visualizer.plot_seasonality(df_features, sales_col, 'day_of_week')
                    st.plotly_chart(fig_dow, use_container_width=True)
            with col2:
                if 'monthly' in seasonality_info:
                    st.markdown("**Monthly Patterns**")
                    fig_month = visualizer.plot_seasonality(df_features, sales_col, 'monthly')
                    st.plotly_chart(fig_month, use_container_width=True)

        # --- TAB 3: FORECASTING MODELS ---
        with tab_forecast:
            st.header("🔮 Forecasting Models")
            st.markdown("Generate future predictions using advanced regression and time-series approaches.")
            
            with st.spinner("🤖 Training forecasting models... This may take a minute..."):
                # Train Prophet model
                forecaster.train_prophet_model(df_clean, date_col, sales_col)
                
                # Generate forecast
                forecast_prophet = forecaster.predict_prophet(periods=forecast_days)
                
                st.session_state.forecast_prophet = forecast_prophet
                st.session_state.forecast_generated = True
            
            # Display model info
            model_info = forecaster.get_model_info()
            model_name = "🤖 Prophet" if model_info['model_type'] == 'prophet' else "📊 SARIMA"
            
            st.success(f"✅ Forecast generated successfully using **{model_name}** model!")
            
            with st.expander("ℹ️ Model Details", expanded=False):
                if model_info['model_type'] == 'prophet':
                    st.markdown("""
                    **Model Used**: Facebook Prophet (Time-Series)
                    - Automatic trend detection & seasonal pattern recognition
                    - 95% confidence intervals
                    """)
                elif model_info['model_type'] == 'sarima':
                    st.markdown("""
                    **Model Used**: SARIMA (Seasonal ARIMA)
                    - Statistical time-series forecasting
                    - Configuration: (1,1,1)x(1,1,1,7)
                    """)
                st.info("💡 A baseline **Linear Regression** model is also trained for comparison.")
            
            # Visualizations
            st.subheader(f"Next {forecast_days} Days Forecast")
            fig_forecast = visualizer.plot_historical_vs_forecast(
                df_clean, forecast_prophet, date_col, sales_col, show_confidence=True
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # detailed table
            with st.expander("📋 View Detailed Forecast Data"):
                future_only = forecast_prophet[forecast_prophet['ds'] > df_clean[date_col].max()]
                forecast_display = future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                forecast_display['Forecast'] = forecast_display['Forecast'].round(0).astype(int)
                forecast_display['Lower Bound'] = forecast_display['Lower Bound'].round(0).astype(int)
                forecast_display['Upper Bound'] = forecast_display['Upper Bound'].round(0).astype(int)
                
                st.dataframe(forecast_display.head(30), use_container_width=True)
                
                csv = forecast_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        # --- TAB 4: MODEL EVALUATION ---
        with tab_eval:
            st.header("📊 Model Evaluation & Error Analysis")
            st.markdown("Comprehensive performance metrics and error analysis.")
            
            with st.spinner("📏 Evaluating forecast accuracy..."):
                # Use last 20% of data for testing
                split_idx = int(len(df_clean) * 0.8)
                train_data = df_clean.iloc[:split_idx]
                test_data = df_clean.iloc[split_idx:]
                
                # Train on training data
                forecaster_eval = SalesForecaster()
                forecaster_eval.train_prophet_model(train_data, date_col, sales_col)
                
                # Predict on test period
                test_periods = len(test_data)
                forecast_test = forecaster_eval.predict_prophet(periods=test_periods)
                
                # Get predictions for test period
                test_predictions = forecast_test.tail(test_periods)['yhat'].values
                test_actuals = test_data[sales_col].values
                
                # Calculate metrics
                metrics = evaluator.calculate_all_metrics(test_actuals, test_predictions)
            
            # Metrics Display
            col1, col2, col3 = st.columns(3)
            with col1:
                confidence_level, confidence_score = evaluator.calculate_confidence_score(metrics['mape'])
                st.metric("MAPE (Accuracy in %)", f"{metrics['mape']:.1f}%")
                st.caption(f"Confidence Level: {confidence_level}")
            with col2:
                st.metric("MAE (Avg Error Units)", f"{metrics['mae']:.0f}")
                pct_of_mean = (metrics['mae'] / metrics['mean_actual'] * 100) if metrics['mean_actual'] > 0 else 0
                st.caption(f"Error represents {pct_of_mean:.1f}% of avg sales")
            with col3:
                st.metric("RMSE (Root Mean Sq Error)", f"{metrics['rmse']:.0f}")
            
            st.subheader("Detailed Performance Report")
            st.markdown(evaluator.generate_evaluation_report(metrics))

        # --- TAB 5: BUSINESS INSIGHTS ---
        with tab_insights:
            st.header("💡 Business-Friendly Visualizations")
            st.markdown("Actionable insights and recommendations for stakeholders.")
            
            st.info("""
            **Decision Support**: These insights are derived directly from the forecasting models to support your strategic planning.
            """)
            
            # Generate insights
            all_insights = insights_gen.generate_complete_insights(
                forecast_prophet,
                trend_info,
                seasonality_info,
                avg_price
            )
            
            # Sub-tabs for insights categorization
            subtab1, subtab2, subtab3, subtab4 = st.tabs(["📦 Inventory Planning", "👥 Staffing/Resources", "💰 Budget & Revenue", "🚀 Growth Strategy"])
            
            with subtab1:
                st.subheader("Inventory Recommendations")
                st.markdown(all_insights['inventory'])
            
            with subtab2:
                st.subheader("Staffing & Resource Allocation")
                st.markdown(all_insights['staffing'])
            
            with subtab3:
                st.subheader("Financial Projections")
                st.markdown(all_insights['budget'])
            
            with subtab4:
                st.subheader("Strategic Growth Opportunities")
                st.markdown(all_insights['growth'])
        
        # === FOOTER ===
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.1); border-radius: 15px; backdrop-filter: blur(10px);'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>🚀 Developed by Shiva</h3>
            <p style='color: #525f7f; font-size: 1.1rem;'>
                <strong>Sales Forecasting ML Platform</strong><br>
                Dashboard Version 1.0 | Powered by Prophet & SARIMA<br>
                Last Updated: {datetime.now().strftime('%B %d, %Y')}
            </p>
        </div>
        """)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

else:
    # Landing page when no file is uploaded
    st.info("👈 **Get started** by uploading your sales data using the sidebar")
    
    st.markdown("""
    ## 📖 How to Use This Dashboard
    
    ### Step 1: Prepare Your Data
    Your data file should be in CSV or Excel format with:
    - **Date column**: Sales date (e.g., '2023-01-15', '01/15/2023')
    - **Sales column**: Sales volume or quantity (numeric)
    
    ### Step 2: Upload & Configure
    1. Upload your file using the sidebar
    2. Confirm the date and sales columns are correctly detected
    3. Choose how many days to forecast (7-90 days)
    
    ### Step 3: Explore the 5 Key Modules
    
    1. **🧹 Data Cleaning**: Automated handling of missing values and quality checks
    2. **📅 Feature Engineering**: Analysis of trends, seasonality, and time patterns
    3. **🔮 Forecasting**: Advanced regression & time-series predictions
    4. **📊 Evaluation**: Strict accuracy checking and error analysis
    5. **💡 Insights**: Business-friendly actionable recommendations
    
    ---
    
    ## 📊 Sample Dataset
    Don't have data ready? 
    **Download**: [sample_sales.csv](data/sample_sales.csv)
    """)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 10px; margin-top: 2rem;'>
    <p style='color: #525f7f; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;'>
    📊 Sales Forecasting Platform
    </p>
    <p style='color: #525f7f; font-size: 0.9rem; margin-bottom: 0.5rem;'>
    Version 1.0 | AI-Powered Predictions
    </p>
    <p style='color: #525f7f; font-size: 0.85rem;'>
    © 2026 | Developed by <strong style='color: #667eea;'>Shiva</strong>
    </p>
    <p style='color: #525f7f; font-size: 0.8rem; margin-top: 0.5rem;'>
    Built with ❤️ using Prophet, SARIMA & Streamlit
    </p>
    </div>
    """, unsafe_allow_html=True)
