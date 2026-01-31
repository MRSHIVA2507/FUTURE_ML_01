# � Sales Forecasting ML Platform

> AI-powered sales forecasting dashboard with Prophet, SARIMA, and Linear Regression models. Built with Streamlit for interactive business intelligence.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Developed by:** [Shiva](https://github.com/MRSHIVA2507)  
**Version:** 1.0  
**Last Updated:** January 2026

---

## 🚀 Features

### **Forecasting Models**
- ✅ **Facebook Prophet** - Advanced time-series with automatic seasonality detection
- ✅ **SARIMA** - Statistical forecasting (automatic fallback, no C++ compiler needed)
- ✅ **Linear Regression** - Baseline model with time features and lags

### **Analytics & Insights**
- 📈 **Trend Analysis** - Detect growing/declining/stable patterns
- 📅 **Seasonality Detection** - Weekly, monthly, yearly patterns
- 💡 **Business Recommendations** - Inventory, staffing, budget planning
- 📊 **Model Accuracy Metrics** - MAPE, MAE, RMSE with confidence scores

### **Interactive Dashboard**
- 🎨 Beautiful balanced bright UI with soft pastel gradients
- 📤 CSV/Excel file upload
- 🔮 Configurable forecast periods (7-90 days)
- 📥 Downloadable forecast CSV
- 📱 Responsive design

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.30.0 |
| **Forecasting** | Prophet 1.1.5, Statsmodels (SARIMA) |
| **ML/Analytics** | Scikit-learn 1.3.2, Pandas, NumPy |
| **Visualization** | Plotly 5.18.0, Matplotlib |
| **Styling** | Custom CSS (Glassmorphism, Gradients) |

---

## ⚡ Quick Start

### **Prerequisites**
- Python 3.8 or higher
- pip package manager

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sales-forecasting-ml.git
cd sales-forecasting-ml
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**
```bash
streamlit run app.py
```

4. **Open your browser**
```
http://localhost:8501
```

---

## � Usage

### **Step 1: Upload Data**
- Click "Browse files" in the sidebar
- Upload CSV or Excel file with:
  - **Date column** (any format)
  - **Sales column** (numeric values)

### **Step 2: Configure Settings**
- Choose forecast period (7-90 days)
- Select data cleaning strategy
- Optional: Add price per unit for revenue projections

### **Step 3: Analyze Results**
The dashboard automatically provides:
- 📊 Data quality report
- 📈 Trend & seasonality analysis
- 🔮 Future sales forecast with confidence intervals
- 💡 Actionable business insights

### **Step 4: Download & Share**
- Download forecast CSV
- Share insights with your team

---

## 📁 Project Structure

```
sales-forecasting-ml/
├── app.py                      # Main Streamlit dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── src/
│   ├── data_handler.py        # Data loading & cleaning
│   ├── feature_engineer.py    # Time features & lags
│   ├── forecaster.py          # Prophet, SARIMA, Linear models
│   ├── evaluator.py           # Accuracy metrics
│   ├── visualizer.py          # Plotly charts
│   └── insights_generator.py  # Business recommendations
├── data/
│   └── sample_sales.csv       # Sample dataset
└── models/                     # Saved model files (auto-generated)
```

---

## 🎯 Sample Dataset

Try the included sample dataset:
- **File:** `data/sample_sales.csv`
- **Period:** 2 years of daily sales
- **Features:** Realistic trends, seasonality, missing values

---

## 🔧 Configuration

### **Forecast Settings**
- **Periods:** 7-90 days
- **Frequency:** Daily (customizable in code)
- **Confidence:** 95% intervals

### **Data Cleaning Strategies**
- `forward_fill` - Fill missing values with last known value
- `mean` - Replace with column mean
- `median` - Replace with column median
- `drop` - Remove rows with missing values

---

## 📊 Model Details

### **Prophet**
- Additive/multiplicative seasonality
- Automatic trend changepoint detection
- Handles missing data robustly
- **Best for:** Regular patterns with seasonality

### **SARIMA**
- Configuration: (1,1,1) x (1,1,1,7)
- Weekly seasonal patterns
- No C++ compiler required
- **Best for:** When Prophet fails to install

### **Linear Regression**
- Features: time, month, day_of_week, lags, rolling stats
- Baseline for comparison
- Fast training
- **Best for:** Understanding feature importance

---

## 🐛 Troubleshooting

### **Prophet Installation Fails**
The dashboard automatically falls back to SARIMA. No action needed!

If you want Prophet:
```bash
pip install cmdstanpy
python -m cmdstanpy.install_cmdstan
```

### **CSV Encoding Errors**
The system auto-detects UTF-8 and ISO-8859-1 encodings.

### **Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

---

## 📈 Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **MAPE** | Mean Absolute Percentage Error | < 10% |
| **MAE** | Mean Absolute Error (in units) | Lower is better |
| **RMSE** | Root Mean Squared Error | Lower is better |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---


## 👨‍💻 Author

**Shiva**
- GitHub: [@yourusername](https://github.com/MRSHIVA2507)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/chittishivacharanreddy)
- Email: shivareddyy.dev@gmail.com

---

## � Acknowledgments

- **Facebook Prophet** - Time-series forecasting framework
- **Streamlit** - Amazing dashboard framework
- **Statsmodels** - SARIMA implementation
- **Plotly** - Interactive visualizations

---

## � Screenshots

### Dashboard Overview
![Dashboard](dashboard.png)

### Forecast Visualization
![Forecast](forecast.png)

### Business Insights
![Insights](businessinsights.png)

---


---



<div align="center">

**Made with ❤️ by Shiva**

[⬆ Back to Top](#-sales-forecasting-ml-platform)

</div>
