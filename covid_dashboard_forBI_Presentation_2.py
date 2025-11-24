import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="COVID-19 Business Intelligence - Anas Hussain", layout="wide")

st.title("📊 COVID-19 Global Dashboard - Business Intelligence Project")
st.markdown("""
#### 🧓‍♂️ **Name**: Anas Hussain (DS-017 23/24)
#### 📔 **GIT**: AnasMega https://github.com/AnasMega/convid-19-dashboard
#### 📞 **Contact**: anashussain0311@gmail.com, Anas.pg4000382@cloud.neduet.edu.pk
#### 🏫 **Department**: Computer Science  
#### 🧪 **Institute**: Data Science Institute, NED University of Engineering & Technology  
#### 📚 **Subject**: Power Bi and Data Analytics
""")

st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv("daily-new-confirmed-covid-19-cases-per-million-people.csv")
    df.columns = ["Country", "Date", "DailyCasesPerMillion"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# Sidebar Filters
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/NEDUET_logo.svg/800px-NEDUET_logo.svg.png", width=110)
    st.title("🔎 Filter Options")
    countries = sorted(df["Country"].unique())
    selected_countries = st.multiselect("Select Countries", countries, default=["Pakistan", "India", "United States"])
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

# Filter Data
filtered_df = df[
    (df["Country"].isin(selected_countries)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

# Create tabs for different analytical phases
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Descriptive Analysis", 
    "🔍 Diagnostic Analysis", 
    "🔮 Predictive Analysis", 
    "💡 Prescriptive Analysis"
])

# =============================================================================
# TAB 1: DESCRIPTIVE ANALYSIS
# =============================================================================
with tab1:
    st.header("📊 Descriptive Analysis: What Happened?")
    st.markdown("""
    This section provides a comprehensive overview of COVID-19 trends through visualizations and summary statistics.
    """)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_cases = filtered_df["DailyCasesPerMillion"].sum()
        st.metric("Total Cases per Million", f"{total_cases:,.0f}")
    with col2:
        avg_cases = filtered_df["DailyCasesPerMillion"].mean()
        st.metric("Average Daily Cases", f"{avg_cases:.2f}")
    with col3:
        peak_cases = filtered_df["DailyCasesPerMillion"].max()
        st.metric("Peak Daily Cases", f"{peak_cases:.2f}")
    with col4:
        st.metric("Countries Analyzed", len(selected_countries))
    
    # Chart 1: Trend Over Time
    st.subheader("📈 Trend Over Time: Daily COVID-19 Cases")
    st.markdown("**7-day rolling average showing the pandemic evolution over time**")
    
    if not filtered_df.empty:
        pivot_df = filtered_df.pivot(index="Date", columns="Country", values="DailyCasesPerMillion")
        rolling_df = pivot_df.rolling(window=7).mean()
        
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        for country in selected_countries:
            if country in rolling_df.columns:
                ax1.plot(rolling_df.index, rolling_df[country], label=country, linewidth=2)
        ax1.set_title("7-Day Rolling Average of Daily New Cases (Per Million People)", fontsize=16)
        ax1.set_ylabel("Daily Cases per 1 Million People")
        ax1.set_xlabel("Date")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
    else:
        st.warning("No data available for the selected filters.")
    
    # Chart 2: Average Cases by Country
    st.subheader("📊 Average Daily Cases by Country")
    st.markdown("**Comparative analysis of normalized case averages across selected nations**")
    
    if not filtered_df.empty:
        avg_df = filtered_df.groupby("Country")["DailyCasesPerMillion"].mean().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.barplot(x=avg_df.index, y=avg_df.values, palette="viridis", ax=ax2)
        ax2.set_ylabel("Average Daily Cases per 1 Million People")
        ax2.set_title("Average Daily Cases (Normalized by Population)")
        plt.xticks(rotation=45)
        st.pyplot(fig2)
    
    # NEW Chart: Cumulative Cases Over Time
    st.subheader("📈 Cumulative Cases Trend")
    st.markdown("**Total accumulated cases showing the pandemic's progressive impact**")
    
    if not filtered_df.empty:
        cumulative_df = filtered_df.copy()
        cumulative_df['CumulativeCases'] = cumulative_df.groupby('Country')['DailyCasesPerMillion'].cumsum()
        
        fig_cum, ax_cum = plt.subplots(figsize=(14, 6))
        for country in selected_countries:
            country_data = cumulative_df[cumulative_df['Country'] == country]
            if not country_data.empty:
                ax_cum.plot(country_data['Date'], country_data['CumulativeCases'], label=country, linewidth=2)
        ax_cum.set_title("Cumulative COVID-19 Cases per Million Over Time", fontsize=16)
        ax_cum.set_ylabel("Cumulative Cases per 1 Million People")
        ax_cum.set_xlabel("Date")
        ax_cum.legend()
        ax_cum.grid(True, alpha=0.3)
        st.pyplot(fig_cum)
    
    # Data Table
    st.subheader("📄 Raw Data Overview")
    if not filtered_df.empty:
        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

# =============================================================================
# TAB 2: DIAGNOSTIC ANALYSIS
# =============================================================================
with tab2:
    st.header("🔍 Diagnostic Analysis: Why Did It Happen?")
    st.markdown("""
    This section investigates patterns, correlations, and underlying factors in COVID-19 spread.
    """)
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()
    
    # Chart 3: Heatmap
    st.subheader("🔥 Weekly Heatmap of Cases")
    st.markdown("**Temporal pattern visualization showing outbreak intensity across weeks and countries**")
    
    heatmap_df = filtered_df.copy()
    heatmap_df["Week"] = heatmap_df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = heatmap_df.groupby(["Week", "Country"])["DailyCasesPerMillion"].mean().unstack().fillna(0)
    
    if not weekly.empty:
        fig3, ax3 = plt.subplots(figsize=(14, 8))
        sns.heatmap(weekly.T, cmap="YlGnBu", linewidths=0.1, linecolor='gray', ax=ax3)
        ax3.set_title("Weekly Average of Daily Cases per 1 Million People", fontsize=14)
        ax3.set_xlabel("Week")
        ax3.set_ylabel("Country")
        st.pyplot(fig3)
    
    # Distribution Analysis
    st.subheader("📊 Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart 4: Boxplot
        st.markdown("**Box Plot: Statistical distribution showing median, quartiles, and outliers**")
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_df, x="Country", y="DailyCasesPerMillion", palette="Set3", ax=ax_box)
        ax_box.set_title("Distribution of Daily Cases per 1 Million")
        ax_box.set_ylabel("Daily Cases per 1 Million")
        ax_box.set_xlabel("Country")
        plt.xticks(rotation=45)
        st.pyplot(fig_box)
    
    with col2:
        # Chart 5: Violin Plot
        st.markdown("**Violin Plot: Combined box plot and density distribution visualization**")
        fig_violin, ax_violin = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=filtered_df, x="Country", y="DailyCasesPerMillion", palette="pastel", ax=ax_violin)
        ax_violin.set_title("Density Distribution of Daily Cases")
        ax_violin.set_ylabel("Daily Cases per 1 Million")
        ax_violin.set_xlabel("Country")
        plt.xticks(rotation=45)
        st.pyplot(fig_violin)
    
    # Chart 6: Correlation Analysis
    st.subheader("🔗 Correlation Analysis Between Countries")
    st.markdown("**Inter-country relationship matrix showing synchronized spread patterns**")
    
    pivot_df = filtered_df.pivot(index="Date", columns="Country", values="DailyCasesPerMillion")
    correlation_df = pivot_df.corr()
    
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_df, annot=True, cmap="coolwarm", center=0, ax=ax_corr, fmt=".2f")
    ax_corr.set_title("Correlation Matrix of COVID-19 Cases Between Countries")
    st.pyplot(fig_corr)
    
    st.markdown("""
    **Insights from Diagnostic Analysis:**
    - Heatmap shows seasonal patterns and outbreak waves
    - Box plots reveal variability and outlier patterns
    - Correlation matrix indicates synchronized spread patterns between countries
    - Peak analysis identifies worst outbreak periods
    """)

# =============================================================================
# TAB 3: PREDICTIVE ANALYSIS
# =============================================================================
with tab3:
    st.header("🔮 Predictive Analysis: What Will Happen?")
    st.markdown("""
    This section uses statistical models to forecast future COVID-19 trends.
    """)
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()
    
    # Prediction Configuration
    col1, col2 = st.columns([1, 3])
    with col1:
        prediction_days = st.slider("Prediction Horizon (days)", min_value=7, max_value=90, value=30)
        confidence_level = st.slider("Confidence Level (%)", min_value=80, max_value=95, value=90)
        use_simple_model = st.checkbox("Use Simple Model (for limited data)", value=True)
    
    # Enhanced prediction function with multiple fallback methods
    def predict_future_enhanced(df, country, days, confidence=90, use_simple=True):
        try:
            country_data = df[df["Country"] == country].copy()
            
            if len(country_data) < 7:
                return None, "Insufficient data (need at least 7 days)"
                
            country_data = country_data.set_index("Date")
            country_data = country_data.resample("D").mean().fillna(0)
            cases_data = country_data["DailyCasesPerMillion"]
            
            # Remove any remaining NaN values
            cases_data = cases_data.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if len(cases_data) < 7:
                return None, "Insufficient data after cleaning"
            
            # Method 1: Simple moving average (works with limited data)
            if use_simple or len(cases_data) < 30:
                st.info(f"Using simple forecasting for {country} (limited data)")
                
                # Use weighted average of recent trend
                recent_avg = cases_data.tail(7).mean()
                trend = 0
                if len(cases_data) >= 14:
                    # Calculate simple trend
                    recent_week = cases_data.tail(7).mean()
                    previous_week = cases_data.tail(14).head(7).mean()
                    trend = (recent_week - previous_week) / max(previous_week, 1) * 0.1
                
                # Generate simple forecast
                forecast_dates = pd.date_range(cases_data.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
                base_prediction = max(recent_avg, 0)
                
                predictions = []
                lower_bounds = []
                upper_bounds = []
                
                for i in range(days):
                    # Simple decay model with some randomness
                    pred = base_prediction * (1 + trend) * (0.95 ** (i/7))  # Gentle decay
                    pred = max(pred, 0)  # Ensure non-negative
                    
                    predictions.append(pred)
                    
                    # Simple confidence intervals
                    uncertainty = pred * 0.3  # 30% uncertainty
                    lower_bounds.append(max(pred - uncertainty, 0))
                    upper_bounds.append(pred + uncertainty)
                
                prediction_df = pd.DataFrame({
                    "Date": forecast_dates,
                    "PredictedCases": predictions,
                    "Lower_CI": lower_bounds,
                    "Upper_CI": upper_bounds
                })
                
                return prediction_df, "Success (Simple Model)"
            
            # Method 2: ARIMA model (for sufficient data)
            else:
                try:
                    # Fit ARIMA model with simpler parameters for stability
                    model = ARIMA(cases_data, order=(2, 1, 1))
                    model_fit = model.fit()
                    
                    # Generate forecast with confidence intervals
                    forecast = model_fit.get_forecast(steps=days)
                    forecast_mean = forecast.predicted_mean
                    confidence_interval = forecast.conf_int(alpha=1-confidence/100)
                    
                    forecast_dates = pd.date_range(cases_data.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
                    
                    prediction_df = pd.DataFrame({
                        "Date": forecast_dates,
                        "PredictedCases": forecast_mean,
                        "Lower_CI": confidence_interval.iloc[:, 0],
                        "Upper_CI": confidence_interval.iloc[:, 1]
                    })
                    
                    return prediction_df, "Success (ARIMA Model)"
                    
                except Exception as arima_error:
                    # Fallback to simple model if ARIMA fails
                    st.warning(f"ARIMA failed for {country}, using simple model: {str(arima_error)}")
                    return predict_future_enhanced(df, country, days, confidence, use_simple=True)
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    # Alternative simple prediction methods
    def simple_trend_prediction(df, country, days):
        """Very simple trend-based prediction"""
        try:
            country_data = df[df["Country"] == country].copy()
            if len(country_data) < 3:
                return None, "Need at least 3 data points"
                
            country_data = country_data.set_index("Date")
            cases_data = country_data["DailyCasesPerMillion"].fillna(0)
            
            # Simple average of last few points
            recent_avg = cases_data.tail(min(7, len(cases_data))).mean()
            
            # Very basic trend calculation
            if len(cases_data) >= 5:
                recent = cases_data.tail(3).mean()
                older = cases_data.tail(6).head(3).mean()
                trend_factor = (recent - older) / max(older, 1) if older > 0 else 0
            else:
                trend_factor = 0
            
            forecast_dates = pd.date_range(cases_data.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
            
            predictions = []
            for i in range(days):
                # Simple projection with trend decay
                pred = recent_avg * (1 + trend_factor * (0.8 ** i))
                predictions.append(max(pred, 0))
            
            # Simple confidence intervals
            uncertainty = np.array(predictions) * 0.5  # 50% uncertainty for simple model
            
            prediction_df = pd.DataFrame({
                "Date": forecast_dates,
                "PredictedCases": predictions,
                "Lower_CI": np.maximum(np.array(predictions) - uncertainty, 0),
                "Upper_CI": np.array(predictions) + uncertainty
            })
            
            return prediction_df, "Success (Trend Model)"
            
        except Exception as e:
            return None, f"Trend prediction error: {str(e)}"
    
    def naive_seasonal_prediction(df, country, days):
        """Naive seasonal prediction based on weekly patterns"""
        try:
            country_data = df[df["Country"] == country].copy()
            if len(country_data) < 7:
                return None, "Need at least 7 days for seasonal pattern"
                
            country_data = country_data.set_index("Date")
            cases_data = country_data["DailyCasesPerMillion"].fillna(0)
            
            # Calculate weekly pattern (very basic)
            weekly_pattern = []
            if len(cases_data) >= 7:
                for i in range(7):
                    day_data = cases_data.iloc[-(7-i)::7]  # Get same weekday data
                    if len(day_data) > 0:
                        weekly_pattern.append(day_data.mean())
                    else:
                        weekly_pattern.append(cases_data.mean())
            else:
                weekly_pattern = [cases_data.mean()] * 7
            
            # Normalize pattern
            pattern_sum = sum(weekly_pattern)
            if pattern_sum > 0:
                weekly_pattern = [p/pattern_sum * 7 * cases_data.mean() for p in weekly_pattern]
            
            forecast_dates = pd.date_range(cases_data.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
            
            predictions = []
            for i in range(days):
                day_of_week = forecast_dates[i].weekday()
                pred = weekly_pattern[day_of_week] * 0.9  # Slight decay
                predictions.append(max(pred, 0))
            
            # Confidence intervals
            uncertainty = np.array(predictions) * 0.6
            
            prediction_df = pd.DataFrame({
                "Date": forecast_dates,
                "PredictedCases": predictions,
                "Lower_CI": np.maximum(np.array(predictions) - uncertainty, 0),
                "Upper_CI": np.array(predictions) + uncertainty
            })
            
            return prediction_df, "Success (Seasonal Model)"
            
        except Exception as e:
            return None, f"Seasonal prediction error: {str(e)}"
    
    # Display predictions for each selected country
    successful_predictions = 0
    
    for country in selected_countries:
        country_data = filtered_df[filtered_df["Country"] == country]
        
        if len(country_data) == 0:
            st.warning(f"No data available for {country}")
            continue
            
        st.subheader(f"📈 {country} - {prediction_days}-Day Forecast")
        
        # Show data availability info
        data_days = len(country_data)
        st.write(f"**Data Availability:** {data_days} days of data")
        
        if data_days < 7:
            st.warning(f"⚠️ Limited data for {country}. Using very simple prediction method.")
        
        # Try multiple prediction methods
        prediction_df = None
        status = ""
        method_used = ""
        
        # Method 1: Enhanced prediction
        prediction_df, status = predict_future_enhanced(filtered_df, country, prediction_days, confidence_level, use_simple_model)
        method_used = "Enhanced Model"
        
        # Method 2: Fallback to simple trend
        if prediction_df is None:
            prediction_df, status = simple_trend_prediction(filtered_df, country, prediction_days)
            method_used = "Trend Model"
        
        # Method 3: Fallback to seasonal
        if prediction_df is None:
            prediction_df, status = naive_seasonal_prediction(filtered_df, country, prediction_days)
            method_used = "Seasonal Model"
        
        if prediction_df is not None:
            successful_predictions += 1
            
            st.markdown(f"**{method_used}** - {status}")
            
            # Create visualization
            fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            historical_data = filtered_df[filtered_df["Country"] == country]
            ax_pred.plot(historical_data["Date"], historical_data["DailyCasesPerMillion"], 
                        label="Historical Data", color="blue", linewidth=2, marker='o', markersize=2)
            
            # Plot prediction
            ax_pred.plot(prediction_df["Date"], prediction_df["PredictedCases"], 
                        label="Predicted", color="red", linewidth=2, linestyle='--', marker='s', markersize=3)
            
            # Plot confidence interval
            ax_pred.fill_between(prediction_df["Date"], 
                               prediction_df["Lower_CI"], 
                               prediction_df["Upper_CI"], 
                               color="red", alpha=0.2, label=f"{confidence_level}% Confidence Interval")
            
            ax_pred.set_title(f"COVID-19 Forecast for {country} ({method_used})", fontsize=14)
            ax_pred.set_xlabel("Date")
            ax_pred.set_ylabel("Cases per 1 Million People")
            ax_pred.legend()
            ax_pred.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_pred)
            
            # Show prediction summary with enhanced metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_prediction = prediction_df["PredictedCases"].mean()
                st.metric("Average Predicted", f"{avg_prediction:.1f}")
            with col2:
                max_prediction = prediction_df["PredictedCases"].max()
                st.metric("Peak Predicted", f"{max_prediction:.1f}")
            with col3:
                current_cases = historical_data["DailyCasesPerMillion"].iloc[-1] if len(historical_data) > 0 else 0
                predicted_change = ((prediction_df["PredictedCases"].iloc[-1] - current_cases) / max(current_cases, 1)) * 100
                st.metric("Projected Change", f"{predicted_change:+.1f}%")
            with col4:
                uncertainty_range = (prediction_df["Upper_CI"] - prediction_df["Lower_CI"]).mean()
                st.metric("Uncertainty", f"±{uncertainty_range:.1f}")
            
            # Additional prediction details
            with st.expander("📊 Prediction Details"):
                st.write(f"**Method:** {method_used}")
                st.write(f"**Status:** {status}")
                st.write(f"**Data Points Used:** {len(country_data)} days")
                st.write(f"**Prediction Horizon:** {prediction_days} days")
                
                # Show prediction table
                st.dataframe(prediction_df.round(2), use_container_width=True)
                
        else:
            st.error(f"❌ Could not generate prediction for {country}")
            st.write(f"Error: {status}")
            
            # Show available data
            if len(country_data) > 0:
                st.write("**Available Data Sample:**")
                st.dataframe(country_data.tail(10).reset_index(drop=True), use_container_width=True)
    
    # Comparative Forecast Summary
    if successful_predictions > 0:
        st.subheader("📊 Comparative Forecast Summary")
        st.markdown("**Overview of predictions across all countries**")
        
        comparison_data = []
        for country in selected_countries:
            country_data = filtered_df[filtered_df["Country"] == country]
            if len(country_data) > 0:
                # Try to get prediction
                pred_df, _ = predict_future_enhanced(filtered_df, country, 30, 90, use_simple_model)
                if pred_df is not None:
                    comparison_data.append({
                        "Country": country,
                        "Current Cases": country_data["DailyCasesPerMillion"].iloc[-1] if len(country_data) > 0 else 0,
                        "Avg Prediction": pred_df["PredictedCases"].mean(),
                        "Peak Prediction": pred_df["PredictedCases"].max(),
                        "Data Points": len(country_data),
                        "Trend": "↑ Increasing" if pred_df["PredictedCases"].iloc[-1] > pred_df["PredictedCases"].iloc[0] else "↓ Decreasing"
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.dataframe(comparison_df.round(2), use_container_width=True)
            
            # Visual comparison
            fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
            
            countries = [item["Country"] for item in comparison_data]
            predictions = [item["Avg Prediction"] for item in comparison_data]
            
            bars = ax_comp.bar(countries, predictions, color=sns.color_palette("viridis", len(countries)))
            ax_comp.set_title("Average Predicted Cases (Next 30 Days)")
            ax_comp.set_ylabel("Cases per Million")
            ax_comp.set_xlabel("Country")
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax_comp.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig_comp)
    
    else:
        st.warning("""
        **No predictions generated. This could be due to:**
        - Insufficient data for selected countries
        - Data quality issues
        - All-zero case patterns
        
        **Suggestions:**
        - Select countries with more historical data
        - Adjust date range to include more data points
        - Try different prediction methods
        """)
    
    # Prediction Methodology Explanation
    with st.expander("🔍 Prediction Methodology"):
        st.markdown("""
        **Prediction Methods Used:**
        
        1. **Enhanced Model**: Combines trend analysis with statistical methods
           - Requires: 7+ days of data
           - Best for: Countries with sufficient historical data
        
        2. **Trend Model**: Simple trend-based projection
           - Requires: 3+ days of data  
           - Best for: Countries with limited data
        
        3. **Seasonal Model**: Basic weekly pattern detection
           - Requires: 7+ days of data
           - Best for: Detecting weekly cycles
        
        **Confidence Intervals:**
        - Represent prediction uncertainty
        - Wider intervals indicate less reliable predictions
        - Based on data quality and quantity
        
        **Note**: Predictions with limited data should be interpreted with caution.
        """)
# =============================================================================
# TAB 4: PRESCRIPTIVE ANALYSIS
# =============================================================================
with tab4:
    st.header("💡 Prescriptive Analysis: What Should We Do?")
    st.markdown("""
    This section provides actionable insights and recommendations based on the analysis.
    """)
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()
    
    # Chart 9: Risk Assessment
    st.subheader("🎯 Risk Assessment & Recommendations")
    st.markdown("**Multi-factor risk scoring system combining recent cases, trends, and volatility**")
    
    # Calculate risk scores for each country with proper error handling
    risk_data = []
    valid_countries = []
    
    for country in selected_countries:
        country_data = filtered_df[filtered_df["Country"] == country]
        if len(country_data) > 0:
            try:
                # Calculate metrics with safe defaults
                recent_avg = country_data.tail(30)["DailyCasesPerMillion"].mean()
                
                # Handle trend calculation safely
                if len(country_data) > 30:
                    recent_week = country_data.tail(7)["DailyCasesPerMillion"].mean()
                    previous_weeks = country_data.iloc[-30:-7]["DailyCasesPerMillion"].mean()
                    trend = recent_week / previous_weeks if previous_weeks > 0 else 1.0
                else:
                    trend = 1.0  # Default neutral trend
                
                volatility = country_data["DailyCasesPerMillion"].std()
                
                # Ensure no NaN values
                recent_avg = recent_avg if not np.isnan(recent_avg) else 0
                trend = trend if not np.isnan(trend) else 1.0
                volatility = volatility if not np.isnan(volatility) else 0
                
                risk_score = (recent_avg * 0.4 + trend * 0.3 + volatility * 0.3)
                
                # Determine risk level
                if risk_score > 50:
                    risk_level = "🔴 High"
                    recommendation = "Implement strict measures: lockdowns, travel restrictions, mass testing"
                elif risk_score > 20:
                    risk_level = "🟡 Medium"
                    recommendation = "Enhanced monitoring: social distancing, mask mandates, capacity limits"
                else:
                    risk_level = "🟢 Low"
                    recommendation = "Maintain vigilance: testing, contact tracing, public awareness"
                
                risk_data.append({
                    "Country": country,
                    "Risk Score": risk_score,
                    "Risk Level": risk_level,
                    "Recent Average": recent_avg,
                    "Trend Indicator": trend,
                    "Recommendation": recommendation
                })
                valid_countries.append(country)
                
            except Exception as e:
                st.warning(f"Could not calculate risk for {country}: {str(e)}")
                continue
    
    if risk_data:
        # Convert to DataFrame for display
        display_risk_data = []
        for item in risk_data:
            display_risk_data.append({
                "Country": item["Country"],
                "Risk Score": f"{item['Risk Score']:.1f}",
                "Risk Level": item["Risk Level"],
                "Recent Average": f"{item['Recent Average']:.1f}",
                "Trend Indicator": f"{item['Trend Indicator']:.2f}",
                "Recommendation": item["Recommendation"]
            })
        
        risk_df = pd.DataFrame(display_risk_data)
        st.dataframe(risk_df, use_container_width=True)
        
        # Resource Allocation Simulation
        st.subheader("📦 Resource Allocation Simulation")
        st.markdown("**Dynamic resource distribution model based on risk assessment and predicted needs**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            total_resources = st.number_input("Total Available Resources (units)", min_value=1000, max_value=1000000, value=10000, step=1000)
            allocation_method = st.selectbox("Allocation Method", 
                                           ["Based on Current Cases", "Based on Prediction", "Based on Population Risk"])
        
        with col2:
            st.markdown("**Resource Allocation Results**")
            
            # Calculate allocation with proper error handling
            total_risk = sum([item["Risk Score"] for item in risk_data])
            
            if total_risk > 0:  # Ensure we don't divide by zero
                allocation_results = []
                
                for item in risk_data:
                    risk_share = item["Risk Score"] / total_risk
                    allocated_resources = int(total_resources * risk_share)
                    allocation_results.append({
                        "Country": item["Country"],
                        "Allocated Resources": allocated_resources,
                        "Percentage": f"{(risk_share * 100):.1f}%"
                    })
                
                allocation_df = pd.DataFrame(allocation_results)
                st.dataframe(allocation_df, use_container_width=True)
                
                # Resource Allocation Visualization
                st.subheader("📊 Resource Allocation Dashboard")
                st.markdown("**Interactive pie chart showing proportional resource distribution**")
                
                fig_resources, ax_resources = plt.subplots(figsize=(10, 8))
                countries = [item['Country'] for item in allocation_results]
                resources = [item['Allocated Resources'] for item in allocation_results]
                
                wedges, texts, autotexts = ax_resources.pie(resources, labels=countries, autopct='%1.1f%%',
                                                           startangle=90, colors=sns.color_palette("Set3"))
                
                ax_resources.set_title("Proportional Resource Allocation", fontsize=16)
                st.pyplot(fig_resources)
            else:
                st.warning("Cannot calculate resource allocation: Total risk score is zero.")
    
    else:
        st.warning("No risk data available for the selected countries.")
    
    # Intervention Impact Analysis
    st.subheader("🛡️ Intervention Impact Analysis")
    st.markdown("**Cost-benefit analysis of different intervention strategies with effectiveness estimates**")
    
    intervention = st.selectbox("Select Intervention Type", 
                              ["Social Distancing", "Mask Mandates", "Vaccination Campaign", "Travel Restrictions", "Lockdown"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        effectiveness = st.slider(f"Estimated Effectiveness of {intervention} (%)", 0, 100, 70)
    with col2:
        implementation_time = st.slider("Implementation Time (days)", 1, 30, 7)
    with col3:
        compliance_rate = st.slider("Expected Compliance Rate (%)", 0, 100, 80)
    
    # Calculate expected impact
    overall_impact = (effectiveness / 100) * (compliance_rate / 100) * 100
    time_to_effect = implementation_time + 14  # Assuming 2 weeks for effect to show
    
    st.info(f"""
    **Expected Impact Analysis:**
    - Overall effectiveness: **{overall_impact:.1f}%** reduction in transmission
    - Time to see effects: **{time_to_effect} days**
    - Recommended duration: **{max(30, time_to_effect + 14)} days**
    """)
    
    # Action Plan Generator
    st.subheader("📋 Generate Action Plan")
    st.markdown("**Comprehensive strategy development based on analytical insights**")
    
    if st.button("Generate Comprehensive Action Plan"):
        if risk_data:
            high_risk_countries = [item["Country"] for item in risk_data if item["Risk Level"] == "🔴 High"]
            medium_risk_countries = [item["Country"] for item in risk_data if item["Risk Level"] == "🟡 Medium"]
            
            st.success(f"""
            **Comprehensive COVID-19 Action Plan Generated:**
            
            1. **Immediate Actions (0-7 days) for High-Risk Countries {high_risk_countries}:**
               - Implement strict lockdown measures
               - Deploy emergency medical resources
               - Enhance mass testing and contact tracing
            
            2. **Short-term Measures (1-4 weeks) for Medium-Risk Countries {medium_risk_countries}:**
               - Implement social distancing and mask mandates
               - Set up temporary healthcare facilities
               - Launch public awareness campaigns
            
            3. **Medium-term Strategy (1-3 months):**
               - Adjust measures based on predictive analytics
               - Scale successful interventions
               - Prepare for potential new waves
            
            4. **Long-term Preparedness (3+ months):**
               - Build resilient healthcare systems
               - Develop early warning systems
               - Invest in research and development
            """)
        else:
            st.success("""
            **General COVID-19 Action Plan:**
            
            1. **Monitoring Phase (0-7 days):**
               - Enhance surveillance and testing
               - Prepare healthcare infrastructure
               - Public awareness campaigns
            
            2. **Preparedness Phase (1-4 weeks):**
               - Stockpile essential medical supplies
               - Train healthcare workers
               - Develop contingency plans
            
            3. **Prevention Phase (1-3 months):**
               - Implement preventive measures
               - Vaccination campaigns
               - International cooperation
            """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
🎓 Developed by **Anas Hussain** anashussain0311@gmail.com | Department of CS | Data Science Institute  
📍 NED University of Engineering and Technology  
🧪 Dataset: [Our World in Data](https://ourworldindata.org/covid-cases)  
📱 Tools Used: Python, Pandas, Seaborn, Matplotlib, Streamlit, Plotly, Statsmodels  
""")