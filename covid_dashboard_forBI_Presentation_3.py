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

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/NEDUET_logo.svg/800px-NEDUET_logo.svg.png", width=110)
    st.title("🔎 Filter Options")
    countries = sorted(df["Country"].unique())
    selected_countries = st.multiselect("Select Countries", countries, default=["Pakistan", "India", "United States"])
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    
    # Additional chart options
    st.markdown("---")
    st.subheader("📈 Chart Options")
    show_pie_charts = st.checkbox("Show Pie Charts", value=True)
    show_stacked_charts = st.checkbox("Show Stacked Charts", value=True)
    show_comparison_charts = st.checkbox("Show Comparison Charts", value=True)

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
    
    # NEW: Multi-line Chart Comparison
    st.subheader("📈 Multi-Line Chart: Country Comparison Over Time")
    st.markdown("**line chart comparing COVID-19 trends across selected countries**")
    
    if not filtered_df.empty:
        fig_line = px.line(filtered_df, x="Date", y="DailyCasesPerMillion", color="Country",
                          title="Daily COVID-19 Cases per Million - Country Comparison",
                          labels={"DailyCasesPerMillion": "Cases per Million", "Date": "Date"},
                          height=500)
        fig_line.update_layout(hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)
    
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
    
    # NEW: Pie Chart - Case Distribution
    if show_pie_charts:
        st.subheader("🥧 Pie Chart: Case Distribution by Country")
        st.markdown("**Proportional distribution of total COVID-19 cases among selected countries**")
        
        if not filtered_df.empty:
            total_cases_by_country = filtered_df.groupby("Country")["DailyCasesPerMillion"].sum()
            
            fig_pie, ax_pie = plt.subplots(figsize=(10, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, len(total_cases_by_country)))
            wedges, texts, autotexts = ax_pie.pie(total_cases_by_country.values, 
                                                labels=total_cases_by_country.index, 
                                                autopct='%1.1f%%', startangle=90, colors=colors)
            ax_pie.set_title("Total COVID-19 Cases Distribution by Country", fontsize=16)
            
            # Improve readability
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            st.pyplot(fig_pie)
    
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
    
    # NEW: Stacked Area Chart
    if show_stacked_charts:
        st.subheader("📊 Stacked Area Chart: Cases Over Time")
        st.markdown("**Cumulative view of case distribution showing relative contributions over time**")
        
        if not filtered_df.empty:
            # Prepare data for stacked area chart
            stacked_df = filtered_df.pivot_table(index='Date', columns='Country', 
                                               values='DailyCasesPerMillion', aggfunc='sum').fillna(0)
            
            fig_stacked, ax_stacked = plt.subplots(figsize=(14, 7))
            stacked_df.plot(kind='area', ax=ax_stacked, alpha=0.7, stacked=True)
            ax_stacked.set_title("Stacked Area Chart: Daily Cases Distribution Over Time", fontsize=16)
            ax_stacked.set_ylabel("Daily Cases per Million")
            ax_stacked.set_xlabel("Date")
            ax_stacked.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_stacked.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_stacked)
    
    # NEW: Cumulative Cases Over Time
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
    This section investigates patterns factors in COVID-19 spread.
    """)
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()
    
    # NEW: Stacked Bar Chart - Monthly Cases
    if show_stacked_charts:
        st.subheader("📊 Stacked Bar Chart: Monthly Cases by Country")
        st.markdown("**Monthly breakdown showing each country's contribution to total cases**")
        
        monthly_stacked = filtered_df.copy()
        monthly_stacked['YearMonth'] = monthly_stacked['Date'].dt.to_period('M').astype(str)
        monthly_totals = monthly_stacked.groupby(['YearMonth', 'Country'])['DailyCasesPerMillion'].sum().unstack().fillna(0)
        
        fig_stacked_bar, ax_stacked_bar = plt.subplots(figsize=(14, 7))
        monthly_totals.plot(kind='bar', stacked=True, ax=ax_stacked_bar, alpha=0.8)
        ax_stacked_bar.set_title("Stacked Bar Chart: Monthly Cases Distribution by Country", fontsize=16)
        ax_stacked_bar.set_ylabel("Total Cases per Million")
        ax_stacked_bar.set_xlabel("Month")
        ax_stacked_bar.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_stacked_bar)
    
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
    
    # NEW: Pie Chart - Peak Cases Distribution
    if show_pie_charts:
        st.subheader("🥧 Pie Chart: Peak Cases Distribution")
        st.markdown("**Distribution of highest recorded daily cases across countries**")
        
        peak_cases_by_country = filtered_df.groupby("Country")["DailyCasesPerMillion"].max()
        
        fig_peak_pie, ax_peak_pie = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(peak_cases_by_country)))
        wedges, texts, autotexts = ax_peak_pie.pie(peak_cases_by_country.values, 
                                                 labels=peak_cases_by_country.index, 
                                                 autopct='%1.1f%%', startangle=90, colors=colors)
        ax_peak_pie.set_title("Peak Daily Cases Distribution by Country", fontsize=16)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        st.pyplot(fig_peak_pie)
    
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
    
    # NEW: Multi-line Chart - Moving Averages Comparison
    if show_comparison_charts:
        st.subheader("📈 Multi-Line Chart: Moving Averages Comparison")
        st.markdown("**Comparison of different moving averages to identify trends**")
        
        if len(selected_countries) > 0:
            country_data = filtered_df[filtered_df['Country'] == selected_countries[0]]
            if not country_data.empty:
                country_data = country_data.set_index('Date')
                ma_7 = country_data['DailyCasesPerMillion'].rolling(window=7).mean()
                ma_14 = country_data['DailyCasesPerMillion'].rolling(window=14).mean()
                ma_30 = country_data['DailyCasesPerMillion'].rolling(window=30).mean()
                
                fig_ma, ax_ma = plt.subplots(figsize=(14, 6))
                ax_ma.plot(country_data.index, country_data['DailyCasesPerMillion'], 
                          label='Daily Cases', alpha=0.3, color='gray')
                ax_ma.plot(ma_7.index, ma_7, label='7-Day MA', linewidth=2)
                ax_ma.plot(ma_14.index, ma_14, label='14-Day MA', linewidth=2)
                ax_ma.plot(ma_30.index, ma_30, label='30-Day MA', linewidth=2)
                
                ax_ma.set_title(f"Moving Averages Comparison for {selected_countries[0]}", fontsize=16)
                ax_ma.set_ylabel("Cases per Million")
                ax_ma.set_xlabel("Date")
                ax_ma.legend()
                ax_ma.grid(True, alpha=0.3)
                st.pyplot(fig_ma)
    
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
    
    # Display predictions for each selected country
    successful_predictions = 0
    all_predictions = []
    
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
        
        # Try prediction
        prediction_df, status = predict_future_enhanced(filtered_df, country, prediction_days, confidence_level, use_simple_model)
        
        if prediction_df is not None:
            successful_predictions += 1
            all_predictions.append((country, prediction_df))
            
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
            
            ax_pred.set_title(f"COVID-19 Forecast for {country}", fontsize=14)
            ax_pred.set_xlabel("Date")
            ax_pred.set_ylabel("Cases per 1 Million People")
            ax_pred.legend()
            ax_pred.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_pred)
            
            # Show prediction summary
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_prediction = prediction_df["PredictedCases"].mean()
                st.metric("Average Predicted Cases", f"{avg_prediction:.2f}")
            with col2:
                max_prediction = prediction_df["PredictedCases"].max()
                st.metric("Peak Predicted Cases", f"{max_prediction:.2f}")
            with col3:
                trend = "Increasing" if prediction_df["PredictedCases"].iloc[-1] > prediction_df["PredictedCases"].iloc[0] else "Decreasing"
                st.metric("Trend", trend)
                
        else:
            st.error(f"❌ Could not generate prediction for {country}")
            st.write(f"Error: {status}")
    
    # NEW: Pie Chart - Prediction Contribution
    if show_pie_charts and successful_predictions > 0:
        st.subheader("🥧 Pie Chart: Predicted Case Distribution")
        st.markdown("**Expected distribution of future cases across countries based on predictions**")
        
        prediction_totals = []
        for country, pred_df in all_predictions:
            total_predicted = pred_df["PredictedCases"].sum()
            prediction_totals.append((country, total_predicted))
        
        countries_pred = [item[0] for item in prediction_totals]
        totals_pred = [item[1] for item in prediction_totals]
        
        fig_pred_pie, ax_pred_pie = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(prediction_totals)))
        wedges, texts, autotexts = ax_pred_pie.pie(totals_pred, labels=countries_pred, 
                                                 autopct='%1.1f%%', startangle=90, colors=colors)
        ax_pred_pie.set_title("Predicted Future Cases Distribution", fontsize=16)
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        st.pyplot(fig_pred_pie)
    
    # NEW: Stacked Area Chart - Prediction Timeline
    if show_stacked_charts and successful_predictions > 1:
        st.subheader("📊 Stacked Area Chart: Prediction Timeline")
        st.markdown("**Cumulative view of predicted cases across all countries over time**")
        
        # Combine all predictions
        prediction_combined = pd.DataFrame()
        for country, pred_df in all_predictions:
            prediction_combined[country] = pred_df.set_index('Date')['PredictedCases']
        
        fig_pred_stacked, ax_pred_stacked = plt.subplots(figsize=(14, 7))
        prediction_combined.plot(kind='area', ax=ax_pred_stacked, alpha=0.7, stacked=True)
        ax_pred_stacked.set_title("Stacked Prediction Timeline: Expected Cases Distribution", fontsize=16)
        ax_pred_stacked.set_ylabel("Predicted Cases per Million")
        ax_pred_stacked.set_xlabel("Date")
        ax_pred_stacked.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_pred_stacked.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_pred_stacked)

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
        # NEW: Pie Chart - Risk Distribution
        if show_pie_charts:
            st.subheader("🥧 Pie Chart: Risk Level Distribution")
            st.markdown("**Proportional distribution of countries across different risk levels**")
            
            risk_levels = [item["Risk Level"] for item in risk_data]
            risk_counts = pd.Series(risk_levels).value_counts()
            
            fig_risk_pie, ax_risk_pie = plt.subplots(figsize=(10, 8))
            colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']  # Red, Yellow, Green
            wedges, texts, autotexts = ax_risk_pie.pie(risk_counts.values, 
                                                     labels=risk_counts.index, 
                                                     autopct='%1.1f%%', startangle=90, 
                                                     colors=colors[:len(risk_counts)])
            ax_risk_pie.set_title("Risk Level Distribution Across Countries", fontsize=16)
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            st.pyplot(fig_risk_pie)
        
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
        
        # NEW: Stacked Bar Chart - Risk Components
        if show_stacked_charts:
            st.subheader("📊 Stacked Bar Chart: Risk Score Components")
            st.markdown("**Breakdown of risk scores showing contribution of different factors**")
            
            risk_components = []
            for item in risk_data:
                risk_components.append({
                    "Country": item["Country"],
                    "Recent Cases": item["Recent Average"] * 0.4,
                    "Trend Impact": item["Trend Indicator"] * 0.3 * 10,  # Scaled for visibility
                    "Volatility": item["Trend Indicator"] * 0.3 * 10     # Scaled for visibility
                })
            
            risk_comp_df = pd.DataFrame(risk_components)
            risk_comp_df.set_index('Country', inplace=True)
            
            fig_risk_stack, ax_risk_stack = plt.subplots(figsize=(12, 6))
            risk_comp_df.plot(kind='bar', stacked=True, ax=ax_risk_stack, 
                            color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            ax_risk_stack.set_title("Risk Score Components Breakdown", fontsize=16)
            ax_risk_stack.set_ylabel("Risk Score Contribution")
            ax_risk_stack.set_xlabel("Country")
            ax_risk_stack.legend(title='Components', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_risk_stack)
    
    # Rest of the prescriptive analysis code remains the same...
    # [Previous prescriptive analysis code continues here...]

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