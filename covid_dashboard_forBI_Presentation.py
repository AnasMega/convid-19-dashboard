import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
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
    
    # Line Chart
    st.subheader("📈 Trend Over Time: Daily COVID-19 Cases")
    pivot_df = filtered_df.pivot(index="Date", columns="Country", values="DailyCasesPerMillion")
    rolling_df = pivot_df.rolling(window=7).mean()
    
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    for country in selected_countries:
        ax1.plot(rolling_df.index, rolling_df[country], label=country, linewidth=2)
    ax1.set_title("7-Day Rolling Average of Daily New Cases (Per Million People)", fontsize=16)
    ax1.set_ylabel("Daily Cases per 1 Million People")
    ax1.set_xlabel("Date")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    # Bar Chart - Average Cases
    st.subheader("📊 Average Daily Cases by Country")
    avg_df = filtered_df.groupby("Country")["DailyCasesPerMillion"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.barplot(x=avg_df.index, y=avg_df.values, palette="viridis", ax=ax2)
    ax2.set_ylabel("Average Daily Cases per 1 Million People")
    ax2.set_title("Average Daily Cases (Normalized by Population)")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    
    # Data Table
    st.subheader("📄 Raw Data Overview")
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# =============================================================================
# TAB 2: DIAGNOSTIC ANALYSIS
# =============================================================================
with tab2:
    st.header("🔍 Diagnostic Analysis: Why Did It Happen?")
    st.markdown("""
    This section investigates patterns, correlations, and underlying factors in COVID-19 spread.
    """)
    
    # Heatmap
    st.subheader("🔥 Weekly Heatmap of Cases")
    heatmap_df = filtered_df.copy()
    heatmap_df["Week"] = heatmap_df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = heatmap_df.groupby(["Week", "Country"])["DailyCasesPerMillion"].mean().unstack().fillna(0)
    
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
        # Boxplot
        st.markdown("**Box Plot: Case Distribution by Country**")
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_df, x="Country", y="DailyCasesPerMillion", palette="Set3", ax=ax_box)
        ax_box.set_title("Distribution of Daily Cases per 1 Million")
        ax_box.set_ylabel("Daily Cases per 1 Million")
        ax_box.set_xlabel("Country")
        plt.xticks(rotation=45)
        st.pyplot(fig_box)
    
    with col2:
        # Violin Plot
        st.markdown("**Violin Plot: Density Distribution**")
        fig_violin, ax_violin = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=filtered_df, x="Country", y="DailyCasesPerMillion", palette="pastel", ax=ax_violin)
        ax_violin.set_title("Density Distribution of Daily Cases")
        ax_violin.set_ylabel("Daily Cases per 1 Million")
        ax_violin.set_xlabel("Country")
        plt.xticks(rotation=45)
        st.pyplot(fig_violin)
    
    # Correlation Analysis
    st.subheader("🔗 Correlation Analysis Between Countries")
    correlation_df = pivot_df.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_df, annot=True, cmap="coolwarm", center=0, ax=ax_corr)
    ax_corr.set_title("Correlation Matrix of COVID-19 Cases Between Countries")
    st.pyplot(fig_corr)
    
    st.markdown("""
    **Insights from Diagnostic Analysis:**
    - Heatmap shows seasonal patterns and outbreak waves
    - Box plots reveal variability and outlier patterns
    - Correlation matrix indicates synchronized spread patterns between countries
    """)

# =============================================================================
# TAB 3: PREDICTIVE ANALYSIS
# =============================================================================
with tab3:
    st.header("🔮 Predictive Analysis: What Will Happen?")
    st.markdown("""
    This section uses statistical models to forecast future COVID-19 trends.
    """)
    
    # Prediction Configuration
    col1, col2 = st.columns([1, 3])
    with col1:
        prediction_days = st.slider("Prediction Horizon (days)", min_value=7, max_value=90, value=30)
        confidence_level = st.slider("Confidence Level (%)", min_value=80, max_value=95, value=90)
    
    # Enhanced prediction function with error handling
    def predict_future_enhanced(df, country, days, confidence=90):
        try:
            country_data = df[df["Country"] == country].copy()
            if len(country_data) < 30:
                return None, "Insufficient data for reliable prediction"
                
            country_data = country_data.set_index("Date")
            country_data = country_data.resample("D").mean().fillna(0)
            cases_data = country_data["DailyCasesPerMillion"]
            
            # Fit ARIMA model
            model = ARIMA(cases_data, order=(5, 1, 0))
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
            
            return prediction_df, "Success"
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    # Display predictions for each selected country
    for country in selected_countries:
        st.subheader(f"📈 {country} - {prediction_days}-Day Forecast")
        
        prediction_df, status = predict_future_enhanced(filtered_df, country, prediction_days, confidence_level)
        
        if status == "Success":
            # Create visualization
            fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            historical_data = filtered_df[filtered_df["Country"] == country]
            ax_pred.plot(historical_data["Date"], historical_data["DailyCasesPerMillion"], 
                        label="Historical Data", color="blue", linewidth=2)
            
            # Plot prediction
            ax_pred.plot(prediction_df["Date"], prediction_df["PredictedCases"], 
                        label="Predicted", color="red", linewidth=2, linestyle="--")
            
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
            st.warning(f"Could not generate prediction for {country}: {status}")
    
    # Comparative Forecast
    st.subheader("📊 Comparative Forecast Analysis")
    comparison_data = []
    for country in selected_countries:
        prediction_df, status = predict_future_enhanced(filtered_df, country, 30, 90)
        if status == "Success":
            avg_pred = prediction_df["PredictedCases"].mean()
            comparison_data.append({"Country": country, "AveragePredictedCases": avg_pred})
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
        sns.barplot(data=comparison_df, x="Country", y="AveragePredictedCases", palette="magma", ax=ax_comp)
        ax_comp.set_title("Comparative 30-Day Forecast Averages")
        ax_comp.set_ylabel("Average Predicted Cases per Million")
        plt.xticks(rotation=45)
        st.pyplot(fig_comp)

# =============================================================================
# TAB 4: PRESCRIPTIVE ANALYSIS
# =============================================================================
with tab4:
    st.header("💡 Prescriptive Analysis: What Should We Do?")
    st.markdown("""
    This section provides actionable insights and recommendations based on the analysis.
    """)
    
    # Risk Assessment
    st.subheader("🎯 Risk Assessment & Recommendations")
    
    # Calculate risk scores for each country
    risk_data = []
    for country in selected_countries:
        country_data = filtered_df[filtered_df["Country"] == country]
        if len(country_data) > 0:
            recent_avg = country_data.tail(30)["DailyCasesPerMillion"].mean()
            trend = country_data.tail(7)["DailyCasesPerMillion"].mean() / country_data.iloc[-30:-7]["DailyCasesPerMillion"].mean() if len(country_data) > 30 else 1
            volatility = country_data["DailyCasesPerMillion"].std()
            
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
                "Risk Score": f"{risk_score:.1f}",
                "Risk Level": risk_level,
                "Recent Average": f"{recent_avg:.1f}",
                "Trend Indicator": f"{trend:.2f}",
                "Recommendation": recommendation
            })
    
    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True)
    
    # Resource Allocation Simulation
    st.subheader("📦 Resource Allocation Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_resources = st.number_input("Total Available Resources (units)", min_value=1000, max_value=1000000, value=10000, step=1000)
        allocation_method = st.selectbox("Allocation Method", 
                                       ["Based on Current Cases", "Based on Prediction", "Based on Population Risk"])
    
    with col2:
        st.markdown("**Resource Allocation Results**")
        if risk_data:
            # Simple allocation logic
            total_risk = sum([float(item["Risk Score"]) for item in risk_data])
            allocation_results = []
            
            for item in risk_data:
                risk_share = float(item["Risk Score"]) / total_risk
                allocated_resources = int(total_resources * risk_share)
                allocation_results.append({
                    "Country": item["Country"],
                    "Allocated Resources": allocated_resources,
                    "Percentage": f"{(risk_share * 100):.1f}%"
                })
            
            allocation_df = pd.DataFrame(allocation_results)
            st.dataframe(allocation_df, use_container_width=True)
    
    # Intervention Impact Analysis
    st.subheader("🛡️ Intervention Impact Analysis")
    
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
    
    if st.button("Generate Comprehensive Action Plan"):
        st.success("""
        **Comprehensive COVID-19 Action Plan Generated:**
        
        1. **Immediate Actions (0-7 days):**
           - Enhance testing and contact tracing
           - Implement targeted communication campaigns
           - Prepare healthcare infrastructure
        
        2. **Short-term Measures (1-4 weeks):**
           - Deploy resources based on risk assessment
           - Implement selected interventions
           - Monitor compliance and effectiveness
        
        3. **Medium-term Strategy (1-3 months):**
           - Adjust measures based on predictive analytics
           - Scale successful interventions
           - Prepare for potential new waves
        
        4. **Long-term Preparedness (3+ months):**
           - Build resilient healthcare systems
           - Develop early warning systems
           - Invest in research and development
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