import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import random  # This is the missing import

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
pivot_df = filtered_df.pivot(index="Date", columns="Country", values="DailyCasesPerMillion")

# Key Metrics
st.header("📌 Key Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Date Range", f"{(date_range[1] - date_range[0]).days} Days")
with col2:
    st.metric("Countries Selected", len(selected_countries))
with col3:
    st.metric("Data Points", f"{filtered_df.shape[0]:,}")

# Line Chart
st.header("📈 Trend Over Time: Daily COVID-19 Cases")
st.subheader("Chart 1: 7-Day Rolling Average of Daily New Cases")
fig1, ax1 = plt.subplots(figsize=(14, 6))
sns.lineplot(data=pivot_df, ax=ax1)
ax1.set_title("7-Day Rolling Average of Daily New Cases (Per Million People)", fontsize=16)
ax1.set_ylabel("Daily Cases per 1 Million People")
ax1.set_xlabel("Date")
ax1.grid(True)
st.pyplot(fig1)

# Bar Chart
st.header("📊 Average Daily Cases by Country")
st.subheader("Chart 2: Average Daily Cases per 1 Million People")
avg_df = filtered_df.groupby("Country")["DailyCasesPerMillion"].mean().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(12, 5))
sns.barplot(x=avg_df.index, y=avg_df.values, palette="coolwarm", ax=ax2)
ax2.set_ylabel("Average Daily Cases per 1 Million People")
ax2.set_title("Average Daily Cases (Normalized by Population)")
st.pyplot(fig2)

# Heatmap
st.header("🔥 Weekly Heatmap of Cases")
st.subheader("Chart 3: Weekly Average of Daily Cases per 1 Million People")
heatmap_df = filtered_df.copy()
heatmap_df["Week"] = heatmap_df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
weekly = heatmap_df.groupby(["Week", "Country"])["DailyCasesPerMillion"].mean().unstack().fillna(0)
fig3, ax3 = plt.subplots(figsize=(14, 6))
sns.heatmap(weekly.T, cmap="YlGnBu", linewidths=0.1, linecolor='gray', ax=ax3)
ax3.set_title("Weekly Average of Daily Cases per 1 Million People", fontsize=14)
ax3.set_xlabel("Week")
ax3.set_ylabel("Country")
st.pyplot(fig3)

# Raw Data Table
st.header("📄 Raw Filtered Data")
st.subheader("Table 1: Filtered COVID-19 Case Data")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# Download Button
st.download_button("📅 Download Filtered Data", filtered_df.to_csv(index=False), "filtered_covid_data.csv", "text/csv")

# Animated Line Chart
st.header("🎬 Animated Trend: Daily Cases Over Time")
st.subheader("Chart 4: Animated Daily New Cases per 1 Million People")
animated_df = filtered_df.copy()
animated_df['Date'] = pd.to_datetime(animated_df['Date']).dt.strftime('%Y-%m-%d')
fig_anim = px.line(
    animated_df,
    x="Date", y="DailyCasesPerMillion", color="Country",
    title="Animated Daily New Cases per 1 Million People",
    animation_frame="Date",
    range_y=[0, animated_df["DailyCasesPerMillion"].max()],
    labels={"DailyCasesPerMillion": "Daily Cases per 1M"},
    height=500
)
st.plotly_chart(fig_anim, use_container_width=True)

# Highest Spike Bar Chart
st.header("📊 Top 5 Countries with Highest Case Spikes")
st.subheader("Chart 5: Countries with Highest Daily New Cases Spike")
spike_df = filtered_df.groupby("Country")["DailyCasesPerMillion"].max().sort_values(ascending=False).head(5)
fig_spike, ax_spike = plt.subplots()
sns.barplot(x=spike_df.values, y=spike_df.index, palette="rocket", ax=ax_spike)
ax_spike.set_title("Countries with Highest Daily New Cases Spike")
ax_spike.set_xlabel("Max Daily Cases per 1 Million People")
st.pyplot(fig_spike)

# Choropleth Map
st.header("🗺️ World View: Cases on Selected Day")
st.subheader("Chart 6: Daily COVID-19 Cases per 1 Million People - World Map")
map_date = st.date_input("Choose a Date for World Map", value=max_date, min_value=min_date, max_value=max_date)
map_df = df[df["Date"] == pd.to_datetime(map_date)]
map_df = map_df.groupby("Country")["DailyCasesPerMillion"].sum().reset_index()
fig_map = px.choropleth(
    map_df,
    locations="Country",
    locationmode="country names",
    color="DailyCasesPerMillion",
    color_continuous_scale="Reds",
    title=f"Daily COVID-19 Cases per 1 Million People - {map_date}",
)
st.plotly_chart(fig_map, use_container_width=True)

# Additional Visualization: Boxplot of Distribution
st.header("📊 Distribution of Daily Cases per Country")
st.subheader("Chart 7: Distribution of Daily Cases per 1 Million (Box Plot)")
fig_box, ax_box = plt.subplots(figsize=(14, 5))
sns.boxplot(data=filtered_df, x="Country", y="DailyCasesPerMillion", palette="Set3", ax=ax_box)
ax_box.set_title("Distribution of Daily Cases per 1 Million (Box Plot)")
ax_box.set_ylabel("Daily Cases per 1 Million")
ax_box.set_xlabel("Country")
st.pyplot(fig_box)

# Prediction Section
st.header("🔮 Predict Future COVID-19 Cases")
st.subheader("Chart 8: COVID-19 Case Predictions")

# User input for prediction
prediction_days = st.slider("How many days would you like to predict?", min_value=1, max_value=30, value=7)

# Function to predict future values using ARIMA
def predict_future(df, country, days):
    country_data = df[df["Country"] == country]
    country_data = country_data.set_index("Date")
    country_data = country_data.resample("D").sum()  # Resample to daily frequency
    country_data = country_data["DailyCasesPerMillion"]

    # Fit ARIMA model
    model = ARIMA(country_data, order=(5, 1, 0))  # You can tune the (p,d,q) parameters
    model_fit = model.fit()

    # Make prediction
    forecast = model_fit.forecast(steps=days)
    forecast_dates = pd.date_range(country_data.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")

    prediction_df = pd.DataFrame({
        "Date": forecast_dates,
        "PredictedCasesPerMillion": forecast
    })

    return prediction_df
 
for country in selected_countries:
    st.subheader(f"🔮 {country} - Prediction for the Next {prediction_days} Days")
    st.subheader(f"Chart 8.{selected_countries.index(country)+1}: {country} - {prediction_days}-Day Forecast")
    prediction_df = predict_future(filtered_df, country, prediction_days)
    
    fig_pred, ax_pred = plt.subplots(figsize=(14, 6))
    sns.lineplot(x=prediction_df["Date"], y=prediction_df["PredictedCasesPerMillion"], ax=ax_pred, label="Predicted", color="red")
    ax_pred.set_title(f"Predicted Daily New COVID-19 Cases per 1 Million People for {country}", fontsize=16)
    ax_pred.set_xlabel("Date")
    ax_pred.set_ylabel("Predicted Cases per 1 Million People")
    ax_pred.grid(True)
    st.pyplot(fig_pred)

# Temperature Data Section
st.header("🌡️ Live Temperature Data (Simulated)")
st.subheader("Chart 9: Simulated Live Temperature Data")

# Function to generate random temperature data
def generate_random_temperature_data(cities):
    data = []
    for city in cities:
        city_data = {
            "City": city,
            "Temperature": round(random.uniform(-10, 40), 2),  # Random temperature between -10 and 40
            "Humidity": random.randint(20, 100),  # Random humidity between 20% and 100%
            "Pressure": random.randint(1000, 1050),  # Random pressure between 1000 and 1050 hPa
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        data.append(city_data)
    return pd.DataFrame(data)

cities = ["New York", "London", "Karachi", "Delhi", "Sydney"]
live_temp_data = generate_random_temperature_data(cities)

# Temperature Data Filters
with st.sidebar:
    st.title("🌍 Temperature Data Filter Options")
    selected_cities = st.multiselect("Select Cities", cities, default=cities)
    date_range = st.slider("Select Date Range", min_value=0, max_value=7, value=7)

filtered_live_data = live_temp_data[live_temp_data["City"].isin(selected_cities)]

# Temperature Metrics
st.header("🌡️ Live Temperature Data - Key Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Cities Selected", len(selected_cities))
with col2:
    st.metric("Live Data Points", f"{filtered_live_data.shape[0]:,}")
with col3:
    st.metric("Latest Update", filtered_live_data["Date"].iloc[0])

# Temperature Line Chart
st.header("📈 Live Temperature Trend")
st.subheader("Chart 9.1: Live Temperature Trend (°C)")
fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=filtered_live_data, x="City", y="Temperature", hue="City", marker='o', ax=ax)
ax.set_title("Live Temperature Trend (°C)", fontsize=16)
ax.set_xlabel("City")
ax.set_ylabel("Temperature (°C)")
ax.grid(True)
st.pyplot(fig)

# Temperature Bar Chart
st.header("📊 Temperature Comparison Across Cities")
st.subheader("Chart 9.2: Temperature Comparison Across Cities")
fig_temp, ax_temp = plt.subplots(figsize=(12, 5))                           
sns.barplot(x="City", y="Temperature", data=filtered_live_data, palette="coolwarm", ax=ax_temp)
ax_temp.set_ylabel("Temperature (°C)")
ax_temp.set_title("Temperature Comparison Across Cities")
st.pyplot(fig_temp)

# Temperature Data Table
st.header("📄 Raw Filtered Live Temperature Data")
st.subheader("Table 2: Simulated Live Temperature Data")
st.dataframe(filtered_live_data, use_container_width=True)

st.download_button("📅 Download Simulated Live Temperature Data", filtered_live_data.to_csv(index=False), "simulated_live_temperature_data.csv", "text/csv")

st.markdown("---")
st.markdown("""
🎓 Developed by **Anas Hussain** anashussain0311@gmail.com | Department of CS | Data Science Institute  
📍 NED University of Engineering and Technology  
🧪 Dataset: [Our World in Data](https://ourworldindata.org/covid-cases)  
📱 Tools Used: Python, Pandas, Seaborn, Matplotlib, Streamlit, Plotly  
""")