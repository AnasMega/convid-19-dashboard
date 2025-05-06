import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page Setup
st.set_page_config(page_title="COVID-19 Data Visualization - Anas Hussain", layout="wide")

# Title & Metadata
st.title("📊 COVID-19 Global Dashboard - Data Visualization Project")
st.markdown("""
#### 🧑‍🎓 **Name**: Anas Hussain (DS-017 23/24)
#### 📔 **GIT**: AnasMega http://github.com/anasMega/
#### 📞 **Contact**: anashussain0311@gmailc.com
#### 🏫 **Department**: Computer Science  
#### 🧪 **Institute**: Data Science Institute, NED University of Engineering & Technology  
#### 📚 **Subject**: Data Visualization  
""")

# Horizontal rule
st.markdown("---")

# Load and Cache Data
@st.cache_data
def load_data():
    df = pd.read_csv("daily-new-confirmed-covid-19-cases-per-million-people.csv")
    df.columns = ["Country", "Date", "CasesPerMillion"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# Sidebar for Filters
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
pivot_df = filtered_df.pivot(index="Date", columns="Country", values="CasesPerMillion")

# Key Metrics Section
st.header("📌 Key Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Date Range", f"{(date_range[1] - date_range[0]).days} Days")
with col2:
    st.metric("Countries Selected", len(selected_countries))
with col3:
    st.metric("Data Points", f"{filtered_df.shape[0]:,}")

# Line Chart
st.header("📈 Trend Over Time: Daily COVID-19 Cases Per Million")
fig1, ax1 = plt.subplots(figsize=(14, 6))
sns.lineplot(data=pivot_df, ax=ax1)
ax1.set_title("7-Day Rolling Average", fontsize=16)
ax1.set_ylabel("Cases Per Million")
ax1.set_xlabel("Date")
ax1.grid(True)
st.pyplot(fig1)

# Bar Chart
st.header("📊 Average Daily Cases by Country")
avg_df = filtered_df.groupby("Country")["CasesPerMillion"].mean().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(12, 5))
sns.barplot(x=avg_df.index, y=avg_df.values, palette="coolwarm", ax=ax2)
ax2.set_ylabel("Average Cases Per Million")
ax2.set_title("Comparison Across Countries")
st.pyplot(fig2)

# Heatmap
st.header("🔥 Weekly Heatmap of Cases")
heatmap_df = filtered_df.copy()
heatmap_df["Week"] = heatmap_df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
weekly = heatmap_df.groupby(["Week", "Country"])["CasesPerMillion"].mean().unstack().fillna(0)
fig3, ax3 = plt.subplots(figsize=(14, 6))
sns.heatmap(weekly.T, cmap="YlGnBu", linewidths=0.1, linecolor='gray', ax=ax3)
ax3.set_title("Weekly Average Cases per Million", fontsize=14)
ax3.set_xlabel("Week")
ax3.set_ylabel("Country")
st.pyplot(fig3)

# Raw Data
st.header("📄 Raw Filtered Data")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# Download Button
st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False), "filtered_covid_data.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("""
🎓 Developed by **Anas Hussain** anashussain0311@gmail.com| Department of CS | Data Science Institute  
📍 NED University of Engineering and Technology  
🧬 Dataset: [Our World in Data](https://ourworldindata.org/covid-cases)  
📘 Tools Used: Python, Pandas, Seaborn, Matplotlib, Streamlit  
""")

import plotly.express as px

st.header("📽️ Animated Trend: Daily Cases Over Time")

animated_df = filtered_df.copy()
animated_df['Date'] = pd.to_datetime(animated_df['Date']).dt.strftime('%Y-%m-%d')

fig_anim = px.line(
    animated_df,
    x="Date", y="CasesPerMillion", color="Country",
    title="Animated Daily Cases Per Million",
    animation_frame="Date",
    range_y=[0, animated_df["CasesPerMillion"].max()],
    labels={"CasesPerMillion": "Cases per Million"},
    height=500
)
st.plotly_chart(fig_anim, use_container_width=True)

st.header("📊 Top 5 Countries with Highest Case Spikes")

spike_df = filtered_df.groupby("Country")["CasesPerMillion"].max().sort_values(ascending=False).head(5)
fig_spike, ax_spike = plt.subplots()
sns.barplot(x=spike_df.values, y=spike_df.index, palette="rocket", ax=ax_spike)
ax_spike.set_title("Countries with Highest Daily Cases Spike")
ax_spike.set_xlabel("Max Daily Cases Per Million")
st.pyplot(fig_spike)

st.header("🗺️ World View: Cases on Selected Day")

map_date = st.date_input("Choose a Date for World Map", value=max_date, min_value=min_date, max_value=max_date)

map_df = df[df["Date"] == pd.to_datetime(map_date)]
map_df = map_df.groupby("Country")["CasesPerMillion"].sum().reset_index()

fig_map = px.choropleth(
    map_df,
    locations="Country",
    locationmode="country names",
    color="CasesPerMillion",
    color_continuous_scale="Reds",
    title=f"COVID Cases per Million - {map_date}",
)
st.plotly_chart(fig_map, use_container_width=True)
