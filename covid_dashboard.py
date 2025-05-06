import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="🌍 COVID-19 Global Dashboard", layout="wide")
st.title("🦠 Global COVID-19 Dashboard")
st.markdown("Visualizing **daily new confirmed COVID-19 cases per million** (7-day rolling average) using official datasets.")

# --- Load and Cache Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("daily-new-confirmed-covid-19-cases-per-million-people.csv")
    df.columns = ["Country", "Date", "CasesPerMillion"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# --- Sidebar Filters ---
with st.sidebar:
    st.image("https://www.who.int/Images/default-source/health-topics/coronavirus/coronavirus-getty.jpg", use_column_width=True)
    st.header("📊 Filter Options")

    countries = sorted(df["Country"].unique())
    selected_countries = st.multiselect("Select Countries", countries, default=["United States", "India", "Pakistan", "Brazil"])

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

# --- Filtered Data ---
filtered_df = df[
    (df["Country"].isin(selected_countries)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

pivot_df = filtered_df.pivot(index="Date", columns="Country", values="CasesPerMillion")

# --- Metrics ---
st.subheader("📌 Key Stats")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📅 Duration", f"{(date_range[1] - date_range[0]).days} Days")
with col2:
    st.metric("🌎 Countries Selected", len(selected_countries))
with col3:
    total_points = filtered_df.shape[0]
    st.metric("📈 Data Points", f"{total_points:,}")

# --- Line Chart (Trend Over Time) ---
st.subheader("📈 Trend: Daily COVID-19 Cases Per Million")

fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=pivot_df, ax=ax)
ax.set_title("7-Day Rolling Avg: Daily COVID-19 Cases Per Million People", fontsize=16)
ax.set_ylabel("Cases Per Million")
ax.set_xlabel("Date")
ax.grid(True)
st.pyplot(fig)

# --- Average Cases Bar Chart ---
st.subheader("📊 Average Daily Cases (Bar Chart)")
avg_df = filtered_df.groupby("Country")["CasesPerMillion"].mean().sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(12, 5))
sns.barplot(x=avg_df.index, y=avg_df.values, palette="coolwarm", ax=ax2)
ax2.set_ylabel("Average Daily Cases Per Million")
ax2.set_title("Average Daily Cases by Country", fontsize=14)
st.pyplot(fig2)

# --- Heatmap by Country (Optional Visual Highlight) ---
st.subheader("🔥 Weekly Heatmap of Cases (Experimental)")

# Prepare weekly heatmap data
heatmap_df = filtered_df.copy()
heatmap_df["Week"] = heatmap_df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
weekly = heatmap_df.groupby(["Week", "Country"])["CasesPerMillion"].mean().unstack().fillna(0)

fig3, ax3 = plt.subplots(figsize=(14, 6))
sns.heatmap(weekly.T, cmap="YlOrRd", linewidths=0.1, linecolor='gray', ax=ax3)
ax3.set_title("Weekly Average COVID-19 Cases Per Million (Heatmap)", fontsize=14)
ax3.set_xlabel("Week")
ax3.set_ylabel("Country")
st.pyplot(fig3)

# --- Raw Data Table ---
st.subheader("🧾 Raw Filtered Data")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# --- Download CSV ---
st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False), "filtered_covid_data.csv", "text/csv")

# --- Footer ---
st.markdown("---")
st.markdown("🔬 Data Source: [Our World in Data](https://ourworldindata.org/covid-cases) | Developed with ❤️ using Streamlit and Python")
