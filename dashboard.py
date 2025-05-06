import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config("COVID-19 Dashboard", layout="wide")

st.title("🌍 COVID-19 Cases Dashboard")
st.markdown("Visualize daily new confirmed COVID-19 cases per million people (7-day rolling average).")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("daily-new-confirmed-covid-19-cases-per-million-people.csv")
    df.columns = ["Country", "Date", "CasesPerMillion"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("🔧 Filter Options")

    # Country selector
    all_countries = sorted(df["Country"].unique())
    selected_countries = st.multiselect("Select countries:", all_countries, default=["United States", "India", "Brazil", "Pakistan"])

    # Date range selector
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.slider("Select date range:", min_value=min_date, max_value=max_date, value=(min_date, max_date))

# --- Data Filtering ---
filtered_df = df[
    (df["Country"].isin(selected_countries)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

pivot_df = filtered_df.pivot(index="Date", columns="Country", values="CasesPerMillion")

# --- Stats Section ---
st.subheader("📊 Average Daily Cases per Million")
col1, col2 = st.columns([2, 1])
with col1:
    avg_cases = filtered_df.groupby("Country")["CasesPerMillion"].mean().round(2).reset_index()
    st.dataframe(avg_cases.rename(columns={"CasesPerMillion": "Average Cases"}), use_container_width=True)
with col2:
    st.metric("📅 Data Range", f"{date_range[0]} to {date_range[1]}")
    st.metric("📌 Countries Selected", len(selected_countries))

# --- Line Chart ---
st.subheader("📈 COVID-19 Trend Over Time")
fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=pivot_df, ax=ax)
ax.set_title("Daily New Confirmed COVID-19 Cases per Million People (7-day Avg)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Cases per Million")
ax.grid(True)
ax.legend(title="Country")
st.pyplot(fig)

# --- Download Section ---
st.subheader("📥 Download Filtered Data")
csv = filtered_df.to_csv(index=False)
st.download_button("Download CSV", csv, "filtered_covid_data.csv", "text/csv")

# --- Raw Data Option ---
with st.expander("📂 Show Raw Data Table"):
    st.dataframe(filtered_df, use_container_width=True)
