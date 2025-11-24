import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import random
import yfinance as yf

st.set_page_config(page_title="COVID-19 & PSX Dashboard - Anas Hussain", layout="wide")

st.title("📊 COVID-19 Global & Pakistan Stock Exchange Dashboard")
st.markdown("""
#### 🧓‍♂️ **Name**: Anas Hussain (DS-017 23/24)
#### 📔 **GIT**: AnasMega https://github.com/AnasMega/convid-19-dashboard
#### 📞 **Contact**: anashussain0311@gmail.com, Anas.pg4000382@cloud.neduet.edu.pk
#### 🏫 **Department**: Computer Science  
#### � **Institute**: Data Science Institute, NED University of Engineering & Technology  
#### 📚 **Subject**: Data Visualization  
""")

st.markdown("---")

# Tab system for different sections
tab1, tab2, tab3 = st.tabs(["COVID-19 Dashboard", "Pakistan Stock Exchange", "Combined Analysis"])

with tab1:
    # COVID-19 Dashboard Content
    st.header("🌍 COVID-19 Global Dashboard")
    
    @st.cache_data
    def load_covid_data():
        df = pd.read_csv("daily-new-confirmed-covid-19-cases-per-million-people.csv")
        df.columns = ["Country", "Date", "DailyCasesPerMillion"]
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    covid_df = load_covid_data()

    # Sidebar Filters for COVID-19
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/NEDUET_logo.svg/800px-NEDUET_logo.svg.png", width=110)
        st.title("🔎 COVID-19 Filter Options")
        countries = sorted(covid_df["Country"].unique())
        selected_countries = st.multiselect("Select Countries", countries, default=["Pakistan", "India", "United States"])
        min_date = covid_df["Date"].min().date()
        max_date = covid_df["Date"].max().date()
        date_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

    # Filter COVID Data
    filtered_covid_df = covid_df[
        (covid_df["Country"].isin(selected_countries)) &
        (covid_df["Date"] >= pd.to_datetime(date_range[0])) &
        (covid_df["Date"] <= pd.to_datetime(date_range[1]))
    ]
    pivot_covid_df = filtered_covid_df.pivot(index="Date", columns="Country", values="DailyCasesPerMillion")

    # COVID Key Metrics
    st.header("📌 COVID-19 Key Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Date Range", f"{(date_range[1] - date_range[0]).days} Days")
    with col2:
        st.metric("Countries Selected", len(selected_countries))
    with col3:
        st.metric("Data Points", f"{filtered_covid_df.shape[0]:,}")

    # COVID Line Chart
    st.header("📈 COVID-19 Trend Over Time")
    st.subheader("Chart 1: 7-Day Rolling Average of Daily New Cases")
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=pivot_covid_df, ax=ax1)
    ax1.set_title("7-Day Rolling Average of Daily New Cases (Per Million People)", fontsize=16)
    ax1.set_ylabel("Daily Cases per 1 Million People")
    ax1.set_xlabel("Date")
    ax1.grid(True)
    st.pyplot(fig1)

    # COVID Bar Chart
    st.header("📊 Average Daily COVID-19 Cases by Country")
    st.subheader("Chart 2: Average Daily Cases per 1 Million People")
    avg_covid_df = filtered_covid_df.groupby("Country")["DailyCasesPerMillion"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.barplot(x=avg_covid_df.index, y=avg_covid_df.values, palette="coolwarm", ax=ax2)
    ax2.set_ylabel("Average Daily Cases per 1 Million People")
    ax2.set_title("Average Daily Cases (Normalized by Population)")
    st.pyplot(fig2)

    # COVID Heatmap
    st.header("🔥 Weekly Heatmap of COVID-19 Cases")
    st.subheader("Chart 3: Weekly Average of Daily Cases per 1 Million People")
    heatmap_covid_df = filtered_covid_df.copy()
    heatmap_covid_df["Week"] = heatmap_covid_df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly_covid = heatmap_covid_df.groupby(["Week", "Country"])["DailyCasesPerMillion"].mean().unstack().fillna(0)
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    sns.heatmap(weekly_covid.T, cmap="YlGnBu", linewidths=0.1, linecolor='gray', ax=ax3)
    ax3.set_title("Weekly Average of Daily Cases per 1 Million People", fontsize=14)
    ax3.set_xlabel("Week")
    ax3.set_ylabel("Country")
    st.pyplot(fig3)

    # COVID Raw Data Table
    st.header("📄 Raw Filtered COVID-19 Data")
    st.subheader("Table 1: Filtered COVID-19 Case Data")
    st.dataframe(filtered_covid_df.reset_index(drop=True), use_container_width=True)

    # COVID Download Button
    st.download_button("📅 Download Filtered COVID-19 Data", filtered_covid_df.to_csv(index=False), "filtered_covid_data.csv", "text/csv")

    # COVID Animated Line Chart
    st.header("🎬 Animated COVID-19 Trend")
    st.subheader("Chart 4: Animated Daily New Cases per 1 Million People")
    animated_covid_df = filtered_covid_df.copy()
    animated_covid_df['Date'] = pd.to_datetime(animated_covid_df['Date']).dt.strftime('%Y-%m-%d')
    fig_anim = px.line(
        animated_covid_df,
        x="Date", y="DailyCasesPerMillion", color="Country",
        title="Animated Daily New Cases per 1 Million People",
        animation_frame="Date",
        range_y=[0, animated_covid_df["DailyCasesPerMillion"].max()],
        labels={"DailyCasesPerMillion": "Daily Cases per 1M"},
        height=500
    )
    st.plotly_chart(fig_anim, use_container_width=True)

    # COVID Highest Spike Bar Chart
    st.header("📊 Top 5 Countries with Highest COVID-19 Case Spikes")
    st.subheader("Chart 5: Countries with Highest Daily New Cases Spike")
    spike_covid_df = filtered_covid_df.groupby("Country")["DailyCasesPerMillion"].max().sort_values(ascending=False).head(5)
    fig_spike, ax_spike = plt.subplots()
    sns.barplot(x=spike_covid_df.values, y=spike_covid_df.index, palette="rocket", ax=ax_spike)
    ax_spike.set_title("Countries with Highest Daily New Cases Spike")
    ax_spike.set_xlabel("Max Daily Cases per 1 Million People")
    st.pyplot(fig_spike)

    # COVID Choropleth Map
    st.header("🗺️ World View: COVID-19 Cases on Selected Day")
    st.subheader("Chart 6: Daily COVID-19 Cases per 1 Million People - World Map")
    map_date = st.date_input("Choose a Date for World Map", value=max_date, min_value=min_date, max_value=max_date)
    map_covid_df = covid_df[covid_df["Date"] == pd.to_datetime(map_date)]
    map_covid_df = map_covid_df.groupby("Country")["DailyCasesPerMillion"].sum().reset_index()
    fig_map = px.choropleth(
        map_covid_df,
        locations="Country",
        locationmode="country names",
        color="DailyCasesPerMillion",
        color_continuous_scale="Reds",
        title=f"Daily COVID-19 Cases per 1 Million People - {map_date}",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # COVID Boxplot of Distribution
    st.header("📊 Distribution of Daily COVID-19 Cases per Country")
    st.subheader("Chart 7: Distribution of Daily Cases per 1 Million (Box Plot)")
    fig_box, ax_box = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=filtered_covid_df, x="Country", y="DailyCasesPerMillion", palette="Set3", ax=ax_box)
    ax_box.set_title("Distribution of Daily Cases per 1 Million (Box Plot)")
    ax_box.set_ylabel("Daily Cases per 1 Million")
    ax_box.set_xlabel("Country")
    st.pyplot(fig_box)

    # COVID Prediction Section
    st.header("🔮 Predict Future COVID-19 Cases")
    st.subheader("Chart 8: COVID-19 Case Predictions")
    prediction_days = st.slider("How many days would you like to predict?", min_value=1, max_value=30, value=7)
    
    def predict_future_covid(df, country, days):
        country_data = df[df["Country"] == country]
        country_data = country_data.set_index("Date")
        country_data = country_data.resample("D").sum()
        country_data = country_data["DailyCasesPerMillion"]

        model = ARIMA(country_data, order=(5, 1, 0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=days)
        forecast_dates = pd.date_range(country_data.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")

        prediction_df = pd.DataFrame({
            "Date": forecast_dates,
            "PredictedCasesPerMillion": forecast
        })

        return prediction_df
     
    for country in selected_countries:
        st.subheader(f"🔮 {country} - Prediction for the Next {prediction_days} Days")
        prediction_covid_df = predict_future_covid(filtered_covid_df, country, prediction_days)
        
        fig_pred, ax_pred = plt.subplots(figsize=(14, 6))
        sns.lineplot(x=prediction_covid_df["Date"], y=prediction_covid_df["PredictedCasesPerMillion"], ax=ax_pred, label="Predicted", color="red")
        ax_pred.set_title(f"Predicted Daily New COVID-19 Cases per 1 Million People for {country}", fontsize=16)
        ax_pred.set_xlabel("Date")
        ax_pred.set_ylabel("Predicted Cases per 1 Million People")
        ax_pred.grid(True)
        st.pyplot(fig_pred)

with tab2:
    # Pakistan Stock Exchange Content
    st.header("📈 Pakistan Stock Exchange (PSX) Dashboard")
    
    # Load PSX data
    @st.cache_data
    def load_psx_data():
        # Using yfinance to get stock data
        psx_tickers = {
            "KSE100": "^KSE",  # KSE100 index
            "OGDC": "OGDC.KA",  # Oil & Gas Development Company
            "PPL": "PPL.KA",    # Pakistan Petroleum Limited
            "LUCK": "LUCK.KA",  # Lucky Cement
            "ENGRO": "ENGRO.KA", # Engro Corporation
            "HBL": "HBL.KA",    # Habib Bank Limited
            "UBL": "UBL.KA",    # United Bank Limited
            "MCB": "MCB.KA",    # MCB Bank Limited
            "EFERT": "EFERT.KA", # Engro Fertilizers
            "PSO": "PSO.KA"     # Pakistan State Oil
        }
        
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=365*3)  # 3 years data
        
        dfs = {}
        for name, ticker in psx_tickers.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                data["Ticker"] = name
                dfs[name] = data
            except:
                st.warning(f"Could not download data for {name} ({ticker})")
        
        if dfs:
            psx_df = pd.concat(dfs.values())
            psx_df = psx_df.reset_index()
            psx_df = psx_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            return psx_df
        else:
            return pd.DataFrame()

    psx_df = load_psx_data()

    if psx_df.empty:
        st.warning("Could not load PSX data. Please check your internet connection.")
    else:
        # Sidebar Filters for PSX
        with st.sidebar:
            st.title("📊 PSX Filter Options")
            tickers = sorted(psx_df["Ticker"].unique())
            selected_tickers = st.multiselect("Select Tickers", tickers, default=["KSE100", "OGDC", "PPL", "LUCK"])
            psx_min_date = psx_df["Date"].min().date()
            psx_max_date = psx_df["Date"].max().date()
            psx_date_range = st.slider("Select Date Range for PSX", 
                                      min_value=psx_min_date, 
                                      max_value=psx_max_date, 
                                      value=(psx_min_date, psx_max_date))

        # Filter PSX Data
        filtered_psx_df = psx_df[
            (psx_df["Ticker"].isin(selected_tickers)) &
            (psx_df["Date"] >= pd.to_datetime(psx_date_range[0])) &
            (psx_df["Date"] <= pd.to_datetime(psx_date_range[1]))
        ]
        pivot_psx_df = filtered_psx_df.pivot(index="Date", columns="Ticker", values="Close")

        # PSX Key Metrics
        st.header("📌 PSX Key Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Date Range", f"{(psx_date_range[1] - psx_date_range[0]).days} Days")
        with col2:
            st.metric("Tickers Selected", len(selected_tickers))
        with col3:
            st.metric("Data Points", f"{filtered_psx_df.shape[0]:,}")

        # PSX Line Chart
        st.header("📈 PSX Trend Over Time")
        st.subheader("Chart 1: Closing Prices Over Time")
        fig_psx1, ax_psx1 = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=pivot_psx_df, ax=ax_psx1)
        ax_psx1.set_title("Closing Prices of Selected Stocks/Index", fontsize=16)
        ax_psx1.set_ylabel("Price (PKR)")
        ax_psx1.set_xlabel("Date")
        ax_psx1.grid(True)
        st.pyplot(fig_psx1)

        # PSX Candlestick Chart
        st.header("🕯️ Candlestick Charts")
        st.subheader("Chart 2: Candlestick Chart for Selected Stocks")
        
        selected_ticker = st.selectbox("Select Ticker for Candlestick Chart", selected_tickers)
        ticker_data = filtered_psx_df[filtered_psx_df["Ticker"] == selected_ticker]
        
        fig_candle = px.line(ticker_data, x='Date', y='Close', title=f'{selected_ticker} Closing Price')
        fig_candle.update_layout(
            xaxis_rangeslider_visible=True,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        # PSX Volume Chart
        st.header("📊 Trading Volume Analysis")
        st.subheader("Chart 3: Trading Volume Over Time")
        fig_vol, ax_vol = plt.subplots(figsize=(14, 6))
        sns.barplot(data=filtered_psx_df, x="Date", y="Volume", hue="Ticker", ax=ax_vol)
        ax_vol.set_title("Trading Volume of Selected Stocks", fontsize=16)
        ax_vol.set_ylabel("Volume")
        ax_vol.set_xlabel("Date")
        plt.xticks(rotation=45)
        st.pyplot(fig_vol)

        # PSX Returns Analysis
        st.header("📈 Returns Analysis")
        st.subheader("Chart 4: Daily Returns of Selected Stocks")
        
        returns_df = pivot_psx_df.pct_change().dropna()
        fig_returns, ax_returns = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=returns_df, ax=ax_returns)
        ax_returns.set_title("Daily Returns of Selected Stocks", fontsize=16)
        ax_returns.set_ylabel("Daily Return")
        ax_returns.set_xlabel("Date")
        ax_returns.grid(True)
        st.pyplot(fig_returns)

        # PSX Correlation Heatmap
        st.header("🔥 Correlation Analysis")
        st.subheader("Chart 5: Correlation Between Selected Stocks")
        corr_matrix = pivot_psx_df.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax_corr)
        ax_corr.set_title("Correlation Between Selected Stocks", fontsize=16)
        st.pyplot(fig_corr)

        # PSX Volatility Analysis
        st.header("📉 Volatility Analysis")
        st.subheader("Chart 6: Rolling 30-Day Volatility")
        rolling_volatility = returns_df.rolling(window=30).std() * np.sqrt(252)  # Annualized volatility
        fig_volatility, ax_volatility = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=rolling_volatility, ax=ax_volatility)
        ax_volatility.set_title("30-Day Rolling Volatility (Annualized)", fontsize=16)
        ax_volatility.set_ylabel("Volatility")
        ax_volatility.set_xlabel("Date")
        ax_volatility.grid(True)
        st.pyplot(fig_volatility)

        # PSX Raw Data Table
        st.header("📄 Raw Filtered PSX Data")
        st.subheader("Table 1: Filtered PSX Data")
        st.dataframe(filtered_psx_df.reset_index(drop=True), use_container_width=True)

        # PSX Download Button
        st.download_button("📅 Download Filtered PSX Data", filtered_psx_df.to_csv(index=False), "filtered_psx_data.csv", "text/csv")

        # PSX Prediction Section
        st.header("🔮 Predict Future Stock Prices")
        st.subheader("Chart 7: Stock Price Predictions")
        psx_prediction_days = st.slider("How many days would you like to predict for stocks?", min_value=1, max_value=30, value=7)
        
        def predict_future_psx(df, ticker, days):
            ticker_data = df[df["Ticker"] == ticker]
            ticker_data = ticker_data.set_index("Date")
            ticker_data = ticker_data.resample("D").mean().ffill()  # Fill missing dates
            ticker_data = ticker_data["Close"]

            model = ARIMA(ticker_data, order=(5, 1, 0))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=days)
            forecast_dates = pd.date_range(ticker_data.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")

            prediction_df = pd.DataFrame({
                "Date": forecast_dates,
                "PredictedPrice": forecast
            })

            return prediction_df
         
        for ticker in selected_tickers:
            st.subheader(f"🔮 {ticker} - Prediction for the Next {psx_prediction_days} Days")
            prediction_psx_df = predict_future_psx(filtered_psx_df, ticker, psx_prediction_days)
            
            fig_psx_pred, ax_psx_pred = plt.subplots(figsize=(14, 6))
            sns.lineplot(x=prediction_psx_df["Date"], y=prediction_psx_df["PredictedPrice"], ax=ax_psx_pred, label="Predicted", color="red")
            ax_psx_pred.set_title(f"Predicted Closing Price for {ticker}", fontsize=16)
            ax_psx_pred.set_xlabel("Date")
            ax_psx_pred.set_ylabel("Price (PKR)")
            ax_psx_pred.grid(True)
            st.pyplot(fig_psx_pred)

with tab3:
    # Combined Analysis Content
    st.header("🌍📈 COVID-19 & PSX Combined Analysis")
    
    if psx_df.empty:
        st.warning("PSX data not available for combined analysis.")
    else:
        # Date range selection for combined analysis
        st.subheader("Select Date Range for Combined Analysis")
        combined_min_date = max(covid_df["Date"].min(), psx_df["Date"].min()).date()
        combined_max_date = min(covid_df["Date"].max(), psx_df["Date"].max()).date()
        combined_date_range = st.slider("Select Combined Date Range", 
                                      min_value=combined_min_date, 
                                      max_value=combined_max_date, 
                                      value=(combined_min_date, combined_max_date))

        # Filter both datasets for the selected date range
        combined_covid_df = covid_df[
            (covid_df["Country"] == "Pakistan") &
            (covid_df["Date"] >= pd.to_datetime(combined_date_range[0])) &
            (covid_df["Date"] <= pd.to_datetime(combined_date_range[1]))
        ]
        
        combined_psx_df = psx_df[
            (psx_df["Ticker"] == "KSE100") &
            (psx_df["Date"] >= pd.to_datetime(combined_date_range[0])) &
            (psx_df["Date"] <= pd.to_datetime(combined_date_range[1]))
        ]

        # Merge the datasets
        merged_df = pd.merge(
            combined_covid_df[["Date", "DailyCasesPerMillion"]],
            combined_psx_df[["Date", "Close"]],
            on="Date",
            how="inner"
        )
        merged_df.columns = ["Date", "DailyCasesPerMillion", "KSE100_Close"]

        if not merged_df.empty:
            # Correlation between COVID cases and stock market
            st.header("📊 Correlation Analysis")
            st.subheader("Chart 1: Correlation Between COVID-19 Cases and KSE100")
            
            correlation = merged_df["DailyCasesPerMillion"].corr(merged_df["KSE100_Close"])
            st.metric("Correlation Coefficient", f"{correlation:.2f}")
            
            fig_corr_combined, ax_corr_combined = plt.subplots(figsize=(10, 6))
            sns.regplot(x="DailyCasesPerMillion", y="KSE100_Close", data=merged_df, ax=ax_corr_combined)
            ax_corr_combined.set_title("COVID-19 Cases vs KSE100 Closing Price", fontsize=16)
            ax_corr_combined.set_xlabel("Daily COVID-19 Cases per Million")
            ax_corr_combined.set_ylabel("KSE100 Closing Price")
            st.pyplot(fig_corr_combined)

            # Dual-axis chart
            st.header("📈 Dual-Axis Time Series")
            st.subheader("Chart 2: COVID-19 Cases and KSE100 Over Time")
            
            fig, ax1 = plt.subplots(figsize=(14, 6))
            
            color = 'tab:red'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('COVID-19 Cases per Million', color=color)
            ax1.plot(merged_df['Date'], merged_df['DailyCasesPerMillion'], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('KSE100 Closing Price', color=color)
            ax2.plot(merged_df['Date'], merged_df['KSE100_Close'], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('COVID-19 Cases and KSE100 Performance Over Time', fontsize=16)
            st.pyplot(fig)

            # Lag analysis
            st.header("⏱️ Lag Analysis")
            st.subheader("Chart 3: Impact of COVID-19 Cases on KSE100 with Time Lags")
            
            max_lag = st.slider("Select Maximum Lag Days", 1, 30, 7)
            
            correlations = []
            lags = range(0, max_lag + 1)
            
            for lag in lags:
                if lag == 0:
                    corr = merged_df["DailyCasesPerMillion"].corr(merged_df["KSE100_Close"])
                else:
                    corr = merged_df["DailyCasesPerMillion"].iloc[:-lag].corr(merged_df["KSE100_Close"].iloc[lag:])
                correlations.append(corr)
            
            fig_lag, ax_lag = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=lags, y=correlations, marker='o', ax=ax_lag)
            ax_lag.set_title("Correlation Between COVID-19 Cases and KSE100 at Different Lags", fontsize=16)
            ax_lag.set_xlabel("Lag (Days)")
            ax_lag.set_ylabel("Correlation Coefficient")
            ax_lag.axhline(0, color='gray', linestyle='--')
            st.pyplot(fig_lag)

            # Rolling correlation
            st.header("🔄 Rolling Correlation")
            st.subheader("Chart 4: 30-Day Rolling Correlation Between COVID-19 and KSE100")
            
            window_size = st.slider("Select Rolling Window Size (Days)", 7, 90, 30)
            
            merged_df['RollingCorrelation'] = merged_df['DailyCasesPerMillion'].rolling(window=window_size).corr(merged_df['KSE100_Close'])
            
            fig_roll, ax_roll = plt.subplots(figsize=(14, 6))
            sns.lineplot(x='Date', y='RollingCorrelation', data=merged_df, ax=ax_roll)
            ax_roll.set_title(f"{window_size}-Day Rolling Correlation Between COVID-19 and KSE100", fontsize=16)
            ax_roll.set_xlabel("Date")
            ax_roll.set_ylabel("Correlation Coefficient")
            ax_roll.axhline(0, color='gray', linestyle='--')
            st.pyplot(fig_roll)

            # Event study - major COVID waves
            st.header("📅 Event Study: Major COVID Waves and Market Impact")
            
            # Define major COVID waves in Pakistan (example dates)
            covid_waves = {
                "First Wave (Jun 2020)": "2020-06-01",
                "Second Wave (Nov 2020)": "2020-11-01",
                "Third Wave (Mar 2021)": "2021-03-01",
                "Omicron Wave (Dec 2021)": "2021-12-01"
            }
            
            selected_wave = st.selectbox("Select COVID Wave to Analyze", list(covid_waves.keys()))
            wave_date = pd.to_datetime(covid_waves[selected_wave])
            
            # Create event window (30 days before and after)
            event_window = pd.date_range(wave_date - pd.Timedelta(days=30), wave_date + pd.Timedelta(days=30))
            
            event_df = merged_df[merged_df["Date"].isin(event_window)].copy()
            event_df["DaysFromEvent"] = (event_df["Date"] - wave_date).dt.days
            
            if not event_df.empty:
                # Normalize prices to 100 at event day
                event_day_price = event_df[event_df["DaysFromEvent"] == 0]["KSE100_Close"].values
                if len(event_day_price) > 0:
                    event_day_price = event_day_price[0]
                    event_df["NormalizedPrice"] = (event_df["KSE100_Close"] / event_day_price) * 100
                    
                    # Plot event study
                    fig_event, ax_event = plt.subplots(figsize=(14, 6))
                    
                    # Price line
                    sns.lineplot(x="DaysFromEvent", y="NormalizedPrice", data=event_df, ax=ax_event, label="KSE100 Index")
                    
                    # Cases bar
                    ax2 = ax_event.twinx()
                    sns.barplot(x="DaysFromEvent", y="DailyCasesPerMillion", data=event_df, ax=ax2, color='red', alpha=0.3, label="COVID Cases")
                    
                    ax_event.set_title(f"Market Impact of {selected_wave}", fontsize=16)
                    ax_event.set_xlabel("Days From Event")
                    ax_event.set_ylabel("Normalized Price (%)")
                    ax2.set_ylabel("Daily COVID Cases per Million")
                    
                    # Add vertical line at event day
                    ax_event.axvline(x=0, color='black', linestyle='--')
                    
                    # Combine legends
                    lines, labels = ax_event.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
                    
                    st.pyplot(fig_event)
                else:
                    st.warning("No data available for the exact event day. Try a different wave.")
            else:
                st.warning("No data available for the selected event window.")

        else:
            st.warning("No overlapping data available for the selected date range.")

st.markdown("---")
st.markdown("""
🎓 Developed by **Anas Hussain** anashussain0311@gmail.com | Department of CS | Data Science Institute  
📍 NED University of Engineering and Technology  
🧪 COVID-19 Dataset: [Our World in Data](https://ourworldindata.org/covid-cases)  
📈 PSX Data: Yahoo Finance via yfinance  
📱 Tools Used: Python, Pandas, Seaborn, Matplotlib, Streamlit, Plotly, yfinance  
""")