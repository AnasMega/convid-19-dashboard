import pandas as pd

# Load the uploaded dataset
file_path = "daily-new-confirmed-covid-19-cases-per-million-people.csv"
df = pd.read_csv(file_path)

# Show the first few rows to understand the structure
df.head()
# print(df)
# Step 1: Clean and rename columns
df.columns = ["Country", "Date", "CasesPerMillion"]
df["Date"] = pd.to_datetime(df["Date"])

# Step 2: Filter selected countries
selected_countries = ["United States", "India", "Brazil", "Pakistan"]
filtered_df = df[df["Country"].isin(selected_countries)]

# Step 3: Pivot data for plotting
pivot_df = filtered_df.pivot(index="Date", columns="Country", values="CasesPerMillion")

# Display the first few rows of the pivoted data
pivot_df.head()


import matplotlib.pyplot as plt
import seaborn as sns

 
 

# Plotting the data
plt.figure(figsize=(14, 7))
sns.lineplot(data=pivot_df)
plt.title("Daily New COVID-19 Cases per Million (7-day Avg)", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Cases per Million")
plt.grid(True)
plt.legend(title="Country", loc='upper left')
plt.tight_layout()
plt.show()