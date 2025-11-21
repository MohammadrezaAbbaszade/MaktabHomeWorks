import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

########################################
# PART1
########################################

df = pd.read_csv("GlobalLandTemperaturesByCountry.csv")

df['dt'] = pd.to_datetime(df['dt'])
df['year'] = df['dt'].dt.year

print(df.isnull().sum())

df = df.fillna(df.mean(numeric_only=True))


########################################
# PART2
########################################

df["Temperature"] = df["AverageTemperature"].fillna(
    df["AverageTemperature"].rolling(window=5, min_periods=1, center=True).mean()
)


df['decade'] = (df['year'] // 10) * 10


########################################
# PART3
########################################


df = df.dropna(subset=['Country', 'year'])


years_per_country = df.groupby('Country')['year'].nunique()


valid_countries = years_per_country[years_per_country >= 100].index

df = df[df['Country'].isin(valid_countries)]

########################################
# PART4
########################################
global_mean = df["Temperature"].mean()
df["anomaly"] = df["Temperature"] - global_mean


pivot = df.pivot_table(
    index='decade',
    columns='Country',
    values='anomaly',
    aggfunc='mean'
)


matrix = pivot.to_numpy()

########################################
# PART5
########################################
slope, intercept = np.polyfit(decades, decade_means, 1)
print(f"\nLinear Trend: slope = {slope:.4f} °C per year, intercept = {intercept:.2f} °C")

heating_rate_per_decade = slope * 10
print(f"Linear Trend slope (°C/year) = {slope:.4f}")
print(f"Heating rate per decade (°C/decade) = {heating_rate_per_decade:.4f}")


plt.figure(figsize=(10,5))
plt.plot(decades, decade_means, 'o', label='Mean anomaly per decade')
plt.plot(decades, slope*decades + intercept, 'r-', label='Linear trend')
plt.xlabel('Decade')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('Global Temperature Trend by Decade')
plt.legend()
plt.show()