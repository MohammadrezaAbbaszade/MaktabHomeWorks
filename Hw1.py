import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########################################
# PART1
########################################
cities = ["Tehran", "Mashhad", "Isfahan", "Tabriz", "Shiraz", "Ahvaz"]
days = np.arange(1, 31)

temperature = np.random.randint(15, 41, size=(30 * len(cities)))
humidity = np.random.randint(20, 81, size=(30 * len(cities)))
rainfall = np.random.randint(0, 51, size=(30 * len(cities)))


########################################
# PART2
########################################

df = pd.DataFrame({
    "City": np.repeat(cities, 30),
    "Day": np.tile(days, len(cities)),
    "Temperature": temperature,
    "Humidity": humidity,
    "Rainfall": rainfall
})


print(df.head(10))


########################################
# PART3
########################################


avg_data = df.groupby("City")[["Temperature", "Humidity", "Rainfall"]].mean()
print("Average Temperature, Humidity, and Rainfall by City:")
print(avg_data)
print()


hottest_city = avg_data["Temperature"].idxmax()
coldest_city = avg_data["Temperature"].idxmin()
print(f"Hottest city: {hottest_city} ({avg_data.loc[hottest_city, 'Temperature']:.2f} °C)")
print(f"Coldest city: {coldest_city} ({avg_data.loc[coldest_city, 'Temperature']:.2f} °C)")
print()


rainy_days = df[df["Rainfall"] > 10].groupby("City")["Day"].count()
print("Number of days with rainfall > 10mm by city:")
print(rainy_days)

########################################
# PART4
########################################
isfahan_data = df[df["City"] == "Isfahan"]
plt.scatter(isfahan_data["Temperature"], isfahan_data["Humidity"], color='teal')

plt.title("Temperature vs Humidity - Isfahan (30 Days)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()


