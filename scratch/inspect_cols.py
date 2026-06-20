import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"

print("--- Checking area.csv columns ---")
area = pd.read_csv(os.path.join(data_dir, "hanwoo_area.csv"), nrows=5)
print("Columns:", list(area.columns))
print(area.head(2))

print("\n--- Checking death.csv columns ---")
death = pd.read_csv(os.path.join(data_dir, "hanwoo_death.csv"), nrows=5)
print("Columns:", list(death.columns))
print(death.head(2))

print("\n--- Checking weather.csv columns ---")
weather = pd.read_csv(os.path.join(data_dir, "hanwoo_weather.csv"), nrows=5)
print("Columns:", list(weather.columns))
print(weather.head(2))
