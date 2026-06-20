import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"

area = pd.read_csv(os.path.join(data_dir, "hanwoo_area.csv"))
print(f"Total rows in area.csv: {len(area)}")
print(f"Unique FARM_UNIQUE_NO: {area['FARM_UNIQUE_NO'].nunique()}")

# C2023, C2024, C2025 are the cattle count for each year
print("\n--- area.csv C2023/C2024/C2025 column stats ---")
print(area[['C2023', 'C2024', 'C2025', 'AREA']].describe())

# How many farms have multiple rows?
area_counts = area['FARM_UNIQUE_NO'].value_counts()
print(f"\nMax row count per farm: {area_counts.max()}")
print(f"Farms with 1 row: {(area_counts == 1).sum()}")
print(f"Farms with >1 row: {(area_counts > 1).sum()}")

weather = pd.read_csv(os.path.join(data_dir, "hanwoo_weather.csv"), parse_dates=['date'])
print(f"\n--- Weather date range ---")
print(f"Min date: {weather['date'].min()}")
print(f"Max date: {weather['date'].max()}")
print(f"Unique stations: {weather['stn'].nunique()}")
print(weather['stn'].unique())
