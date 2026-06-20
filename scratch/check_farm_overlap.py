import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
train_path = os.path.join(data_dir, "hanwoo_train.csv")
test_path = os.path.join(data_dir, "test_hanwoo.csv")
area_path = os.path.join(data_dir, "hanwoo_area.csv")
death_path = os.path.join(data_dir, "hanwoo_death.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
area = pd.read_csv(area_path)
death = pd.read_csv(death_path)

train['FARM_UNIQUE_NO'] = train['FARM_UNIQUE_NO'].astype(str).str.strip()
test['FARM_UNIQUE_NO'] = test['FARM_UNIQUE_NO'].astype(str).str.strip()
area['FARM_UNIQUE_NO'] = area['FARM_UNIQUE_NO'].astype(str).str.strip()
death['FARM_UNIQUE_NO'] = death['FARM_UNIQUE_NO'].astype(str).str.strip()

print("--- Overlap of FARM_UNIQUE_NO ---")
train_farms = set(train['FARM_UNIQUE_NO'])
test_farms = set(test['FARM_UNIQUE_NO'])
area_farms = set(area['FARM_UNIQUE_NO'])
death_farms = set(death['FARM_UNIQUE_NO'])

print(f"Number of unique farms in Train: {len(train_farms)}")
print(f"Number of unique farms in Test: {len(test_farms)}")
print(f"Number of unique farms in Area: {len(area_farms)}")
print(f"Number of unique farms in Death: {len(death_farms)}")

print(f"Train and Test overlap: {len(train_farms & test_farms)}")
print(f"Test and Area overlap: {len(test_farms & area_farms)} ({(len(test_farms & area_farms)/len(test_farms))*100:.2f}%)")
print(f"Test and Death overlap: {len(test_farms & death_farms)} ({(len(test_farms & death_farms)/len(test_farms))*100:.2f}%)")

print(f"Train and Area overlap: {len(train_farms & area_farms)} ({(len(train_farms & area_farms)/len(train_farms))*100:.2f}%)")
print(f"Train and Death overlap: {len(train_farms & death_farms)} ({(len(train_farms & death_farms)/len(train_farms))*100:.2f}%)")
