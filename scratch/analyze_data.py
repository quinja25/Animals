import pandas as pd
import numpy as np
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
train_path = os.path.join(data_dir, "hanwoo_train.csv")
test_path = os.path.join(data_dir, "test_hanwoo.csv")

print("Loading train...")
train = pd.read_csv(train_path)
print("Loading test...")
test = pd.read_csv(test_path)

print("\n--- Train Shape & Columns ---")
print(train.shape)
print(train.columns)

print("\n--- Test Shape & Columns ---")
print(test.shape)
print(test.columns)

print("\n--- LAST_GRADE Distribution in Train ---")
print(train['LAST_GRADE'].value_counts(dropna=False))

print("\n--- Missing values in Train ---")
print(train.isnull().sum())

print("\n--- Missing values in Test ---")
print(test.isnull().sum())
