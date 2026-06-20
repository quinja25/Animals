import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
lineage_path = os.path.join(data_dir, "hanwoo_lineage_0612.csv")

df = pd.read_csv(lineage_path)
print("--- lineage file shape ---")
print(df.shape)
print("\n--- lineage missing values ---")
print(df.isnull().sum())
print(df.isnull().mean())

print("\n--- lineage sample ---")
print(df.head())
