import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
train_path = os.path.join(data_dir, "hanwoo_train.csv")
lineage_path = os.path.join(data_dir, "hanwoo_lineage_0612.csv")

train = pd.read_csv(train_path, usecols=['CATTLE_NO'], nrows=10)
lineage = pd.read_csv(lineage_path, usecols=['CATTLE_NO'], nrows=10)

print("Train CATTLE_NO samples:")
for val in train['CATTLE_NO']:
    print(f"'{val}' (len={len(str(val))})")

print("\nLineage CATTLE_NO samples:")
for val in lineage['CATTLE_NO']:
    print(f"'{val}' (len={len(str(val))})")
