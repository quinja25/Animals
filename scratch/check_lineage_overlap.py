import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
train_path = os.path.join(data_dir, "hanwoo_train.csv")
test_path = os.path.join(data_dir, "test_hanwoo.csv")
lineage_path = os.path.join(data_dir, "hanwoo_lineage_0612.csv")
if not os.path.exists(lineage_path):
    lineage_path = os.path.join(data_dir, "hanwoo_lineage.csv")

print(f"Reading lineage from: {lineage_path}")
lineage = pd.read_csv(lineage_path, usecols=['CATTLE_NO', 'KPN_NO'])
lineage['CATTLE_NO'] = lineage['CATTLE_NO'].astype(str).str.strip()
lineage = lineage.drop_duplicates('CATTLE_NO')

print("Reading train...")
train = pd.read_csv(train_path, usecols=['CATTLE_NO'])
train['CATTLE_NO'] = train['CATTLE_NO'].astype(str).str.strip()

print("Reading test...")
test = pd.read_csv(test_path, usecols=['CATTLE_NO'])
test['CATTLE_NO'] = test['CATTLE_NO'].astype(str).str.strip()

train_cattle = set(train['CATTLE_NO'])
test_cattle = set(test['CATTLE_NO'])
lineage_cattle = set(lineage['CATTLE_NO'])

print(f"Total train cattle: {len(train_cattle)}")
print(f"Total test cattle: {len(test_cattle)}")
print(f"Total lineage cattle: {len(lineage_cattle)}")

train_in_lineage = train_cattle & lineage_cattle
test_in_lineage = test_cattle & lineage_cattle

print(f"Train cattle in lineage: {len(train_in_lineage)} ({len(train_in_lineage)/len(train_cattle)*100:.2f}%)")
print(f"Test cattle in lineage: {len(test_in_lineage)} ({len(test_in_lineage)/len(test_cattle)*100:.2f}%)")
