import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
train_path = os.path.join(data_dir, "hanwoo_train.csv")
test_path = os.path.join(data_dir, "test_hanwoo.csv")

print("Reading train...")
train = pd.read_csv(train_path, usecols=['CATTLE_NO'])
train['CATTLE_NO'] = train['CATTLE_NO'].astype(str).str.strip()
train_cattle = set(train['CATTLE_NO'])

print("Reading test...")
test = pd.read_csv(test_path, usecols=['CATTLE_NO'])
test['CATTLE_NO'] = test['CATTLE_NO'].astype(str).str.strip()
test_cattle = set(test['CATTLE_NO'])

for filename in ["hanwoo_lineage_0612.csv", "hanwoo_lineage.csv"]:
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        print(f"\n--- Checking {filename} ---")
        # Let's inspect columns first
        df_cols = pd.read_csv(path, nrows=5)
        print("Columns:", list(df_cols.columns))
        
        # Load and check overlap
        lineage = pd.read_csv(path, usecols=['CATTLE_NO', 'KPN_NO'])
        lineage['CATTLE_NO'] = lineage['CATTLE_NO'].astype(str).str.strip()
        lineage = lineage.drop_duplicates('CATTLE_NO')
        lineage_cattle = set(lineage['CATTLE_NO'])
        
        print(f"Total unique cattle: {len(lineage_cattle)}")
        train_in_lineage = train_cattle & lineage_cattle
        test_in_lineage = test_cattle & lineage_cattle
        print(f"Train in lineage: {len(train_in_lineage)} ({len(train_in_lineage)/len(train_cattle)*100:.2f}%)")
        print(f"Test in lineage: {len(test_in_lineage)} ({len(test_in_lineage)/len(test_cattle)*100:.2f}%)")
