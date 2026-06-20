import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
train_path = os.path.join(data_dir, "hanwoo_train.csv")
lineage_path = os.path.join(data_dir, "hanwoo_lineage_0612.csv")

train = pd.read_csv(train_path, usecols=['CATTLE_NO', 'BIRTH_YMD'])
train['CATTLE_NO'] = train['CATTLE_NO'].astype(str).str.strip()
train['birth_year'] = pd.to_datetime(train['BIRTH_YMD'].astype(str), format="%Y%m%d", errors='coerce').dt.year

lineage = pd.read_csv(lineage_path, usecols=['CATTLE_NO'])
lineage['CATTLE_NO'] = lineage['CATTLE_NO'].astype(str).str.strip()
lineage_cattle = set(lineage['CATTLE_NO'])

train['in_lineage'] = train['CATTLE_NO'].isin(lineage_cattle)

print("--- Missingness by Birth Year ---")
print(train.groupby('birth_year')['in_lineage'].agg(['count', 'mean']))
