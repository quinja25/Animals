import pandas as pd
import os
import sys
sys.path.append("C:/Users/jaeyo/Projects/Animals/pipelines")
from data_processor import HanwooDataProcessor

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
processor = HanwooDataProcessor(data_dir)
processor.load_auxiliary_data()

train_raw = pd.read_csv(f"{data_dir}/hanwoo_train.csv")
train = processor.transform(train_raw, is_train=True)

# Let's see: how many train rows have a non-null KPN_NO?
has_kpn = train[train['KPN_NO'].notnull() & (train['KPN_NO'] != 'nan')]
print(f"Train rows with non-null KPN_NO: {len(has_kpn)} ({len(has_kpn)/len(train)*100:.2f}%)")

# How many of these matched with kpn_weight_sbv?
matched_kpn = has_kpn[has_kpn['kpn_weight_sbv'].notnull()]
print(f"Of those, matched with KPN Excel: {len(matched_kpn)} ({len(matched_kpn)/len(has_kpn)*100:.2f}%)")

# Let's inspect unique KPN_NOs in train that matched vs failed
all_train_kpns = set(has_kpn['KPN_NO'].unique())
excel_kpns = set(processor.kpn_bv['KPN_NO'].unique())

print(f"\nUnique KPNs in Train: {len(all_train_kpns)}")
print(f"Unique KPNs in Excel: {len(excel_kpns)}")

matched_kpns = all_train_kpns & excel_kpns
print(f"Overlap: {len(matched_kpns)}")

print("\nSample of matched KPNs in Train:")
print(list(matched_kpns)[:5])

failed_kpns = all_train_kpns - excel_kpns
print("\nSample of failed KPNs in Train:")
print(list(failed_kpns)[:5])

print("\nSample of KPNs in Excel:")
print(list(excel_kpns)[:5])
