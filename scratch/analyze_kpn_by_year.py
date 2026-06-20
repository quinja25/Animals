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

# For each birth year, let's see how many have KPN, and of those, how many matched
train['has_kpn'] = train['KPN_NO'].notnull() & (train['KPN_NO'] != 'nan')
train['matched_kpn'] = train['has_kpn'] & train['kpn_weight_sbv'].notnull()

print("--- KPN Availability and Match Rate by Birth Year ---")
summary = train.groupby('birth_year').agg(
    total_cows=('has_kpn', 'count'),
    pct_has_kpn=('has_kpn', 'mean'),
    pct_matched_of_all=('matched_kpn', 'mean'),
)
# For those with KPN, what pct matched?
# We can compute it as pct_matched_of_all / pct_has_kpn where pct_has_kpn > 0
summary['pct_matched_of_has_kpn'] = summary['pct_matched_of_all'] / (summary['pct_has_kpn'] + 1e-9)

print(summary)
