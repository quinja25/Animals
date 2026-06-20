import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
kpn_path = os.path.join(data_dir, "KPN 유전능력 자료.xlsx")

kpn_df = pd.read_excel(kpn_path)
kpn_cols = {
    'KPN명호': 'KPN_NO',
    '도체중 표준화육종가': 'kpn_weight_sbv',
    '등심단면적 표준화육종가': 'kpn_rea_sbv',
    '등지방두께 표준화육종가': 'kpn_backfat_sbv',
    '근내지방도 표준화육종가': 'kpn_insfat_sbv'
}

kpn_sub = kpn_df[list(kpn_cols.keys())].rename(columns=kpn_cols)
print("--- KPN Excel Sub Missingness ---")
print(kpn_sub.isnull().sum())
print(kpn_sub.isnull().mean())

print("\n--- Describing values in KPN Excel Sub ---")
print(kpn_sub.describe())
