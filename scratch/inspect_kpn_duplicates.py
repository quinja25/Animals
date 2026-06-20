import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
kpn_path = os.path.join(data_dir, "KPN 유전능력 자료.xlsx")

kpn_df = pd.read_excel(kpn_path)
kpn_df['KPN명호'] = kpn_df['KPN명호'].astype(str).str.strip()

# Let's find a KPN that is duplicated
dup_kpns = kpn_df[kpn_df.duplicated('KPN명호')]['KPN명호'].unique()
print(f"Number of duplicated KPNs: {len(dup_kpns)}")

if len(dup_kpns) > 0:
    sample_dup = dup_kpns[0]
    print(f"\nShowing rows for duplicated KPN: {sample_dup}")
    print(kpn_df[kpn_df['KPN명호'] == sample_dup])
    
    # Are the values identical?
    all_dups = kpn_df[kpn_df['KPN명호'].isin(dup_kpns)]
    print("\nAre all duplicated rows identical?")
    # Check if duplicates are exact matches across all columns
    print(kpn_df.duplicated().sum(), "exact duplicate rows out of", len(kpn_df))
