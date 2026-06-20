import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
kpn_path = os.path.join(data_dir, "KPN 유전능력 자료.xlsx")
lineage_path = os.path.join(data_dir, "hanwoo_lineage_0612.csv")

print("Reading Excel...")
kpn_df = pd.read_excel(kpn_path)
print("Excel columns:", kpn_df.columns)
print("Excel samples of KPN명호:")
print(kpn_df['KPN명호'].head(10).tolist())

print("\nReading Lineage CSV...")
lineage_df = pd.read_csv(lineage_path, usecols=['KPN_NO'], nrows=1000)
print("Lineage samples of KPN_NO:")
print(lineage_df['KPN_NO'].dropna().unique()[:10].tolist())

# Let's count how many KPNs in lineage match Excel
excel_kpn = set(kpn_df['KPN명호'].astype(str).str.strip())
lineage_kpn_all = pd.read_csv(lineage_path, usecols=['KPN_NO'])
lineage_kpn = set(lineage_kpn_all['KPN_NO'].dropna().astype(str).str.strip())

print(f"\nUnique KPNs in Excel: {len(excel_kpn)}")
print(f"Unique KPNs in Lineage: {len(lineage_kpn)}")
overlap = excel_kpn & lineage_kpn
print(f"Overlap: {len(overlap)}")
print(f"Percentage of lineage KPNs in Excel: {len(overlap)/len(lineage_kpn)*100:.2f}%")
