import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
kpn_path = os.path.join(data_dir, "KPN 유전능력 자료.xlsx")

kpn_df = pd.read_excel(kpn_path)
kpn_df['KPN명호'] = kpn_df['KPN명호'].astype(str).str.strip()

# Let's inspect the columns for a duplicated KPN
sample_kpn = kpn_df['KPN명호'].value_counts().index[0]
sub = kpn_df[kpn_df['KPN명호'] == sample_kpn]

print(f"Duplicated KPN: {sample_kpn}")
print(f"Number of rows: {len(sub)}")

# Find columns where values are NOT all the same
diff_cols = []
for col in sub.columns:
    if sub[col].nunique() > 1:
        diff_cols.append(col)

print("\nColumns that differ across duplicate rows:")
print(diff_cols)

print("\nShowing first few differing columns:")
print(sub[diff_cols[:10]].head(10))
