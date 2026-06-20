import hashlib
import base64
import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
kpn_path = os.path.join(data_dir, "KPN 유전능력 자료.xlsx")
lineage_path = os.path.join(data_dir, "hanwoo_lineage_0612.csv")

kpn_df = pd.read_excel(kpn_path)
kpn_df['KPN명호'] = kpn_df['KPN명호'].astype(str).str.strip()

print("Excel top KPN frequencies:")
print(kpn_df['KPN명호'].value_counts().head(10))

lineage_df = pd.read_csv(lineage_path)
lineage_df['KPN_NO'] = lineage_df['KPN_NO'].astype(str).str.strip()
print("\nLineage top KPN frequencies:")
print(lineage_df['KPN_NO'].value_counts().head(10))

# Check common strings hashes
common_strings = [
    "nan", "NaN", "None", "NULL", "", "0", "공란", "없음", "kpn", "KPN", "KPN_NO"
]
for s in common_strings:
    for encoding in ['utf-8', 'cp949', 'euc-kr']:
        try:
            b = s.encode(encoding)
            h = hashlib.md5(b).digest()
            b64 = base64.b64encode(h).decode('utf-8')
            if b64 in kpn_df['KPN명호'].values or b64 in lineage_df['KPN_NO'].values:
                print(f"Match found: '{s}' (encoding={encoding}) -> {b64}")
        except Exception:
            pass
