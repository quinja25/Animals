import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
kpn_path = os.path.join(data_dir, "KPN 유전능력 자료.xlsx")

kpn_df = pd.read_excel(kpn_path)
kpn_df['KPN명호'] = kpn_df['KPN명호'].astype(str).str.strip()

counts = kpn_df['KPN명호'].value_counts()
print("Value count frequencies of KPN명호 in Excel:")
print(counts.value_counts())
