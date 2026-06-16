
import pandas as pd

def sample_kpn():
    print("Pedigree (Lineage) sample:")
    lineage = pd.read_csv('data/hanwoo_lineage.csv', usecols=['KPN_NO'], nrows=5)
    print(lineage['KPN_NO'].tolist())

    print("\nExcel KPN sample:")
    excel = pd.read_excel('data/KPN 유전능력 자료.xlsx', usecols=['KPN명호'], nrows=5)
    print(excel['KPN명호'].tolist())

if __name__ == "__main__":
    sample_kpn()
