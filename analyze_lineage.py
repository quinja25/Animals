
import pandas as pd
import numpy as np

def analyze_lineage():
    train = pd.read_csv('data/hanwoo_train.csv', usecols=['CATTLE_NO'])
    test = pd.read_csv('data/test_hanwoo.csv', usecols=['CATTLE_NO'])
    lineage = pd.read_csv('data/hanwoo_lineage.csv')

    train_cattle = set(train['CATTLE_NO'].unique())
    test_cattle = set(test['CATTLE_NO'].unique())
    
    # Get KPN for train and test
    train_lineage = lineage[lineage['CATTLE_NO'].isin(train_cattle)]
    test_lineage = lineage[lineage['CATTLE_NO'].isin(test_cattle)]
    
    train_kpn = set(train_lineage['KPN_NO'].dropna().unique())
    test_kpn = set(test_lineage['KPN_NO'].dropna().unique())
    
    print(f"Train unique KPN: {len(train_kpn)}")
    print(f"Test unique KPN: {len(test_kpn)}")
    print(f"KPN in both: {len(train_kpn & test_kpn)}")
    print(f"KPN only in Test: {len(test_kpn - train_kpn)}")
    print(f"Test KPN coverage: {len(train_kpn & test_kpn) / len(test_kpn) * 100:.2f}%")

    train_father = set(train_lineage['FATHER_CATTLE_NO'].dropna().unique())
    test_father = set(test_lineage['FATHER_CATTLE_NO'].dropna().unique())
    print(f"\nTrain unique Father: {len(train_father)}")
    print(f"Test unique Father: {len(test_father)}")
    print(f"Father in both: {len(train_father & test_father)}")
    print(f"Father coverage: {len(train_father & test_father) / len(test_father) * 100:.2f}%")

if __name__ == "__main__":
    analyze_lineage()
