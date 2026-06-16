
import pandas as pd
import numpy as np

def analyze_stn():
    train = pd.read_csv('data/hanwoo_train.csv', usecols=['stn', 'sido'])
    test = pd.read_csv('data/test_hanwoo.csv', usecols=['stn', 'sido'])

    train_stn = set(train['stn'].unique())
    test_stn = set(test['stn'].unique())

    print(f"Train unique STNs: {len(train_stn)}")
    print(f"Test unique STNs: {len(test_stn)}")
    print(f"STNs in both: {len(train_stn & test_stn)}")
    print(f"STNs only in Test: {len(test_stn - train_stn)}")
    
    # Check sido overlap
    train_sido = set(train['sido'].unique())
    test_sido = set(test['sido'].unique())
    print(f"\nTrain unique Sido: {train_sido}")
    print(f"Test unique Sido: {test_sido}")
    print(f"Sido only in Test: {test_sido - train_sido}")

if __name__ == "__main__":
    analyze_stn()
