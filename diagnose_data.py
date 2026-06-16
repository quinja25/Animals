
import pandas as pd
import numpy as np
import os

def analyze():
    print("Loading data...")
    train = pd.read_csv('data/hanwoo_train.csv', usecols=['FARM_UNIQUE_NO', 'LAST_GRADE', 'ABATT_DATE'])
    test = pd.read_csv('data/test_hanwoo.csv', usecols=['FARM_UNIQUE_NO', 'ABATT_DATE'])

    train_farms = set(train['FARM_UNIQUE_NO'].unique())
    test_farms = set(test['FARM_UNIQUE_NO'].unique())

    print(f"Train unique farms: {len(train_farms):,}")
    print(f"Test unique farms: {len(test_farms):,}")
    print(f"Farms in both: {len(train_farms & test_farms):,}")
    print(f"Farms only in Test: {len(test_farms - train_farms):,}")
    print(f"Test coverage (farms also in train): {len(train_farms & test_farms) / len(test_farms) * 100:.2f}%")

    train['ABATT_DATE'] = pd.to_datetime(train['ABATT_DATE'])
    test['ABATT_DATE'] = pd.to_datetime(test['ABATT_DATE'])

    print(f"\nTrain date range: {train['ABATT_DATE'].min()} ~ {train['ABATT_DATE'].max()}")
    print(f"Test date range: {test['ABATT_DATE'].min()} ~ {test['ABATT_DATE'].max()}")

    print(f"\nRows in Train: {len(train):,}")
    print(f"Rows in Test: {len(test):,}")

    # Check for duplicate farm entries in area.csv
    area = pd.read_csv('data/hanwoo_area.csv')
    print(f"\nArea.csv unique farms: {area['FARM_UNIQUE_NO'].nunique():,}")
    print(f"Area.csv total rows: {len(area):,}")
    if area['FARM_UNIQUE_NO'].nunique() != len(area):
        print("!!! Warning: area.csv contains duplicate FARM_UNIQUE_NO entries.")

    # Distribution of targets in train
    print("\nTrain Target Distribution:")
    print(train['LAST_GRADE'].value_counts(normalize=True).sort_index())

if __name__ == "__main__":
    analyze()
