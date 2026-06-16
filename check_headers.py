
import pandas as pd
import os

def check_datasets():
    data_dir = 'data'
    files = [
        'hanwoo_train.csv',
        'test_hanwoo.csv',
        'hanwoo_weather.csv',
        'hanwoo_area.csv',
        'hanwoo_death.csv',
        'hanwoo_lineage.csv',
        'hanwoo_lineage_0612.csv',
        'KPN 유전능력 자료.xlsx'
    ]
    
    for f in files:
        path = os.path.join(data_dir, f)
        print(f"\n--- {f} ---")
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(path, nrows=0)
                print(f"Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading {f}: {e}")
        elif f.endswith('.xlsx'):
            try:
                df = pd.read_excel(path, nrows=0)
                print(f"Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading {f}: {e}")

if __name__ == "__main__":
    check_datasets()
