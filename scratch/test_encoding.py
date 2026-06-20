import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
train_path = os.path.join(data_dir, "hanwoo_train.csv")

for enc in ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']:
    try:
        df = pd.read_csv(train_path, encoding=enc, nrows=100)
        print(f"Encoding {enc} read first 100 rows successfully.")
    except Exception as e:
        print(f"Encoding {enc} failed: {e}")

# Let's see what the unique values of LAST_GRADE are when read with different encodings (if successful)
# We can read a portion or the whole file, but let's read the whole file with cp949 or euc-kr if they worked, or read it by scanning
print("\nReading with cp949...")
try:
    df_cp = pd.read_csv(train_path, encoding='cp949')
    print("Unique values of LAST_GRADE (cp949):")
    print(df_cp['LAST_GRADE'].value_counts())
except Exception as e:
    print(f"cp949 full read failed: {e}")

print("\nReading with euc-kr...")
try:
    df_ek = pd.read_csv(train_path, encoding='euc-kr')
    print("Unique values of LAST_GRADE (euc-kr):")
    print(df_ek['LAST_GRADE'].value_counts())
except Exception as e:
    print(f"euc-kr full read failed: {e}")

print("\nReading with utf-8...")
try:
    df_u8 = pd.read_csv(train_path, encoding='utf-8')
    print("Unique values of LAST_GRADE (utf-8):")
    print(df_u8['LAST_GRADE'].value_counts())
except Exception as e:
    print(f"utf-8 full read failed: {e}")
