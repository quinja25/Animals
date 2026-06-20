import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
train_path = os.path.join(data_dir, "hanwoo_train.csv")

df = pd.read_csv(train_path)
bad_grade = df[~df['LAST_GRADE'].str.contains('[0-9]')]
print("Unique bad grades:")
for val in bad_grade['LAST_GRADE'].unique():
    print(val, repr(val), val.encode('utf-8'))
