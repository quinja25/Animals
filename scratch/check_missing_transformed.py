import pandas as pd
import numpy as np
import os
import sys
sys.path.append("C:/Users/jaeyo/Projects/Animals/pipelines")
from data_processor import HanwooDataProcessor

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
processor = HanwooDataProcessor(data_dir)
processor.load_auxiliary_data()

print("Loading train...")
train_raw = pd.read_csv(f"{data_dir}/hanwoo_train.csv")
processor.fit_target_stats(train_raw)

print("Transforming train...")
train = processor.transform(train_raw, is_train=True)
print("Train missing percentages:")
print(train.isnull().mean() * 100)

print("\nLoading test...")
test_raw = pd.read_csv(f"{data_dir}/test_hanwoo.csv")
print("Transforming test...")
test = processor.transform(test_raw, is_train=False)
print("Test missing percentages:")
print(test.isnull().mean() * 100)
