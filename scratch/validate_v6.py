import pandas as pd
import numpy as np
import gc
import time
import os
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

sys.path.append("C:/Users/jaeyo/Projects/Animals/pipelines")
from data_processor import HanwooDataProcessor

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
processor = HanwooDataProcessor(data_dir)
processor.load_auxiliary_data()

print("Loading train raw...")
train_raw = pd.read_csv(f"{data_dir}/hanwoo_train.csv")

# Sample 20% of the unique farms for fast validation
np.random.seed(42)
unique_farms = train_raw['FARM_UNIQUE_NO'].unique()
sampled_farms = np.random.choice(unique_farms, size=int(len(unique_farms) * 0.2), replace=False)
train_raw_sampled = train_raw[train_raw['FARM_UNIQUE_NO'].isin(sampled_farms)].copy()

print(f"Sampled train shape: {train_raw_sampled.shape}")

# Fit stats on the sampled data (to avoid leakage from the rest of the train set)
processor.fit_target_stats(train_raw_sampled)
train = processor.transform(train_raw_sampled, is_train=True)

# Free memory
del train_raw, train_raw_sampled
gc.collect()

TARGET_RELATED = ['BACKFAT', 'REA', 'WINDEX', 'WGRADE', 'INSFAT', 'YUKSAK', 'FATSAK', 'TISSUE', 'GROWTH', 'COST_AMT']
NON_FEATURE = TARGET_RELATED + ['LAST_GRADE', 'target_q', 'target_y', 'FARM_UNIQUE_NO', 'KPN_NO', 'grade_score']
FEATURES = [c for c in train.columns if c not in NON_FEATURE and train[c].dtype in ['int32', 'int64', 'float32', 'float64']]

print(f"Features: {FEATURES}")

le_q = LabelEncoder().fit(processor.QUALITY_ORDER)
le_y = LabelEncoder().fit(processor.YIELD_ORDER)
le_final = LabelEncoder().fit(processor.FINAL_GRADES)

X = train[FEATURES].fillna(-999).astype("float32")
y_q = le_q.transform(train["target_q"])
y_y = le_y.transform(train["target_y"])
y_final = le_final.transform(train["LAST_GRADE"])
groups = train["FARM_UNIQUE_NO"]

gkf = GroupKFold(n_splits=5)
LGB_PARAMS = {
    "objective": "multiclass",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 500,
}

oof_preds_final = np.zeros(len(train), dtype=int)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_q, groups=groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_q_tr, y_q_val = y_q[tr_idx], y_q[val_idx]
    y_y_tr, y_y_val = y_y[tr_idx], y_y[val_idx]
    
    print(f"\n--- Fold {fold+1} ---")
    
    # Train Quality Model
    m_q = lgb.LGBMClassifier(**LGB_PARAMS, num_class=len(processor.QUALITY_ORDER))
    m_q.fit(X_tr, y_q_tr, eval_set=[(X_val, y_q_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    val_pred_q = m_q.predict(X_val)
    val_pred_q_str = le_q.inverse_transform(val_pred_q)
    
    # Train Yield Model
    m_y = lgb.LGBMClassifier(**LGB_PARAMS, num_class=len(processor.YIELD_ORDER))
    m_y.fit(X_tr, y_y_tr, eval_set=[(X_val, y_y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    val_pred_y = m_y.predict(X_val)
    val_pred_y_str = le_y.inverse_transform(val_pred_y)
    
    # Combine predictions
    def combine_grade(q, y):
        if q == "등외": return "등외"
        return f"{q}{y}"
    
    combined_preds = [combine_grade(q, y) for q, y in zip(val_pred_q_str, val_pred_y_str)]
    oof_preds_final[val_idx] = le_final.transform(combined_preds)
    
    # Calculate fold macro F1
    fold_f1 = f1_score(y_final[val_idx], oof_preds_final[val_idx], average='macro')
    print(f"Fold {fold+1} 16-class Combined Macro F1: {fold_f1:.4f}")

# Calculate overall OOF macro F1
overall_f1 = f1_score(y_final, oof_preds_final, average='macro')
print(f"\n======================================")
print(f"Overall OOF 16-class Combined Macro F1: {overall_f1:.4f}")
print(f"======================================")
