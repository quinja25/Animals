import pandas as pd
import numpy as np
import gc
import time
import warnings
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from data_processor import HanwooDataProcessor

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Configuration ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VERSION  = "v6"
OUT_DIR  = os.path.join(BASE_DIR, "submissions", VERSION)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def elapsed(start_time):
    return f"{(time.time() - start_time) / 60:.1f}min"

# ─── STEP 1: Data Preparation ─────────────────────
print(f"\n[{VERSION}] Starting Pipeline...")
t_start = time.time()

processor = HanwooDataProcessor(DATA_DIR)
processor.load_auxiliary_data()

print(f"\n[{VERSION}] Loading training data...")
train_raw = pd.read_csv(f"{DATA_DIR}/hanwoo_train.csv")
processor.fit_target_stats(train_raw)
train = processor.transform(train_raw, is_train=True)
del train_raw
gc.collect()

# ─── STEP 2: Feature Selection ────────────────────
# These columns are in train but NOT in test (internal grading metrics)
TARGET_RELATED = ['BACKFAT', 'REA', 'WINDEX', 'WGRADE', 'INSFAT', 'YUKSAK', 'FATSAK', 'TISSUE', 'GROWTH', 'COST_AMT']
NON_FEATURE = TARGET_RELATED + ['LAST_GRADE', 'target_q', 'target_y', 'FARM_UNIQUE_NO', 'KPN_NO', 'grade_score']
FEATURES = [c for c in train.columns if c not in NON_FEATURE and train[c].dtype in ['int32', 'int64', 'float32', 'float64']]

print(f"  Features ({len(FEATURES)}): {FEATURES}")

le_q = LabelEncoder().fit(processor.QUALITY_ORDER)
le_y = LabelEncoder().fit(processor.YIELD_ORDER)

X = train[FEATURES].fillna(-999).astype("float32")
y_q = le_q.transform(train["target_q"])
y_y = le_y.transform(train["target_y"])
groups = train["FARM_UNIQUE_NO"]

# ─── STEP 3: Hierarchical Training (LightGBM) ──────
print(f"\n[{VERSION}] Training Hierarchical Models (GroupKFold)...")
gkf = GroupKFold(n_splits=5)
LGB_PARAMS = {
    "objective": "multiclass",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1000,
}

models_q, models_y = [], []
scores_q, scores_y = [], []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_q, groups=groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_q_tr, y_q_val = y_q[tr_idx], y_q[val_idx]
    y_y_tr, y_y_val = y_y[tr_idx], y_y[val_idx]
    
    print(f"\n[Fold {fold+1}/5] Processing...")
    
    # Quality Model
    print(f"  Training Quality Model (6 classes)...")
    m_q = lgb.LGBMClassifier(**LGB_PARAMS, num_class=len(processor.QUALITY_ORDER))
    m_q.fit(X_tr, y_q_tr, eval_set=[(X_val, y_q_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=100)])
    
    val_pred_q = m_q.predict(X_val)
    f1_q = f1_score(y_q_val, val_pred_q, average='macro')
    scores_q.append(f1_q)
    models_q.append(m_q)
    
    # Yield Model
    print(f"  Training Yield Model (3 classes)...")
    m_y = lgb.LGBMClassifier(**LGB_PARAMS, num_class=len(processor.YIELD_ORDER))
    m_y.fit(X_tr, y_y_tr, eval_set=[(X_val, y_y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=100)])
    
    val_pred_y = m_y.predict(X_val)
    f1_y = f1_score(y_y_val, val_pred_y, average='macro')
    scores_y.append(f1_y)
    models_y.append(m_y)
    
    print(f"  --> Fold {fold+1} Results: Quality F1: {f1_q:.4f} | Yield F1: {f1_y:.4f}")
    
    # Free memory
    del X_tr, X_val, y_q_tr, y_q_val, y_y_tr, y_y_val
    gc.collect()

print(f"\n[{VERSION}] Training Complete!")
print(f"  Average Quality Macro-F1: {np.mean(scores_q):.4f} (+/- {np.std(scores_q):.4f})")
print(f"  Average Yield Macro-F1:   {np.mean(scores_y):.4f} (+/- {np.std(scores_y):.4f})")

# ─── STEP 4: Prediction & Submission ──────────────
print(f"\n[{VERSION}] Predicting on test data...")
test_raw = pd.read_csv(f"{DATA_DIR}/test_hanwoo.csv")
test = processor.transform(test_raw, is_train=False)

X_test = test[FEATURES].fillna(-999).astype("float32")

prob_q = np.zeros((len(X_test), len(processor.QUALITY_ORDER)))
prob_y = np.zeros((len(X_test), len(processor.YIELD_ORDER)))

for m_q, m_y in zip(models_q, models_y):
    prob_q += m_q.predict_proba(X_test) / 5
    prob_y += m_y.predict_proba(X_test) / 5

pred_q = le_q.inverse_transform(np.argmax(prob_q, axis=1))
pred_y = le_y.inverse_transform(np.argmax(prob_y, axis=1))

def combine_grade(q, y):
    if q == "등외": return "등외"
    return f"{q}{y}"

test_raw["LAST_GRADE"] = [combine_grade(q, y) for q, y in zip(pred_q, pred_y)]

out_path = f"{OUT_DIR}/260418.csv"
test_raw.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"  ✅ Results saved to: {out_path}")
print(f"  Total Time: {elapsed(t_start)}")

# Feature Importance (Quality model from fold 1 as reference)
fi = pd.DataFrame({'feature': FEATURES, 'importance': models_q[0].feature_importances_}).sort_values('importance', ascending=False)
fi.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
