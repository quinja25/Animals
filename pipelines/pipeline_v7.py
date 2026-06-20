import ast
import gc
import os
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from data_processor_v7 import HanwooDataProcessorV7

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VERSION = "v7"
OUT_DIR = os.path.join(BASE_DIR, "submissions", VERSION)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

SMOOTHING = 50
N_SPLITS = 5


def elapsed(start_time):
    return f"{(time.time() - start_time) / 60:.1f}min"


def normalize_grade(value):
    if pd.isna(value):
        return None

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="ignore")

    text = str(value).strip()
    if text.startswith("b'") or text.startswith('b"'):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, bytes):
                return parsed.decode("utf-8", errors="ignore")
        except Exception:
            pass
    return text


def attach_group_stats(train_fold, val_fold, test_fold=None):
    global_mean = train_fold["grade_score"].mean()

    def summarize(group_col, prefix, source_frame):
        summary = (
            source_frame.groupby(group_col, dropna=False)["grade_score"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": f"{prefix}_grade_avg", "count": f"{prefix}_grade_cnt"})
        )
        summary[f"{prefix}_grade_smooth"] = (
            summary[f"{prefix}_grade_cnt"] * summary[f"{prefix}_grade_avg"] + SMOOTHING * global_mean
        ) / (summary[f"{prefix}_grade_cnt"] + SMOOTHING)
        return summary

    kpn_stats = summarize("KPN_NO", "kpn", train_fold) if "KPN_NO" in train_fold.columns else None
    stn_stats = summarize("stn", "stn", train_fold) if "stn" in train_fold.columns else None

    def merge_stats(frame):
        merged = frame.copy()
        if kpn_stats is not None and "KPN_NO" in merged.columns:
            merged = merged.merge(kpn_stats, on="KPN_NO", how="left")
        if stn_stats is not None and "stn" in merged.columns:
            merged = merged.merge(stn_stats, on="stn", how="left")
        if "kpn_grade_cnt" in merged.columns:
            merged["kpn_grade_cnt"] = merged["kpn_grade_cnt"].fillna(0)
        if "stn_grade_cnt" in merged.columns:
            merged["stn_grade_cnt"] = merged["stn_grade_cnt"].fillna(0)
        for col in ["kpn_grade_avg", "kpn_grade_smooth", "stn_grade_avg", "stn_grade_smooth"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(global_mean)
        return merged

    train_aug = merge_stats(train_fold)
    val_aug = merge_stats(val_fold)
    test_aug = merge_stats(test_fold) if test_fold is not None else None
    return train_aug, val_aug, test_aug


def build_sample_weights(y_encoded, class_count):
    counts = np.bincount(y_encoded, minlength=class_count)
    max_count = counts.max()
    weights = max_count / np.maximum(counts[y_encoded], 1)
    return weights.astype("float32")


print(f"\n[{VERSION}] Starting pipeline...")
t_start = time.time()

processor = HanwooDataProcessorV7(DATA_DIR)
processor.load_auxiliary_data()

print(f"\n[{VERSION}] Loading training data...")
train_raw = pd.read_csv(f"{DATA_DIR}/hanwoo_train.csv")
train = processor.transform(train_raw, is_train=True)
del train_raw
gc.collect()

train["LAST_GRADE"] = train["LAST_GRADE"].map(normalize_grade).fillna(processor.FINAL_GRADES[-1])
train["grade_score"] = train["LAST_GRADE"].map(processor.GRADE_SCORE).astype("float32")

TARGET_RELATED = ["BACKFAT", "REA", "WINDEX", "WGRADE", "INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH", "COST_AMT"]
NON_FEATURE = TARGET_RELATED + ["LAST_GRADE", "target_q", "target_y", "FARM_UNIQUE_NO", "KPN_NO", "grade_score"]
BASE_FEATURES = [
    c for c in train.columns
    if c not in NON_FEATURE and train[c].dtype in ["int32", "int64", "float32", "float64"]
]

STAT_FEATURES = [
    "kpn_grade_avg",
    "kpn_grade_cnt",
    "kpn_grade_smooth",
    "stn_grade_avg",
    "stn_grade_cnt",
    "stn_grade_smooth",
]
FEATURES = BASE_FEATURES + STAT_FEATURES

print(f"  Base features ({len(BASE_FEATURES)}): {BASE_FEATURES}")
print(f"  Stat features ({len(STAT_FEATURES)}): {STAT_FEATURES}")

le_final = LabelEncoder().fit(processor.FINAL_GRADES)
y_final = le_final.transform(train["LAST_GRADE"])
groups = train["FARM_UNIQUE_NO"].astype(str)

gkf = GroupKFold(n_splits=N_SPLITS)
LGB_PARAMS = {
    "objective": "multiclass",
    "learning_rate": 0.03,
    "num_leaves": 255,
    "min_child_samples": 25,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 2000,
}

oof_preds = np.zeros((len(train), len(processor.FINAL_GRADES)), dtype="float32")
fold_models = []
fold_scores = []

print(f"\n[{VERSION}] Running GroupKFold validation with fold-safe stats...")
for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, y_final, groups=groups), start=1):
    print(f"\n[Fold {fold}/{N_SPLITS}] Preparing data...")
    tr_fold = train.iloc[tr_idx].copy()
    val_fold = train.iloc[val_idx].copy()

    tr_aug, val_aug, _ = attach_group_stats(tr_fold, val_fold)
    X_tr = tr_aug[FEATURES].fillna(-999).astype("float32")
    X_val = val_aug[FEATURES].fillna(-999).astype("float32")
    y_tr = y_final[tr_idx]
    y_val = y_final[val_idx]
    sample_weight = build_sample_weights(y_tr, len(processor.FINAL_GRADES))

    print(f"  Training direct 16-class model...")
    model = lgb.LGBMClassifier(**LGB_PARAMS, num_class=len(processor.FINAL_GRADES))
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=100)],
    )

    val_proba = model.predict_proba(X_val)
    oof_preds[val_idx] = val_proba
    val_pred = np.argmax(val_proba, axis=1)
    fold_f1 = f1_score(y_val, val_pred, average="macro")
    fold_scores.append(fold_f1)
    fold_models.append(model)
    print(f"  Fold {fold} Macro F1: {fold_f1:.4f}")

    del tr_fold, val_fold, tr_aug, val_aug, X_tr, X_val, y_tr, y_val, sample_weight
    gc.collect()

oof_pred_labels = np.argmax(oof_preds, axis=1)
overall_f1 = f1_score(y_final, oof_pred_labels, average="macro")
print(f"\n[{VERSION}] OOF Macro F1: {overall_f1:.4f}")
print(f"[{VERSION}] Fold scores: {', '.join(f'{s:.4f}' for s in fold_scores)}")

print(f"\n[{VERSION}] Training final model on full data...")
full_stats_train, _, _ = attach_group_stats(train, train.iloc[:0].copy())
X_full = full_stats_train[FEATURES].fillna(-999).astype("float32")
full_weights = build_sample_weights(y_final, len(processor.FINAL_GRADES))

final_model = lgb.LGBMClassifier(**LGB_PARAMS, num_class=len(processor.FINAL_GRADES))
final_model.fit(
    X_full,
    y_final,
    sample_weight=full_weights,
    callbacks=[lgb.log_evaluation(period=100)],
)

print(f"\n[{VERSION}] Predicting on test data...")
test_raw = pd.read_csv(f"{DATA_DIR}/test_hanwoo.csv")
test = processor.transform(test_raw, is_train=False)
test = test.copy()
test["LAST_GRADE"] = test.get("LAST_GRADE", pd.Series([None] * len(test))).map(normalize_grade)

full_stats_train, _, test_aug = attach_group_stats(train, train.iloc[:0].copy(), test)
X_test = test_aug[FEATURES].fillna(-999).astype("float32")
test_pred = np.argmax(final_model.predict_proba(X_test), axis=1)
test_raw["LAST_GRADE"] = le_final.inverse_transform(test_pred)

out_path = f"{OUT_DIR}/260418.csv"
test_raw.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"  Results saved to: {out_path}")
print(f"  Total Time: {elapsed(t_start)}")

fi = pd.DataFrame(
    {"feature": FEATURES, "importance": final_model.feature_importances_}
).sort_values("importance", ascending=False)
fi.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
