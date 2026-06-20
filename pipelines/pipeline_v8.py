import gc
import os
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from data_processor_v8 import HanwooDataProcessorV8

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VERSION = "v8"
OUT_DIR = os.path.join(BASE_DIR, "submissions", VERSION)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
SMOOTHING = 80
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 21)
CATBOOST_TASK_TYPE = os.getenv("CATBOOST_TASK_TYPE", "CPU")


def elapsed(start_time):
    return f"{(time.time() - start_time) / 60:.1f}min"


def add_grade_flags(frame):
    graded = frame["LAST_GRADE"].astype(str)
    frame["is_top"] = graded.isin(["1++A", "1++B", "1++C"]).astype("float32")
    frame["is_high_quality"] = graded.str.startswith(("1++", "1+")).astype("float32")
    frame["is_low"] = graded.isin(["3A", "3B", "3C", "등외"]).astype("float32")
    frame["is_a"] = graded.str.endswith("A").astype("float32")
    frame["is_b"] = graded.str.endswith("B").astype("float32")
    frame["is_c"] = graded.str.endswith("C").astype("float32")
    return frame


def build_group_summary(source_frame, group_col, prefix):
    summary = (
        source_frame.groupby(group_col, dropna=False)
        .agg(
            grade_sum=("grade_score", "sum"),
            grade_cnt=("grade_score", "count"),
            top_sum=("is_top", "sum"),
            highq_sum=("is_high_quality", "sum"),
            low_sum=("is_low", "sum"),
            a_sum=("is_a", "sum"),
            b_sum=("is_b", "sum"),
            c_sum=("is_c", "sum"),
        )
        .reset_index()
    )
    summary = summary.rename(
        columns={
            "grade_sum": f"{prefix}_grade_sum",
            "grade_cnt": f"{prefix}_grade_cnt",
            "top_sum": f"{prefix}_top_sum",
            "highq_sum": f"{prefix}_highq_sum",
            "low_sum": f"{prefix}_low_sum",
            "a_sum": f"{prefix}_a_sum",
            "b_sum": f"{prefix}_b_sum",
            "c_sum": f"{prefix}_c_sum",
        }
    )
    return summary


def attach_group_stats(train_fold, val_fold, test_fold=None):
    group_specs = [("KPN_NO", "kpn"), ("stn", "stn"), ("sido", "sido"), ("sigungu", "sigungu")]
    global_mean = train_fold["grade_score"].mean()
    global_rates = {
        "top_rate": train_fold["is_top"].mean(),
        "highq_rate": train_fold["is_high_quality"].mean(),
        "low_rate": train_fold["is_low"].mean(),
        "a_rate": train_fold["is_a"].mean(),
        "b_rate": train_fold["is_b"].mean(),
        "c_rate": train_fold["is_c"].mean(),
    }

    summaries = {}
    for group_col, prefix in group_specs:
        if group_col in train_fold.columns:
            summaries[group_col] = build_group_summary(train_fold, group_col, prefix)

    flag_specs = [
        ("top", "is_top"),
        ("highq", "is_high_quality"),
        ("low", "is_low"),
        ("a", "is_a"),
        ("b", "is_b"),
        ("c", "is_c"),
    ]

    def merge_stats(frame, leave_one_out=False):
        merged = frame.copy()
        for group_col, prefix in group_specs:
            if group_col in merged.columns and group_col in summaries:
                merged = merged.merge(summaries[group_col], on=group_col, how="left")
        for group_col, prefix in group_specs:
            sum_col = f"{prefix}_grade_sum"
            avg_col = f"{prefix}_grade_avg"
            cnt_col = f"{prefix}_grade_cnt"
            smooth_col = f"{prefix}_grade_smooth"
            if sum_col not in merged.columns:
                continue

            counts = merged[cnt_col].fillna(0)
            grade_sums = merged[sum_col].fillna(0)
            if leave_one_out:
                counts = (counts - 1).clip(lower=0)
                grade_sums = grade_sums - merged["grade_score"]

            merged[cnt_col] = counts.astype("float32")
            merged[avg_col] = np.where(counts > 0, grade_sums / counts, global_mean).astype("float32")
            merged[smooth_col] = (
                (grade_sums + SMOOTHING * global_mean) / (counts + SMOOTHING)
            ).astype("float32")

            for short_name, flag_col in flag_specs:
                flag_sum_col = f"{prefix}_{short_name}_sum"
                rate_col = f"{prefix}_{short_name}_rate"
                flag_sums = merged[flag_sum_col].fillna(0)
                if leave_one_out:
                    flag_sums = flag_sums - merged[flag_col]
                merged[rate_col] = (
                    (flag_sums + SMOOTHING * global_rates[f"{short_name}_rate"])
                    / (counts + SMOOTHING)
                ).astype("float32")

            internal_cols = [sum_col] + [f"{prefix}_{name}_sum" for name, _ in flag_specs]
            merged.drop(columns=internal_cols, inplace=True)
        return merged

    return (
        merge_stats(train_fold, leave_one_out=True),
        merge_stats(val_fold),
        merge_stats(test_fold) if test_fold is not None else None,
    )


def build_sample_weights(y_encoded, class_count, power=0.0, max_weight=1.0):
    counts = np.bincount(y_encoded, minlength=class_count)
    max_count = counts.max()
    weights = (max_count / np.maximum(counts[y_encoded], 1)) ** power
    weights = np.minimum(weights, max_weight)
    weights /= weights.mean()
    return weights.astype("float32")


def lgb_macro_f1(y_true, y_pred):
    return "macro_f1", f1_score(y_true, np.argmax(y_pred, axis=1), average="macro"), True


def prepare_features(frame, feature_cols, categorical_cols, category_levels):
    features = frame[feature_cols].copy()
    for column in categorical_cols:
        if column in features.columns:
            levels = category_levels[column]
            values = features[column].astype("string").fillna("__MISSING__")
            values = values.where(values.isin(levels), "__UNKNOWN__")
            features[column] = values.astype(pd.CategoricalDtype(categories=levels))
    numeric_cols = [column for column in feature_cols if column not in categorical_cols]
    for column in numeric_cols:
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(-999).astype("float32")
    return features


def prepare_catboost_features(features, categorical_cols):
    prepared = features.copy()
    for column in categorical_cols:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__").astype(str)
    return prepared


def select_blend_weight(y_true, lgb_oof, cat_oof):
    rows = []
    for lgb_weight in BLEND_WEIGHTS:
        blended = lgb_weight * lgb_oof + (1.0 - lgb_weight) * cat_oof
        score = f1_score(y_true, np.argmax(blended, axis=1), average="macro")
        rows.append({"lgb_weight": lgb_weight, "catboost_weight": 1.0 - lgb_weight, "macro_f1": score})
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    return float(results.iloc[0]["lgb_weight"]), results


print(f"\n[{VERSION}] Starting pipeline...")
t_start = time.time()

processor = HanwooDataProcessorV8(DATA_DIR)
processor.load_auxiliary_data()

print(f"\n[{VERSION}] Loading training data...")
train_raw = pd.read_csv(f"{DATA_DIR}/hanwoo_train.csv")
train = processor.transform(train_raw, is_train=True)
del train_raw
gc.collect()

print(f"\n[{VERSION}] Loading test data...")
test_raw = pd.read_csv(f"{DATA_DIR}/test_hanwoo.csv")
test_base = processor.transform(test_raw, is_train=False)

train["LAST_GRADE"] = train["LAST_GRADE"].astype(str)
train["grade_score"] = train["LAST_GRADE"].map(processor.GRADE_SCORE).astype("float32")
train = add_grade_flags(train)

TARGET_RELATED = {
    "BACKFAT",
    "REA",
    "WINDEX",
    "WGRADE",
    "INSFAT",
    "YUKSAK",
    "FATSAK",
    "TISSUE",
    "GROWTH",
    "COST_AMT",
}
NON_FEATURE = TARGET_RELATED | {
    "LAST_GRADE",
    "target_q",
    "target_y",
    "FARM_UNIQUE_NO",
    "grade_score",
    "is_top",
    "is_high_quality",
    "is_low",
    "is_a",
    "is_b",
    "is_c",
}

full_train_aug, _, _ = attach_group_stats(train, train.iloc[:0].copy())
FEATURES = [
    column
    for column in full_train_aug.columns
    if column not in NON_FEATURE
    and column in test_base.columns
    and (is_numeric_dtype(full_train_aug[column]) or is_categorical_dtype(full_train_aug[column]))
]

# Target-stat columns are generated after the fold split and therefore are not in test_base yet.
FEATURES.extend(
    column
    for column in full_train_aug.columns
    if column not in FEATURES
    and column not in NON_FEATURE
    and column.endswith("_grade_cnt")
    and is_numeric_dtype(full_train_aug[column])
)

CATEGORICAL_FEATURES = [
    column for column in processor.CATEGORICAL_COLUMNS if column in FEATURES
]
CATEGORY_LEVELS = {
    column: sorted(
        set(train[column].astype("string").fillna("__MISSING__").unique().tolist())
        | {"__MISSING__", "__UNKNOWN__"}
    )
    for column in CATEGORICAL_FEATURES
}

print(f"  Features ({len(FEATURES)}): {FEATURES}")
print(f"  Categorical ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")

le_final = LabelEncoder().fit(processor.FINAL_GRADES)
y_final = le_final.transform(train["LAST_GRADE"])
groups = train["FARM_UNIQUE_NO"].astype(str)

gkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
LGB_PARAMS = {
    "objective": "multiclass",
    "metric": "None",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 2.0,
    "max_depth": 12,
    "min_split_gain": 0.0,
    "max_bin": 255,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 3000,
}

CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1:average=Macro",
    "iterations": 2000,
    "learning_rate": 0.04,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.5,
    "random_seed": 42,
    "task_type": CATBOOST_TASK_TYPE,
    "allow_writing_files": False,
    "verbose": 100,
}

lgb_oof = np.zeros((len(train), len(processor.FINAL_GRADES)), dtype="float32")
cat_oof = np.zeros_like(lgb_oof)
lgb_test = np.zeros((len(test_base), len(processor.FINAL_GRADES)), dtype="float32")
cat_test = np.zeros_like(lgb_test)
lgb_scores = []
cat_scores = []
lgb_importance = np.zeros(len(FEATURES), dtype="float64")

print(f"\n[{VERSION}] Running LightGBM + CatBoost GroupKFold validation...")
for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, y_final, groups=groups), start=1):
    print(f"\n[Fold {fold}/{N_SPLITS}] Preparing data...")
    tr_fold = train.iloc[tr_idx].copy()
    val_fold = train.iloc[val_idx].copy()

    tr_aug, val_aug, test_aug = attach_group_stats(tr_fold, val_fold, test_base)
    X_tr = prepare_features(tr_aug, FEATURES, CATEGORICAL_FEATURES, CATEGORY_LEVELS)
    X_val = prepare_features(val_aug, FEATURES, CATEGORICAL_FEATURES, CATEGORY_LEVELS)
    X_test = prepare_features(test_aug, FEATURES, CATEGORICAL_FEATURES, CATEGORY_LEVELS)
    y_tr = y_final[tr_idx]
    y_val = y_final[val_idx]
    sample_weight = build_sample_weights(y_tr, len(processor.FINAL_GRADES))
    train_counts = np.bincount(y_tr, minlength=len(processor.FINAL_GRADES))
    val_counts = np.bincount(y_val, minlength=len(processor.FINAL_GRADES))
    print(f"  Train class range: {train_counts.min():,} - {train_counts.max():,}")
    print(f"  Validation class range: {val_counts.min():,} - {val_counts.max():,}")
    print(f"  Sample weight range: {sample_weight.min():.3f} - {sample_weight.max():.3f}")

    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS, num_class=len(processor.FINAL_GRADES))
    lgb_model.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric=lgb_macro_f1,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(period=100)],
    )

    lgb_val_proba = lgb_model.predict_proba(X_val)
    lgb_oof[val_idx] = lgb_val_proba
    lgb_test += lgb_model.predict_proba(X_test) / N_SPLITS
    lgb_fold_f1 = f1_score(y_val, np.argmax(lgb_val_proba, axis=1), average="macro")
    lgb_scores.append(lgb_fold_f1)
    lgb_importance += lgb_model.feature_importances_ / N_SPLITS
    print(f"  LightGBM fold Macro F1: {lgb_fold_f1:.4f}")
    del lgb_model, lgb_val_proba
    gc.collect()

    print("  Training CatBoost...")
    X_tr_cat = prepare_catboost_features(X_tr, CATEGORICAL_FEATURES)
    X_val_cat = prepare_catboost_features(X_val, CATEGORICAL_FEATURES)
    X_test_cat = prepare_catboost_features(X_test, CATEGORICAL_FEATURES)
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS)
    cat_model.fit(
        X_tr_cat,
        y_tr,
        cat_features=CATEGORICAL_FEATURES,
        sample_weight=sample_weight,
        eval_set=(X_val_cat, y_val),
        early_stopping_rounds=150,
        use_best_model=True,
    )
    cat_val_proba = cat_model.predict_proba(X_val_cat)
    cat_oof[val_idx] = cat_val_proba
    cat_test += cat_model.predict_proba(X_test_cat) / N_SPLITS
    cat_fold_f1 = f1_score(y_val, np.argmax(cat_val_proba, axis=1), average="macro")
    cat_scores.append(cat_fold_f1)
    print(f"  CatBoost fold Macro F1: {cat_fold_f1:.4f}")

    del (
        tr_fold,
        val_fold,
        tr_aug,
        val_aug,
        test_aug,
        X_tr,
        X_val,
        X_test,
        X_tr_cat,
        X_val_cat,
        X_test_cat,
        y_tr,
        y_val,
        sample_weight,
        cat_model,
        cat_val_proba,
    )
    gc.collect()

lgb_f1 = f1_score(y_final, np.argmax(lgb_oof, axis=1), average="macro")
cat_f1 = f1_score(y_final, np.argmax(cat_oof, axis=1), average="macro")
best_lgb_weight, blend_results = select_blend_weight(y_final, lgb_oof, cat_oof)
blend_results.to_csv(f"{OUT_DIR}/blend_search.csv", index=False)
best_blend_f1 = float(blend_results.iloc[0]["macro_f1"])

print(f"\n[{VERSION}] LightGBM OOF Macro F1: {lgb_f1:.4f}")
print(f"[{VERSION}] CatBoost OOF Macro F1: {cat_f1:.4f}")
print(f"[{VERSION}] Best blend OOF Macro F1: {best_blend_f1:.4f}")
print(f"[{VERSION}] Blend weights: LightGBM={best_lgb_weight:.2f}, CatBoost={1.0 - best_lgb_weight:.2f}")
print(f"[{VERSION}] LightGBM folds: {', '.join(f'{score:.4f}' for score in lgb_scores)}")
print(f"[{VERSION}] CatBoost folds: {', '.join(f'{score:.4f}' for score in cat_scores)}")

test_preds = best_lgb_weight * lgb_test + (1.0 - best_lgb_weight) * cat_test
test_pred_labels = np.argmax(test_preds, axis=1)

test_raw["LAST_GRADE"] = le_final.inverse_transform(test_pred_labels)
out_path = f"{OUT_DIR}/260418.csv"
test_raw.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"  Results saved to: {out_path}")
print(f"  Total Time: {elapsed(t_start)}")

fi = pd.DataFrame(
    {"feature": FEATURES, "importance": lgb_importance}
).sort_values("importance", ascending=False)
fi.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
