import gc
import os
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from data_processor_v8 import HanwooDataProcessorV8

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VERSION = "v9"
OUT_DIR = os.path.join(BASE_DIR, "submissions", VERSION)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 21)
CATBOOST_TASK_TYPE = os.getenv("CATBOOST_TASK_TYPE", "CPU")
SAVE_OOF = os.getenv("SAVE_OOF", "1") == "1"
GROUP_SPECS = [("KPN_NO", "kpn"), ("stn", "stn"), ("sido", "sido"), ("sigungu", "sigungu")]


def elapsed(start_time):
    return f"{(time.time() - start_time) / 60:.1f}min"


def attach_group_counts(train_fold, val_fold, test_fold=None):
    train_aug = train_fold.copy()
    val_aug = val_fold.copy()
    test_aug = test_fold.copy() if test_fold is not None else None
    for group_col, prefix in GROUP_SPECS:
        if group_col not in train_fold.columns:
            continue
        counts = train_fold.groupby(group_col, observed=True, dropna=False).size()
        count_col = f"{prefix}_group_count"
        train_aug[count_col] = train_fold[group_col].map(counts).astype("float32") - 1
        val_aug[count_col] = val_fold[group_col].map(counts).fillna(0).astype("float32")
        if test_aug is not None:
            test_aug[count_col] = test_fold[group_col].map(counts).fillna(0).astype("float32")
    return train_aug, val_aug, test_aug


def prepare_features(frame, feature_cols, categorical_cols, category_levels):
    features = frame[feature_cols].copy()
    for column in categorical_cols:
        values = features[column].astype("string").fillna("__MISSING__")
        values = values.where(values.isin(category_levels[column]), "__UNKNOWN__")
        features[column] = values.astype(pd.CategoricalDtype(categories=category_levels[column]))
    for column in feature_cols:
        if column not in categorical_cols:
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(-999).astype("float32")
    return features


def prepare_catboost_features(features, categorical_cols):
    prepared = features.copy()
    for column in categorical_cols:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__").astype(str)
    return prepared


def lgb_macro_f1(y_true, y_pred):
    return "macro_f1", f1_score(y_true, np.argmax(y_pred, axis=1), average="macro"), True


def combine_hierarchical_probabilities(q_proba, y_proba, q_encoder, y_encoder, final_encoder):
    combined = np.zeros((len(q_proba), len(final_encoder.classes_)), dtype="float32")
    q_index = {label: index for index, label in enumerate(q_encoder.classes_)}
    y_index = {label: index for index, label in enumerate(y_encoder.classes_)}
    final_index = {label: index for index, label in enumerate(final_encoder.classes_)}
    outlier = "등외"

    for grade in final_encoder.classes_:
        if grade == outlier:
            combined[:, final_index[grade]] = q_proba[:, q_index[outlier]]
            continue
        quality = grade[:-1]
        yield_grade = grade[-1]
        combined[:, final_index[grade]] = (
            q_proba[:, q_index[quality]] * y_proba[:, y_index[yield_grade]]
        )
    combined /= np.maximum(combined.sum(axis=1, keepdims=True), 1e-12)
    return combined


def apply_class_scales(probabilities, scales):
    calibrated = probabilities * scales.reshape(1, -1)
    calibrated /= np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-12)
    return calibrated


def optimize_class_scales(y_true, probabilities, max_rows=300_000):
    rng = np.random.default_rng(42)
    if len(y_true) > max_rows:
        indices = rng.choice(len(y_true), size=max_rows, replace=False)
        y_search = y_true[indices]
        p_search = probabilities[indices]
    else:
        y_search = y_true
        p_search = probabilities

    scales = np.ones(probabilities.shape[1], dtype="float32")
    factors = [0.80, 0.90, 1.0, 1.10, 1.25]
    best_score = f1_score(y_search, np.argmax(p_search, axis=1), average="macro")
    for _ in range(2):
        improved = False
        for class_index in range(len(scales)):
            current_scale = scales[class_index]
            class_best_scale = current_scale
            class_best_score = best_score
            for factor in factors:
                candidate_scale = np.clip(current_scale * factor, 0.25, 4.0)
                candidate_scales = scales.copy()
                candidate_scales[class_index] = candidate_scale
                predictions = np.argmax(p_search * candidate_scales.reshape(1, -1), axis=1)
                score = f1_score(y_search, predictions, average="macro")
                if score > class_best_score:
                    class_best_score = score
                    class_best_scale = candidate_scale
            if class_best_scale != current_scale:
                scales[class_index] = class_best_scale
                best_score = class_best_score
                improved = True
        if not improved:
            break
    return scales


def blend_probabilities(direct, hierarchical, direct_weight, mode):
    if mode == "geometric":
        blended = np.exp(
            direct_weight * np.log(np.clip(direct, 1e-8, 1.0))
            + (1.0 - direct_weight) * np.log(np.clip(hierarchical, 1e-8, 1.0))
        )
        return blended / np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)
    return direct_weight * direct + (1.0 - direct_weight) * hierarchical


def search_blend(y_true, direct, hierarchical):
    rows = []
    for mode in ["arithmetic", "geometric"]:
        for direct_weight in BLEND_WEIGHTS:
            blended = blend_probabilities(direct, hierarchical, direct_weight, mode)
            score = f1_score(y_true, np.argmax(blended, axis=1), average="macro")
            rows.append(
                {
                    "mode": mode,
                    "direct_weight": direct_weight,
                    "hierarchical_weight": 1.0 - direct_weight,
                    "macro_f1": score,
                }
            )
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    return results.iloc[0], results


def save_class_metrics(y_true, probabilities, encoder, path):
    predicted = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=np.arange(len(encoder.classes_)),
        zero_division=0,
    )
    pd.DataFrame(
        {
            "class": encoder.classes_,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).to_csv(path, index=False)


print(f"\n[{VERSION}] Starting direct + hierarchical pipeline...")
t_start = time.time()
processor = HanwooDataProcessorV8(DATA_DIR)
processor.load_auxiliary_data()

print(f"\n[{VERSION}] Loading data...")
train_raw = pd.read_csv(f"{DATA_DIR}/hanwoo_train.csv")
test_raw = pd.read_csv(f"{DATA_DIR}/test_hanwoo.csv")
train = processor.transform(train_raw, is_train=True)
test_base = processor.transform(test_raw, is_train=False)
del train_raw
gc.collect()

full_aug, _, _ = attach_group_counts(train, train.iloc[:0].copy())
TARGET_RELATED = {
    "BACKFAT", "REA", "WINDEX", "WGRADE", "INSFAT", "YUKSAK", "FATSAK",
    "TISSUE", "GROWTH", "COST_AMT",
}
NON_FEATURE = TARGET_RELATED | {
    "LAST_GRADE", "target_q", "target_y", "FARM_UNIQUE_NO", "grade_score",
}
FEATURES = [
    column
    for column in full_aug.columns
    if column not in NON_FEATURE
    and (
        column.endswith("_group_count")
        or (
            column in test_base.columns
            and (is_numeric_dtype(full_aug[column]) or isinstance(full_aug[column].dtype, pd.CategoricalDtype))
        )
    )
]
CATEGORICAL_FEATURES = [column for column in processor.CATEGORICAL_COLUMNS if column in FEATURES]
CATEGORY_LEVELS = {
    column: sorted(
        set(train[column].astype("string").fillna("__MISSING__").unique().tolist())
        | {"__MISSING__", "__UNKNOWN__"}
    )
    for column in CATEGORICAL_FEATURES
}
print(f"  Features ({len(FEATURES)}): {FEATURES}")
print(f"  Categorical ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")

final_encoder = LabelEncoder().fit(processor.FINAL_GRADES)
q_encoder = LabelEncoder().fit(processor.QUALITY_ORDER)
y_encoder = LabelEncoder().fit(processor.YIELD_ORDER)
y_final = final_encoder.transform(train["LAST_GRADE"].astype(str))
y_quality = q_encoder.transform(train["target_q"].astype(str))
y_yield = y_encoder.transform(train["target_y"].astype(str))
is_graded = train["target_q"].astype(str).to_numpy() != "등외"
groups = train["FARM_UNIQUE_NO"].astype(str)

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
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 3000,
}
CAT_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 2500,
    "learning_rate": 0.04,
    "depth": 9,
    "l2_leaf_reg": 7.0,
    "random_strength": 0.75,
    "random_seed": 42,
    "task_type": CATBOOST_TASK_TYPE,
    "allow_writing_files": False,
    "verbose": 100,
}

direct_oof = np.zeros((len(train), len(final_encoder.classes_)), dtype="float32")
hierarchical_oof = np.zeros_like(direct_oof)
direct_test = np.zeros((len(test_base), len(final_encoder.classes_)), dtype="float32")
hierarchical_test = np.zeros_like(direct_test)
direct_scores = []
hierarchical_scores = []
lgb_importance = np.zeros(len(FEATURES), dtype="float64")

splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(splitter.split(train, y_final, groups=groups), start=1):
    print(f"\n[Fold {fold}/{N_SPLITS}] Preparing data...")
    tr_fold = train.iloc[tr_idx].copy()
    val_fold = train.iloc[val_idx].copy()
    tr_aug, val_aug, test_aug = attach_group_counts(tr_fold, val_fold, test_base)
    X_tr = prepare_features(tr_aug, FEATURES, CATEGORICAL_FEATURES, CATEGORY_LEVELS)
    X_val = prepare_features(val_aug, FEATURES, CATEGORICAL_FEATURES, CATEGORY_LEVELS)
    X_test = prepare_features(test_aug, FEATURES, CATEGORICAL_FEATURES, CATEGORY_LEVELS)

    print("  Training direct LightGBM...")
    direct_model = lgb.LGBMClassifier(**LGB_PARAMS, num_class=len(final_encoder.classes_))
    direct_model.fit(
        X_tr,
        y_final[tr_idx],
        eval_set=[(X_val, y_final[val_idx])],
        eval_metric=lgb_macro_f1,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(period=100)],
    )
    direct_val = direct_model.predict_proba(X_val).astype("float32")
    direct_oof[val_idx] = direct_val
    direct_test += direct_model.predict_proba(X_test).astype("float32") / N_SPLITS
    direct_score = f1_score(y_final[val_idx], np.argmax(direct_val, axis=1), average="macro")
    direct_scores.append(direct_score)
    lgb_importance += direct_model.feature_importances_ / N_SPLITS
    print(f"  Direct fold Macro F1: {direct_score:.4f}")
    del direct_model
    gc.collect()

    X_tr_cat = prepare_catboost_features(X_tr, CATEGORICAL_FEATURES)
    X_val_cat = prepare_catboost_features(X_val, CATEGORICAL_FEATURES)
    X_test_cat = prepare_catboost_features(X_test, CATEGORICAL_FEATURES)

    print("  Training CatBoost quality model...")
    quality_model = CatBoostClassifier(**CAT_PARAMS)
    quality_model.fit(
        X_tr_cat,
        y_quality[tr_idx],
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_val_cat, y_quality[val_idx]),
        early_stopping_rounds=200,
        use_best_model=True,
    )
    quality_val = quality_model.predict_proba(X_val_cat)
    quality_test = quality_model.predict_proba(X_test_cat)
    del quality_model
    gc.collect()

    print("  Training CatBoost yield model...")
    tr_graded = is_graded[tr_idx]
    val_graded = is_graded[val_idx]
    yield_model = CatBoostClassifier(**CAT_PARAMS)
    yield_model.fit(
        X_tr_cat.loc[tr_graded],
        y_yield[tr_idx][tr_graded],
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_val_cat.loc[val_graded], y_yield[val_idx][val_graded]),
        early_stopping_rounds=200,
        use_best_model=True,
    )
    yield_val = yield_model.predict_proba(X_val_cat)
    yield_test = yield_model.predict_proba(X_test_cat)
    del yield_model
    gc.collect()

    hierarchical_val = combine_hierarchical_probabilities(
        quality_val, yield_val, q_encoder, y_encoder, final_encoder
    )
    hierarchical_oof[val_idx] = hierarchical_val
    hierarchical_test += combine_hierarchical_probabilities(
        quality_test, yield_test, q_encoder, y_encoder, final_encoder
    ) / N_SPLITS
    hierarchical_score = f1_score(
        y_final[val_idx], np.argmax(hierarchical_val, axis=1), average="macro"
    )
    hierarchical_scores.append(hierarchical_score)
    print(f"  Hierarchical fold Macro F1: {hierarchical_score:.4f}")

    del (
        tr_fold, val_fold, tr_aug, val_aug, test_aug, X_tr, X_val, X_test,
        X_tr_cat, X_val_cat, X_test_cat, quality_val, quality_test, yield_val,
        yield_test, hierarchical_val,
    )
    gc.collect()

direct_raw_f1 = f1_score(y_final, np.argmax(direct_oof, axis=1), average="macro")
hierarchical_raw_f1 = f1_score(y_final, np.argmax(hierarchical_oof, axis=1), average="macro")
direct_scales = optimize_class_scales(y_final, direct_oof)
hierarchical_scales = optimize_class_scales(y_final, hierarchical_oof)
direct_oof_cal = apply_class_scales(direct_oof, direct_scales)
hierarchical_oof_cal = apply_class_scales(hierarchical_oof, hierarchical_scales)
direct_test_cal = apply_class_scales(direct_test, direct_scales)
hierarchical_test_cal = apply_class_scales(hierarchical_test, hierarchical_scales)

best_blend, blend_results = search_blend(y_final, direct_oof_cal, hierarchical_oof_cal)
blend_results.to_csv(f"{OUT_DIR}/blend_search.csv", index=False)
pd.DataFrame(
    {
        "class": final_encoder.classes_,
        "direct_scale": direct_scales,
        "hierarchical_scale": hierarchical_scales,
    }
).to_csv(f"{OUT_DIR}/class_scales.csv", index=False)

best_oof = blend_probabilities(
    direct_oof_cal,
    hierarchical_oof_cal,
    float(best_blend["direct_weight"]),
    best_blend["mode"],
)
best_test = blend_probabilities(
    direct_test_cal,
    hierarchical_test_cal,
    float(best_blend["direct_weight"]),
    best_blend["mode"],
)
best_f1 = f1_score(y_final, np.argmax(best_oof, axis=1), average="macro")

print(f"\n[{VERSION}] Direct raw OOF Macro F1: {direct_raw_f1:.4f}")
print(f"[{VERSION}] Hierarchical raw OOF Macro F1: {hierarchical_raw_f1:.4f}")
print(f"[{VERSION}] Best calibrated blend OOF Macro F1: {best_f1:.4f}")
print(
    f"[{VERSION}] Blend: mode={best_blend['mode']}, "
    f"direct={best_blend['direct_weight']:.2f}, "
    f"hierarchical={best_blend['hierarchical_weight']:.2f}"
)
print(f"[{VERSION}] Direct folds: {', '.join(f'{score:.4f}' for score in direct_scores)}")
print(f"[{VERSION}] Hierarchical folds: {', '.join(f'{score:.4f}' for score in hierarchical_scores)}")

save_class_metrics(y_final, best_oof, final_encoder, f"{OUT_DIR}/oof_class_metrics.csv")
if SAVE_OOF:
    np.savez_compressed(
        f"{OUT_DIR}/oof_predictions.npz",
        y_true=y_final.astype("int8"),
        direct=direct_oof.astype("float16"),
        hierarchical=hierarchical_oof.astype("float16"),
    )

test_raw["LAST_GRADE"] = final_encoder.inverse_transform(np.argmax(best_test, axis=1))
out_path = f"{OUT_DIR}/260418.csv"
test_raw.to_csv(out_path, index=False, encoding="utf-8-sig")
pd.DataFrame({"feature": FEATURES, "importance": lgb_importance}).sort_values(
    "importance", ascending=False
).to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
print(f"  Results saved to: {out_path}")
print(f"  Total Time: {elapsed(t_start)}")
