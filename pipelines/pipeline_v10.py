"""V10: fold-safe pedigree priors + direct/hierarchical ensemble + honest calibration."""

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
VERSION = "v10"
OUT_DIR = os.path.join(BASE_DIR, "submissions", VERSION)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
TARGET_SMOOTHING = 100.0
TARGET_STAT_GROUPS = [("KPN_NO", "kpn"), ("stn", "stn")]
COUNT_GROUPS = [("KPN_NO", "kpn"), ("stn", "stn"), ("sido", "sido"), ("sigungu", "sigungu")]
CATBOOST_TASK_TYPE = os.getenv("CATBOOST_TASK_TYPE", "CPU")
SAVE_OOF = os.getenv("SAVE_OOF", "1") == "1"


def elapsed(start):
    return f"{(time.time() - start) / 60:.1f}min"


def normalized_keys(series):
    return series.astype("string").fillna("__MISSING__")


def attach_fold_statistics(train_fold, val_fold, test_fold=None):
    """Add counts and smoothed component priors using only the current training fold.

    Training rows receive leave-one-out statistics. Validation and test rows never
    use their own targets. This makes the features valid for unseen-farm CV.
    """
    outputs = [train_fold.copy(), val_fold.copy()]
    if test_fold is not None:
        outputs.append(test_fold.copy())

    for group_col, prefix in COUNT_GROUPS:
        if group_col not in train_fold.columns:
            continue
        train_keys = normalized_keys(train_fold[group_col])
        counts = train_keys.value_counts(dropna=False)
        for output_index, output in enumerate(outputs):
            keys = normalized_keys(output[group_col])
            values = keys.map(counts).fillna(0).to_numpy(dtype="float32")
            if output_index == 0:
                values = np.maximum(values - 1.0, 0.0)
            output[f"{prefix}_group_count"] = values
            output[f"{prefix}_group_log_count"] = np.log1p(values).astype("float32")

    q_classes = sorted(train_fold["_q_idx"].unique())
    y_classes = sorted(train_fold.loc[train_fold["_graded"], "_y_idx"].unique())
    q_prior = train_fold["_q_idx"].value_counts(normalize=True).to_dict()
    graded_train = train_fold.loc[train_fold["_graded"]]
    y_prior = graded_train["_y_idx"].value_counts(normalize=True).to_dict()

    for group_col, prefix in TARGET_STAT_GROUPS:
        if group_col not in train_fold.columns:
            continue
        train_keys = normalized_keys(train_fold[group_col])
        q_count = train_keys.value_counts(dropna=False)
        graded_keys = train_keys.loc[train_fold["_graded"]]
        y_count = graded_keys.value_counts(dropna=False)

        q_sums = {
            cls: train_keys.loc[train_fold["_q_idx"] == cls].value_counts(dropna=False)
            for cls in q_classes
        }
        y_sums = {
            cls: graded_keys.loc[graded_train["_y_idx"] == cls].value_counts(dropna=False)
            for cls in y_classes
        }

        for output_index, output in enumerate(outputs):
            keys = normalized_keys(output[group_col])
            qc = keys.map(q_count).fillna(0).to_numpy(dtype="float32")
            yc = keys.map(y_count).fillna(0).to_numpy(dtype="float32")
            if output_index == 0:
                qc = np.maximum(qc - 1.0, 0.0)
                yc = np.maximum(yc - output["_graded"].to_numpy(dtype="float32"), 0.0)

            for cls in q_classes:
                numerator = keys.map(q_sums[cls]).fillna(0).to_numpy(dtype="float32")
                if output_index == 0:
                    numerator -= (output["_q_idx"].to_numpy() == cls)
                output[f"{prefix}_q{cls}_prior"] = (
                    (numerator + TARGET_SMOOTHING * q_prior[cls])
                    / (qc + TARGET_SMOOTHING)
                ).astype("float32")

            for cls in y_classes:
                numerator = keys.map(y_sums[cls]).fillna(0).to_numpy(dtype="float32")
                if output_index == 0:
                    numerator -= (
                        output["_graded"].to_numpy()
                        & (output["_y_idx"].to_numpy() == cls)
                    )
                output[f"{prefix}_y{cls}_prior"] = (
                    (numerator + TARGET_SMOOTHING * y_prior[cls])
                    / (yc + TARGET_SMOOTHING)
                ).astype("float32")

    return outputs[0], outputs[1], outputs[2] if test_fold is not None else None


def prepare_features(frame, feature_cols, categorical_cols, category_levels):
    result = frame[feature_cols].copy()
    for column in categorical_cols:
        values = result[column].astype("string").fillna("__MISSING__")
        values = values.where(values.isin(category_levels[column]), "__UNKNOWN__")
        result[column] = values.astype(pd.CategoricalDtype(categories=category_levels[column]))
    for column in feature_cols:
        if column not in categorical_cols:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(-999).astype("float32")
    return result


def prepare_catboost_features(frame, categorical_cols):
    result = frame.copy()
    for column in categorical_cols:
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    return result


def lgb_macro_f1(y_true, probabilities):
    return "macro_f1", f1_score(y_true, np.argmax(probabilities, axis=1), average="macro"), True


def combine_hierarchical_probabilities(q_proba, y_proba, q_encoder, y_encoder, final_encoder):
    result = np.zeros((len(q_proba), len(final_encoder.classes_)), dtype="float32")
    qi = {label: i for i, label in enumerate(q_encoder.classes_)}
    yi = {label: i for i, label in enumerate(y_encoder.classes_)}
    fi = {label: i for i, label in enumerate(final_encoder.classes_)}
    outlier = "등외"
    for grade in final_encoder.classes_:
        if grade == outlier:
            result[:, fi[grade]] = q_proba[:, qi[outlier]]
        else:
            result[:, fi[grade]] = q_proba[:, qi[grade[:-1]]] * y_proba[:, yi[grade[-1]]]
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def apply_class_scales(probabilities, scales):
    result = probabilities * scales.reshape(1, -1)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def optimize_class_scales(y_true, probabilities, max_rows=500_000):
    """Coordinate-search class thresholds on a prevalence-preserving subset."""
    if len(y_true) > max_rows:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(y_true), max_rows, replace=False)
        y_search, p_search = y_true[indices], probabilities[indices]
    else:
        y_search, p_search = y_true, probabilities

    scales = np.ones(probabilities.shape[1], dtype="float32")
    factors = [0.85, 0.925, 1.0, 1.08, 1.18]
    best = f1_score(y_search, np.argmax(p_search, axis=1), average="macro")
    for _ in range(3):
        improved = False
        for cls in range(len(scales)):
            current = scales[cls]
            candidates = np.unique(np.clip(current * np.asarray(factors), 0.4, 2.5))
            class_best = current
            for candidate in candidates:
                trial = scales.copy()
                trial[cls] = candidate
                score = f1_score(y_search, np.argmax(p_search * trial, axis=1), average="macro")
                if score > best + 1e-7:
                    best, class_best = score, candidate
            if class_best != current:
                scales[cls] = class_best
                improved = True
        if not improved:
            break
    return scales


def blend_probabilities(direct, hierarchical, direct_weight, mode):
    if mode == "geometric":
        result = np.exp(
            direct_weight * np.log(np.clip(direct, 1e-8, 1.0))
            + (1.0 - direct_weight) * np.log(np.clip(hierarchical, 1e-8, 1.0))
        )
        return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)
    return direct_weight * direct + (1.0 - direct_weight) * hierarchical


def select_blend(y_true, direct, hierarchical):
    rows = []
    for mode in ["arithmetic", "geometric"]:
        for weight in np.linspace(0.0, 1.0, 41):
            score = f1_score(
                y_true,
                np.argmax(blend_probabilities(direct, hierarchical, weight, mode), axis=1),
                average="macro",
            )
            rows.append((score, mode, weight))
    return max(rows, key=lambda row: row[0])


def fit_calibrator(y_true, direct, hierarchical):
    direct_scales = optimize_class_scales(y_true, direct)
    hierarchical_scales = optimize_class_scales(y_true, hierarchical)
    direct_cal = apply_class_scales(direct, direct_scales)
    hierarchical_cal = apply_class_scales(hierarchical, hierarchical_scales)
    score, mode, weight = select_blend(y_true, direct_cal, hierarchical_cal)
    return {
        "direct_scales": direct_scales,
        "hierarchical_scales": hierarchical_scales,
        "mode": mode,
        "direct_weight": weight,
        "fit_score": score,
    }


def apply_calibrator(direct, hierarchical, calibrator):
    return blend_probabilities(
        apply_class_scales(direct, calibrator["direct_scales"]),
        apply_class_scales(hierarchical, calibrator["hierarchical_scales"]),
        calibrator["direct_weight"],
        calibrator["mode"],
    )


def save_class_metrics(y_true, probabilities, encoder, path):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, np.argmax(probabilities, axis=1), labels=np.arange(len(encoder.classes_)), zero_division=0
    )
    pd.DataFrame({"class": encoder.classes_, "precision": precision, "recall": recall, "f1": f1, "support": support}).to_csv(path, index=False)


print(f"\n[{VERSION}] Starting fold-safe direct + hierarchical pipeline...")
started = time.time()
processor = HanwooDataProcessorV8(DATA_DIR)
processor.load_auxiliary_data()
train_raw = pd.read_csv(f"{DATA_DIR}/hanwoo_train.csv")
test_raw = pd.read_csv(f"{DATA_DIR}/test_hanwoo.csv")
train = processor.transform(train_raw, is_train=True)
test_base = processor.transform(test_raw, is_train=False)
del train_raw
gc.collect()

final_encoder = LabelEncoder().fit(processor.FINAL_GRADES)
q_encoder = LabelEncoder().fit(processor.QUALITY_ORDER)
y_encoder = LabelEncoder().fit(processor.YIELD_ORDER)
y_final = final_encoder.transform(train["LAST_GRADE"].astype(str))
y_quality = q_encoder.transform(train["target_q"].astype(str))
y_yield = y_encoder.transform(train["target_y"].astype(str))
is_graded = train["target_q"].astype(str).to_numpy() != "등외"
groups = train["FARM_UNIQUE_NO"].astype(str)

train["_q_idx"] = y_quality
train["_y_idx"] = y_yield
train["_graded"] = is_graded
test_base["_q_idx"] = -1
test_base["_y_idx"] = -1
test_base["_graded"] = False

prototype, _, _ = attach_fold_statistics(train, train.iloc[:0].copy())
target_related = {"BACKFAT", "REA", "WINDEX", "WGRADE", "INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH", "COST_AMT"}
non_features = target_related | {"LAST_GRADE", "target_q", "target_y", "FARM_UNIQUE_NO", "grade_score", "_q_idx", "_y_idx", "_graded"}
features = [
    column for column in prototype.columns
    if column not in non_features
    and (column in test_base.columns or column.endswith(("_count", "_prior")))
    and (is_numeric_dtype(prototype[column]) or isinstance(prototype[column].dtype, pd.CategoricalDtype))
]
categorical_features = [column for column in processor.CATEGORICAL_COLUMNS if column in features]
category_levels = {
    column: sorted(set(train[column].astype("string").fillna("__MISSING__").unique()) | {"__MISSING__", "__UNKNOWN__"})
    for column in categorical_features
}
del prototype
gc.collect()
print(f"[{VERSION}] Features: {len(features)} ({len(categorical_features)} categorical)")

lgb_params = dict(
    objective="multiclass", metric="None", learning_rate=0.03, num_leaves=127,
    min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.1, lambda_l2=2.0, max_depth=12, random_state=42, verbose=-1,
    n_estimators=3000,
)
cat_params = dict(
    loss_function="MultiClass", eval_metric="MultiClass", iterations=2500,
    learning_rate=0.04, depth=9, l2_leaf_reg=7.0, random_strength=0.75,
    random_seed=42, task_type=CATBOOST_TASK_TYPE, allow_writing_files=False, verbose=100,
)

n_classes = len(final_encoder.classes_)
direct_oof = np.zeros((len(train), n_classes), dtype="float32")
hierarchical_oof = np.zeros_like(direct_oof)
direct_test = np.zeros((len(test_base), n_classes), dtype="float32")
hierarchical_test = np.zeros_like(direct_test)
fold_ids = np.full(len(train), -1, dtype="int8")
importance = np.zeros(len(features), dtype="float64")
fold_rows = []

splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(splitter.split(train, y_final, groups=groups)):
    print(f"\n[Fold {fold + 1}/{N_SPLITS}] Building fold-safe statistics...")
    tr_aug, val_aug, test_aug = attach_fold_statistics(train.iloc[tr_idx], train.iloc[val_idx], test_base)
    x_tr = prepare_features(tr_aug, features, categorical_features, category_levels)
    x_val = prepare_features(val_aug, features, categorical_features, category_levels)
    x_test = prepare_features(test_aug, features, categorical_features, category_levels)
    fold_ids[val_idx] = fold

    direct_model = lgb.LGBMClassifier(**lgb_params, num_class=n_classes)
    direct_model.fit(
        x_tr, y_final[tr_idx], eval_set=[(x_val, y_final[val_idx])], eval_metric=lgb_macro_f1,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )
    direct_val = direct_model.predict_proba(x_val).astype("float32")
    direct_oof[val_idx] = direct_val
    direct_test += direct_model.predict_proba(x_test).astype("float32") / N_SPLITS
    importance += direct_model.feature_importances_ / N_SPLITS
    del direct_model
    gc.collect()

    x_tr_cat = prepare_catboost_features(x_tr, categorical_features)
    x_val_cat = prepare_catboost_features(x_val, categorical_features)
    x_test_cat = prepare_catboost_features(x_test, categorical_features)
    quality_model = CatBoostClassifier(**cat_params)
    quality_model.fit(
        x_tr_cat, y_quality[tr_idx], cat_features=categorical_features,
        eval_set=(x_val_cat, y_quality[val_idx]), early_stopping_rounds=200, use_best_model=True,
    )
    quality_val = quality_model.predict_proba(x_val_cat)
    quality_test = quality_model.predict_proba(x_test_cat)
    del quality_model
    gc.collect()

    train_graded = is_graded[tr_idx]
    val_graded = is_graded[val_idx]
    yield_model = CatBoostClassifier(**cat_params)
    yield_model.fit(
        x_tr_cat.loc[train_graded], y_yield[tr_idx][train_graded], cat_features=categorical_features,
        eval_set=(x_val_cat.loc[val_graded], y_yield[val_idx][val_graded]),
        early_stopping_rounds=200, use_best_model=True,
    )
    yield_val = yield_model.predict_proba(x_val_cat)
    yield_test = yield_model.predict_proba(x_test_cat)
    del yield_model

    hierarchical_val = combine_hierarchical_probabilities(quality_val, yield_val, q_encoder, y_encoder, final_encoder)
    hierarchical_oof[val_idx] = hierarchical_val
    hierarchical_test += combine_hierarchical_probabilities(quality_test, yield_test, q_encoder, y_encoder, final_encoder) / N_SPLITS
    fold_rows.append({
        "fold": fold + 1,
        "direct_f1": f1_score(y_final[val_idx], direct_val.argmax(1), average="macro"),
        "hierarchical_f1": f1_score(y_final[val_idx], hierarchical_val.argmax(1), average="macro"),
    })
    print(f"  Direct={fold_rows[-1]['direct_f1']:.4f}, hierarchical={fold_rows[-1]['hierarchical_f1']:.4f}")
    del tr_aug, val_aug, test_aug, x_tr, x_val, x_test, x_tr_cat, x_val_cat, x_test_cat
    del quality_val, quality_test, yield_val, yield_test, hierarchical_val, direct_val
    gc.collect()

# Every reported calibrated OOF prediction is calibrated without seeing its fold.
calibrated_oof = np.zeros_like(direct_oof)
calibrated_test = np.zeros_like(direct_test)
calibration_rows = []
scale_rows = []
for heldout_fold in range(N_SPLITS):
    fit_mask = fold_ids != heldout_fold
    heldout_mask = fold_ids == heldout_fold
    calibrator = fit_calibrator(
        y_final[fit_mask], direct_oof[fit_mask], hierarchical_oof[fit_mask]
    )
    calibrated_oof[heldout_mask] = apply_calibrator(
        direct_oof[heldout_mask], hierarchical_oof[heldout_mask], calibrator
    )
    calibrated_test += apply_calibrator(direct_test, hierarchical_test, calibrator) / N_SPLITS
    heldout_score = f1_score(y_final[heldout_mask], calibrated_oof[heldout_mask].argmax(1), average="macro")
    calibration_rows.append({
        "heldout_fold": heldout_fold + 1, "mode": calibrator["mode"],
        "direct_weight": calibrator["direct_weight"], "calibration_fit_f1": calibrator["fit_score"],
        "heldout_f1": heldout_score,
    })
    for class_index, class_name in enumerate(final_encoder.classes_):
        scale_rows.append({
            "heldout_fold": heldout_fold + 1, "class": class_name,
            "direct_scale": calibrator["direct_scales"][class_index],
            "hierarchical_scale": calibrator["hierarchical_scales"][class_index],
        })

raw_direct_f1 = f1_score(y_final, direct_oof.argmax(1), average="macro")
raw_hierarchical_f1 = f1_score(y_final, hierarchical_oof.argmax(1), average="macro")
honest_f1 = f1_score(y_final, calibrated_oof.argmax(1), average="macro")
print(f"\n[{VERSION}] Raw direct OOF: {raw_direct_f1:.4f}")
print(f"[{VERSION}] Raw hierarchical OOF: {raw_hierarchical_f1:.4f}")
print(f"[{VERSION}] Cross-fitted calibrated OOF: {honest_f1:.4f}")

pd.DataFrame(fold_rows).to_csv(f"{OUT_DIR}/fold_metrics.csv", index=False)
pd.DataFrame(calibration_rows).to_csv(f"{OUT_DIR}/calibration_folds.csv", index=False)
pd.DataFrame(scale_rows).to_csv(f"{OUT_DIR}/class_scales.csv", index=False)
save_class_metrics(y_final, calibrated_oof, final_encoder, f"{OUT_DIR}/oof_class_metrics.csv")
pd.DataFrame({"feature": features, "importance": importance}).sort_values("importance", ascending=False).to_csv(
    f"{OUT_DIR}/feature_importance.csv", index=False
)
if SAVE_OOF:
    np.savez_compressed(
        f"{OUT_DIR}/oof_predictions.npz", y_true=y_final.astype("int8"), fold_id=fold_ids,
        direct=direct_oof.astype("float16"), hierarchical=hierarchical_oof.astype("float16"),
        calibrated=calibrated_oof.astype("float16"),
    )

test_raw["LAST_GRADE"] = final_encoder.inverse_transform(calibrated_test.argmax(1))
output_path = f"{OUT_DIR}/260418.csv"
test_raw.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"[{VERSION}] Saved {output_path}; total time {elapsed(started)}")
