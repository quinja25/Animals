"""V11: specialized LightGBM, CatBoost, and ordinal-hurdle XGBoost ensemble."""

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
from xgboost import XGBClassifier

from data_processor_v8 import HanwooDataProcessorV8

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VERSION = "v11"
OUT_DIR = os.path.join(BASE_DIR, "submissions", VERSION)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
GROUP_SPECS = [("KPN_NO", "kpn"), ("stn", "stn"), ("sido", "sido"), ("sigungu", "sigungu")]
CATBOOST_TASK_TYPE = os.getenv("CATBOOST_TASK_TYPE", "GPU")
XGBOOST_DEVICE = os.getenv("XGBOOST_DEVICE", "cuda")
SAVE_OOF = os.getenv("SAVE_OOF", "1") == "1"
SMOKE_STAGE = os.getenv("V11_SMOKE_STAGE", "").lower()  # "lgb" or "components"


def elapsed(start):
    return f"{(time.time() - start) / 60:.1f}min"


def normalized_keys(series):
    return series.astype("string").fillna("__MISSING__")


def attach_fold_features(train_fold, val_fold, test_fold=None):
    """Create target-free fold statistics and cohort-normalized physical features."""
    outputs = [train_fold.copy(), val_fold.copy()]
    if test_fold is not None:
        outputs.append(test_fold.copy())

    for group_col, prefix in GROUP_SPECS:
        if group_col not in train_fold.columns:
            continue
        train_keys = normalized_keys(train_fold[group_col])
        counts = train_keys.value_counts(dropna=False)
        for output_index, output in enumerate(outputs):
            values = normalized_keys(output[group_col]).map(counts).fillna(0).to_numpy(dtype="float32")
            if output_index == 0:
                values = np.maximum(values - 1.0, 0.0)
            output[f"{prefix}_group_count"] = values
            output[f"{prefix}_group_log_count"] = np.log1p(values).astype("float32")

    # Weight has different meaning across sex and slaughter age. These target-free
    # fold statistics express whether an animal is unusually heavy for its cohort.
    age_bin = (pd.to_numeric(train_fold["AGE"], errors="coerce").fillna(-1) // 2 * 2).astype(int)
    sex = normalized_keys(train_fold["sex_code"])
    cohort_source = pd.DataFrame(
        {"sex": sex, "age_bin": age_bin, "weight": pd.to_numeric(train_fold["WEIGHT"], errors="coerce")},
        index=train_fold.index,
    )
    cohort_stats = cohort_source.groupby(["sex", "age_bin"], dropna=False)["weight"].agg(
        cohort_weight_mean="mean",
        cohort_weight_std="std",
        cohort_weight_median="median",
        cohort_weight_q25=lambda values: values.quantile(0.25),
        cohort_weight_q75=lambda values: values.quantile(0.75),
        cohort_count="count",
    )
    global_weight = float(cohort_source["weight"].mean())
    global_std = float(cohort_source["weight"].std())

    for output in outputs:
        output_age_bin = (pd.to_numeric(output["AGE"], errors="coerce").fillna(-1) // 2 * 2).astype(int)
        output_sex = normalized_keys(output["sex_code"])
        lookup = pd.MultiIndex.from_arrays([output_sex, output_age_bin], names=["sex", "age_bin"])
        mapped = cohort_stats.reindex(lookup)
        weight = pd.to_numeric(output["WEIGHT"], errors="coerce").to_numpy(dtype="float32")
        mean = mapped["cohort_weight_mean"].fillna(global_weight).to_numpy(dtype="float32")
        std = mapped["cohort_weight_std"].fillna(global_std).clip(lower=5.0).to_numpy(dtype="float32")
        median = mapped["cohort_weight_median"].fillna(global_weight).to_numpy(dtype="float32")
        q25 = mapped["cohort_weight_q25"].fillna(global_weight - global_std).to_numpy(dtype="float32")
        q75 = mapped["cohort_weight_q75"].fillna(global_weight + global_std).to_numpy(dtype="float32")
        output["cohort_count"] = mapped["cohort_count"].fillna(0).to_numpy(dtype="float32")
        output["weight_cohort_residual"] = weight - mean
        output["weight_cohort_z"] = (weight - mean) / std
        output["weight_cohort_median_residual"] = weight - median
        output["weight_cohort_iqr_position"] = (weight - q25) / np.maximum(q75 - q25, 5.0)
        output["xgb_sex_code"] = pd.to_numeric(output["sex_code"].astype("string"), errors="coerce").fillna(-1).astype("float32")
        output["age_rearing_gap"] = (
            pd.to_numeric(output["AGE"], errors="coerce")
            - pd.to_numeric(output["rearing_months"], errors="coerce")
        ).astype("float32")

        rearing = pd.to_numeric(output["rearing_months"], errors="coerce").fillna(0).clip(lower=1)
        for source, target in [
            ("s_heat_3m", "heat_3m_per_rearing_month"),
            ("s_heat_6m", "heat_6m_per_rearing_month"),
            ("s_heat_12m", "heat_12m_per_rearing_month"),
        ]:
            if source in output.columns:
                output[target] = (pd.to_numeric(output[source], errors="coerce") / rearing).astype("float32")

        output["has_kpn"] = (normalized_keys(output["KPN_NO"]) != "__MISSING__").astype("float32")
        bv_columns = [column for column in output.columns if column.startswith("kpn_") and column.endswith("_bv")]
        if bv_columns:
            output["has_kpn_bv"] = output[bv_columns].notna().any(axis=1).astype("float32")
            output["kpn_bv_missing_count"] = output[bv_columns].isna().sum(axis=1).astype("float32")
        else:
            output["has_kpn_bv"] = np.float32(0.0)
            output["kpn_bv_missing_count"] = np.float32(0.0)
        weather_columns = [column for column in output.columns if column.startswith("s_")]
        output["weather_missing_count"] = output[weather_columns].isna().sum(axis=1).astype("float32")

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


def prepare_xgboost_features(frame, feature_cols):
    return frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-999).astype("float32")


def lgb_macro_f1(y_true, probabilities):
    return "macro_f1", f1_score(y_true, np.argmax(probabilities, axis=1), average="macro"), True


def build_xgb_classifier(positive_rate):
    # Mild square-root balancing makes the lower-frequency boundary visible
    # without destroying probability ranking as full inverse weighting can.
    scale_pos_weight = float(np.sqrt((1.0 - positive_rate) / max(positive_rate, 1e-6)))
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=1800,
        learning_rate=0.035,
        max_depth=8,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        max_bin=256,
        tree_method="hist",
        device=XGBOOST_DEVICE,
        eval_metric="logloss",
        early_stopping_rounds=120,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )


def combine_yield_hurdle(c_probability, a_given_non_c):
    """Reconstruct P(A), P(B), P(C) from ordered binary boundaries."""
    p_c = np.clip(c_probability, 1e-6, 1.0 - 1e-6)
    p_non_c = 1.0 - p_c
    p_a = p_non_c * np.clip(a_given_non_c, 1e-6, 1.0 - 1e-6)
    p_b = p_non_c - p_a
    result = np.column_stack([p_a, p_b, p_c]).astype("float32")
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def cumulative_to_ordinal_probabilities(cumulative):
    """Convert P(rank <= k), k=0..3, into five ordered class probabilities."""
    cumulative = np.clip(np.asarray(cumulative, dtype="float32"), 1e-6, 1.0 - 1e-6)
    cumulative = np.maximum.accumulate(cumulative, axis=1)
    result = np.column_stack(
        [
            cumulative[:, 0],
            cumulative[:, 1] - cumulative[:, 0],
            cumulative[:, 2] - cumulative[:, 1],
            cumulative[:, 3] - cumulative[:, 2],
            1.0 - cumulative[:, 3],
        ]
    ).astype("float32")
    result = np.clip(result, 1e-8, 1.0)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def blend_quality_probabilities(cat_quality, ordinal_graded, ordinal_weight, q_encoder, graded_order):
    """Keep CatBoost's outlier gate and blend conditional graded quality probabilities."""
    outlier = "등외"
    q_index = {label: index for index, label in enumerate(q_encoder.classes_)}
    outlier_index = q_index[outlier]
    graded_indices = [q_index[label] for label in graded_order]
    outlier_probability = np.clip(cat_quality[:, outlier_index], 1e-6, 1.0 - 1e-6)
    cat_graded = cat_quality[:, graded_indices]
    cat_graded /= np.maximum(cat_graded.sum(axis=1, keepdims=True), 1e-12)
    conditional = (1.0 - ordinal_weight) * cat_graded + ordinal_weight * ordinal_graded
    result = np.zeros_like(cat_quality, dtype="float32")
    result[:, outlier_index] = outlier_probability
    result[:, graded_indices] = conditional * (1.0 - outlier_probability).reshape(-1, 1)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


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


def optimize_class_scales(y_true, probabilities, max_rows=250_000):
    if len(y_true) > max_rows:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(y_true), max_rows, replace=False)
        y_search, p_search = y_true[indices], probabilities[indices]
    else:
        y_search, p_search = y_true, probabilities
    scales = np.ones(probabilities.shape[1], dtype="float32")
    factors = [0.85, 0.925, 1.0, 1.08, 1.18]
    best = f1_score(y_search, np.argmax(p_search, axis=1), average="macro")
    for _ in range(2):
        improved = False
        for cls in range(len(scales)):
            current = scales[cls]
            class_best = current
            for candidate in np.unique(np.clip(current * np.asarray(factors), 0.4, 2.5)):
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


def fit_calibrator(y_true, direct, hierarchical):
    direct_scales = optimize_class_scales(y_true, direct)
    hierarchical_scales = optimize_class_scales(y_true, hierarchical)
    direct_cal = apply_class_scales(direct, direct_scales)
    hierarchical_cal = apply_class_scales(hierarchical, hierarchical_scales)
    candidates = []
    for mode in ["arithmetic", "geometric"]:
        for weight in np.linspace(0.0, 1.0, 21):
            probabilities = blend_probabilities(direct_cal, hierarchical_cal, weight, mode)
            candidates.append((f1_score(y_true, probabilities.argmax(1), average="macro"), mode, weight))
    score, mode, weight = max(candidates, key=lambda row: row[0])
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


def save_class_metrics(y_true, probabilities, labels, path):
    predicted = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=np.arange(len(labels)), zero_division=0
    )
    pd.DataFrame({"class": labels, "precision": precision, "recall": recall, "f1": f1, "support": support}).to_csv(path, index=False)


print(f"\n[{VERSION}] Starting specialized three-model pipeline...")
print(f"[{VERSION}] CatBoost={CATBOOST_TASK_TYPE}, XGBoost={XGBOOST_DEVICE}, smoke={SMOKE_STAGE or 'off'}")
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
graded_quality_order = ["1++", "1+", "1", "2", "3"]
quality_rank_map = {label: rank for rank, label in enumerate(graded_quality_order)}
y_quality_rank = train["target_q"].astype(str).map(quality_rank_map).fillna(-1).to_numpy(dtype="int8")
groups = train["FARM_UNIQUE_NO"].astype(str)

# A small prototype is enough to discover generated column names and dtypes.
# Building fold aggregates on all 2.4M rows here would duplicate work before CV.
prototype_source = train.iloc[: min(10_000, len(train))].copy()
prototype, _, _ = attach_fold_features(prototype_source, prototype_source.iloc[:0].copy())
target_related = {"BACKFAT", "REA", "WINDEX", "WGRADE", "INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH", "COST_AMT"}
non_features = target_related | {"LAST_GRADE", "target_q", "target_y", "FARM_UNIQUE_NO", "grade_score"}
generated_features = [column for column in prototype.columns if column not in train.columns]
base_features = [
    column for column in train.columns
    if column not in non_features
    and column in test_base.columns
    and (is_numeric_dtype(train[column]) or isinstance(train[column].dtype, pd.CategoricalDtype))
]
lgb_features = base_features + [f"{prefix}_group_count" for _, prefix in GROUP_SPECS]

cat_quality_features = list(dict.fromkeys(
    base_features
    + [f"{prefix}_group_count" for _, prefix in GROUP_SPECS]
    + [column for column in generated_features if column.startswith(("has_", "weather_missing"))]
))
categorical_features = [column for column in processor.CATEGORICAL_COLUMNS if column in cat_quality_features]

xgb_numeric_prefixes = (
    "s_", "kpn_", "weight_", "cohort_", "heat_", "age_", "has_", "weather_",
)
xgb_explicit = {
    "WEIGHT", "AGE", "rearing_months", "density", "death_cnt", "abatt_year", "birth_year",
    "abatt_month_sin", "abatt_month_cos", "birth_month_sin", "birth_month_cos", "xgb_sex_code",
}
xgb_yield_features = [
    column for column in prototype.columns
    if column not in non_features
    and is_numeric_dtype(prototype[column])
    and (column in xgb_explicit or column.startswith(xgb_numeric_prefixes) or column.endswith("_group_count"))
]

category_levels = {
    column: sorted(set(train[column].astype("string").fillna("__MISSING__").unique()) | {"__MISSING__", "__UNKNOWN__"})
    for column in categorical_features
}
del prototype, prototype_source
gc.collect()
print(
    f"[{VERSION}] Feature views: LightGBM={len(lgb_features)}, "
    f"CatBoost-quality={len(cat_quality_features)}, XGBoost-yield={len(xgb_yield_features)}"
)

lgb_params = dict(
    objective="multiclass", metric="None", learning_rate=0.03, num_leaves=127,
    min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.1, lambda_l2=2.0, max_depth=12, random_state=42, verbose=-1,
    n_estimators=3000,
)
cat_params = dict(
    loss_function="MultiClass", eval_metric="MultiClass", iterations=1800,
    learning_rate=0.04, depth=8, l2_leaf_reg=8.0, random_strength=0.75,
    random_seed=42, task_type=CATBOOST_TASK_TYPE, allow_writing_files=False, verbose=100,
)
ordinal_lgb_params = dict(
    objective="binary", metric="binary_logloss", learning_rate=0.04, num_leaves=63,
    min_child_samples=150, feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.2, lambda_l2=4.0, max_depth=10, random_state=42, verbose=-1,
    n_estimators=1800,
)

n_rows, n_test, n_classes = len(train), len(test_base), len(final_encoder.classes_)
direct_oof = np.zeros((n_rows, n_classes), dtype="float32")
hierarchical_oof = np.zeros_like(direct_oof)
quality_oof = np.zeros((n_rows, len(q_encoder.classes_)), dtype="float32")
ordinal_quality_oof = np.zeros((n_rows, len(graded_quality_order)), dtype="float32")
yield_oof = np.zeros((n_rows, len(y_encoder.classes_)), dtype="float32")
direct_test = np.zeros((n_test, n_classes), dtype="float32")
hierarchical_test = np.zeros_like(direct_test)
quality_test_average = np.zeros((n_test, len(q_encoder.classes_)), dtype="float32")
ordinal_quality_test = np.zeros((n_test, len(graded_quality_order)), dtype="float32")
yield_test_average = np.zeros((n_test, len(y_encoder.classes_)), dtype="float32")
fold_ids = np.full(n_rows, -1, dtype="int8")
importance = np.zeros(len(lgb_features), dtype="float64")
component_rows = []

splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(splitter.split(train, y_final, groups=groups)):
    print(f"\n[Fold {fold + 1}/{N_SPLITS}] Building target-free feature views...")
    tr_aug, val_aug, test_aug = attach_fold_features(train.iloc[tr_idx], train.iloc[val_idx], test_base)
    all_features = list(dict.fromkeys(lgb_features + cat_quality_features + xgb_yield_features))
    x_tr = prepare_features(tr_aug, all_features, categorical_features, category_levels)
    x_val = prepare_features(val_aug, all_features, categorical_features, category_levels)
    x_test = prepare_features(test_aug, all_features, categorical_features, category_levels)
    fold_ids[val_idx] = fold

    print("  [1/4] LightGBM direct 16-class model")
    direct_model = lgb.LGBMClassifier(**lgb_params, num_class=n_classes)
    direct_model.fit(
        x_tr[lgb_features], y_final[tr_idx],
        eval_set=[(x_val[lgb_features], y_final[val_idx])], eval_metric=lgb_macro_f1,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )
    direct_val = direct_model.predict_proba(x_val[lgb_features]).astype("float32")
    direct_oof[val_idx] = direct_val
    direct_test += direct_model.predict_proba(x_test[lgb_features]).astype("float32") / N_SPLITS
    importance += direct_model.feature_importances_ / N_SPLITS
    direct_f1 = f1_score(y_final[val_idx], direct_val.argmax(1), average="macro")
    print(f"  Direct Macro F1: {direct_f1:.4f}")
    del direct_model
    gc.collect()
    if SMOKE_STAGE == "lgb":
        print(f"[{VERSION}] LightGBM smoke stage complete.")
        raise SystemExit(0)

    tr_graded = is_graded[tr_idx]
    val_graded = is_graded[val_idx]

    print("  [2/4] CatBoost quality/outlier model")
    x_tr_cat = prepare_catboost_features(x_tr[cat_quality_features], categorical_features)
    x_val_cat = prepare_catboost_features(x_val[cat_quality_features], categorical_features)
    x_test_cat = prepare_catboost_features(x_test[cat_quality_features], categorical_features)
    quality_model = CatBoostClassifier(**cat_params)
    quality_model.fit(
        x_tr_cat, y_quality[tr_idx], cat_features=categorical_features,
        eval_set=(x_val_cat, y_quality[val_idx]), early_stopping_rounds=150, use_best_model=True,
    )
    quality_val = quality_model.predict_proba(x_val_cat).astype("float32")
    quality_test = quality_model.predict_proba(x_test_cat).astype("float32")
    quality_oof[val_idx] = quality_val
    quality_test_average += quality_test / N_SPLITS
    quality_f1 = f1_score(y_quality[val_idx], quality_val.argmax(1), average="macro")
    print(f"  Quality Macro F1: {quality_f1:.4f}")
    del quality_model, x_tr_cat, x_val_cat, x_test_cat
    gc.collect()

    print("  [3/4] LightGBM ordinal quality boundaries")
    ordinal_val_cumulative = np.zeros((len(val_idx), 4), dtype="float32")
    ordinal_test_cumulative = np.zeros((n_test, 4), dtype="float32")
    for threshold in range(4):
        ordinal_train_target = (y_quality_rank[tr_idx][tr_graded] <= threshold).astype("int8")
        ordinal_val_target = (y_quality_rank[val_idx][val_graded] <= threshold).astype("int8")
        ordinal_model = lgb.LGBMClassifier(**ordinal_lgb_params)
        ordinal_model.fit(
            x_tr.loc[tr_graded, lgb_features],
            ordinal_train_target,
            eval_set=[(x_val.loc[val_graded, lgb_features], ordinal_val_target)],
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
        )
        ordinal_val_cumulative[:, threshold] = ordinal_model.predict_proba(x_val[lgb_features])[:, 1]
        ordinal_test_cumulative[:, threshold] = ordinal_model.predict_proba(x_test[lgb_features])[:, 1]
        del ordinal_model
        gc.collect()
    ordinal_val = cumulative_to_ordinal_probabilities(ordinal_val_cumulative)
    ordinal_test = cumulative_to_ordinal_probabilities(ordinal_test_cumulative)
    ordinal_quality_oof[val_idx] = ordinal_val
    ordinal_quality_test += ordinal_test / N_SPLITS
    ordinal_quality_f1 = f1_score(
        y_quality_rank[val_idx][val_graded], ordinal_val[val_graded].argmax(1), average="macro"
    )
    print(f"  Ordinal quality Macro F1: {ordinal_quality_f1:.4f}")

    print("  [4/4] XGBoost ordinal yield hurdle")
    x_tr_xgb = prepare_xgboost_features(x_tr, xgb_yield_features)
    x_val_xgb = prepare_xgboost_features(x_val, xgb_yield_features)
    x_test_xgb = prepare_xgboost_features(x_test, xgb_yield_features)
    c_index = int(np.flatnonzero(y_encoder.classes_ == "C")[0])
    a_index = int(np.flatnonzero(y_encoder.classes_ == "A")[0])

    c_train = (y_yield[tr_idx][tr_graded] == c_index).astype("int8")
    c_val = (y_yield[val_idx][val_graded] == c_index).astype("int8")
    c_model = build_xgb_classifier(float(c_train.mean()))
    c_model.fit(
        x_tr_xgb.loc[tr_graded], c_train,
        eval_set=[(x_val_xgb.loc[val_graded], c_val)], verbose=100,
    )
    c_val_probability = c_model.predict_proba(x_val_xgb)[:, 1]
    c_test_probability = c_model.predict_proba(x_test_xgb)[:, 1]
    del c_model
    gc.collect()

    tr_ab = tr_graded & (y_yield[tr_idx] != c_index)
    val_ab = val_graded & (y_yield[val_idx] != c_index)
    a_train = (y_yield[tr_idx][tr_ab] == a_index).astype("int8")
    a_val = (y_yield[val_idx][val_ab] == a_index).astype("int8")
    ab_model = build_xgb_classifier(float(a_train.mean()))
    ab_model.fit(
        x_tr_xgb.loc[tr_ab], a_train,
        eval_set=[(x_val_xgb.loc[val_ab], a_val)], verbose=100,
    )
    a_val_probability = ab_model.predict_proba(x_val_xgb)[:, 1]
    a_test_probability = ab_model.predict_proba(x_test_xgb)[:, 1]
    del ab_model
    gc.collect()

    yield_val = combine_yield_hurdle(c_val_probability, a_val_probability)
    yield_test = combine_yield_hurdle(c_test_probability, a_test_probability)
    yield_oof[val_idx] = yield_val
    yield_test_average += yield_test / N_SPLITS
    yield_f1 = f1_score(y_yield[val_idx][val_graded], yield_val[val_graded].argmax(1), average="macro")
    c_recall = float(np.mean(yield_val[val_graded][c_val == 1].argmax(1) == c_index))
    print(f"  Yield Macro F1: {yield_f1:.4f}; C recall: {c_recall:.4f}")

    cat_hierarchical_val = combine_hierarchical_probabilities(
        quality_val, yield_val, q_encoder, y_encoder, final_encoder
    )
    ordinal_quality_val = blend_quality_probabilities(
        quality_val, ordinal_val, 1.0, q_encoder, graded_quality_order
    )
    ordinal_hierarchical_val = combine_hierarchical_probabilities(
        ordinal_quality_val, yield_val, q_encoder, y_encoder, final_encoder
    )
    cat_hierarchical_f1 = f1_score(
        y_final[val_idx], cat_hierarchical_val.argmax(1), average="macro"
    )
    ordinal_hierarchical_f1 = f1_score(
        y_final[val_idx], ordinal_hierarchical_val.argmax(1), average="macro"
    )

    diagnostic_hierarchical_f1 = -1.0
    diagnostic_ordinal_weight = 0.0
    for ordinal_weight in np.linspace(0.0, 1.0, 11):
        blended_quality = blend_quality_probabilities(
            quality_val, ordinal_val, ordinal_weight, q_encoder, graded_quality_order
        )
        candidate_hierarchy = combine_hierarchical_probabilities(
            blended_quality, yield_val, q_encoder, y_encoder, final_encoder
        )
        candidate_score = f1_score(
            y_final[val_idx], candidate_hierarchy.argmax(1), average="macro"
        )
        if candidate_score > diagnostic_hierarchical_f1:
            diagnostic_hierarchical_f1 = candidate_score
            diagnostic_ordinal_weight = ordinal_weight
    diagnostic_quality = blend_quality_probabilities(
        quality_val, ordinal_val, diagnostic_ordinal_weight, q_encoder, graded_quality_order
    )
    diagnostic_hierarchy = combine_hierarchical_probabilities(
        diagnostic_quality, yield_val, q_encoder, y_encoder, final_encoder
    )
    # Store the CatBoost-only path temporarily. It is replaced below by an
    # honestly cross-fitted CatBoost/ordinal quality blend.
    hierarchical_oof[val_idx] = cat_hierarchical_val
    disagreement = float(np.mean(direct_val.argmax(1) != diagnostic_hierarchy.argmax(1)))
    best_fold_blend = max(
        (
            f1_score(
                y_final[val_idx],
                (weight * direct_val + (1.0 - weight) * diagnostic_hierarchy).argmax(1),
                average="macro",
            ),
            weight,
        )
        for weight in np.linspace(0.0, 1.0, 21)
    )
    component_rows.append({
        "fold": fold + 1,
        "direct_f1": direct_f1,
        "cat_quality_f1": quality_f1,
        "ordinal_quality_f1": ordinal_quality_f1,
        "yield_f1": yield_f1,
        "c_recall": c_recall,
        "cat_hierarchical_f1": cat_hierarchical_f1,
        "ordinal_hierarchical_f1": ordinal_hierarchical_f1,
        "diagnostic_hierarchical_f1": diagnostic_hierarchical_f1,
        "diagnostic_ordinal_weight": diagnostic_ordinal_weight,
        "disagreement": disagreement,
        "diagnostic_best_blend_f1": best_fold_blend[0],
        "diagnostic_direct_weight": best_fold_blend[1],
    })
    print(
        f"  Hierarchy: cat={cat_hierarchical_f1:.4f}, ordinal={ordinal_hierarchical_f1:.4f}, "
        f"diagnostic-best={diagnostic_hierarchical_f1:.4f} (ordinal={diagnostic_ordinal_weight:.1f})"
    )
    print(
        f"  Disagreement={disagreement:.3f}; diagnostic direct/hierarchy blend={best_fold_blend[0]:.4f}"
    )
    if SMOKE_STAGE == "components":
        pd.DataFrame(component_rows).to_csv(f"{OUT_DIR}/smoke_component_metrics.csv", index=False)
        print(f"[{VERSION}] Component smoke stage complete.")
        raise SystemExit(0)

    del tr_aug, val_aug, test_aug, x_tr, x_val, x_test
    del x_tr_xgb, x_val_xgb, x_test_xgb, direct_val, quality_val, quality_test
    del ordinal_val, ordinal_test, ordinal_val_cumulative, ordinal_test_cumulative
    del yield_val, yield_test, cat_hierarchical_val, ordinal_hierarchical_val
    del ordinal_quality_val, diagnostic_quality, diagnostic_hierarchy
    gc.collect()

# Cross-fit the CatBoost/ordinal quality blend before final calibration.
quality_blend_rows = []
hierarchical_oof.fill(0)
hierarchical_test.fill(0)
for heldout_fold in range(N_SPLITS):
    fit_mask = fold_ids != heldout_fold
    heldout_mask = fold_ids == heldout_fold
    candidates = []
    for ordinal_weight in np.linspace(0.0, 1.0, 11):
        fit_quality = blend_quality_probabilities(
            quality_oof[fit_mask], ordinal_quality_oof[fit_mask], ordinal_weight,
            q_encoder, graded_quality_order,
        )
        fit_hierarchy = combine_hierarchical_probabilities(
            fit_quality, yield_oof[fit_mask], q_encoder, y_encoder, final_encoder
        )
        candidates.append(
            (f1_score(y_final[fit_mask], fit_hierarchy.argmax(1), average="macro"), ordinal_weight)
        )
    fit_score, best_ordinal_weight = max(candidates, key=lambda row: row[0])
    heldout_quality = blend_quality_probabilities(
        quality_oof[heldout_mask], ordinal_quality_oof[heldout_mask], best_ordinal_weight,
        q_encoder, graded_quality_order,
    )
    hierarchical_oof[heldout_mask] = combine_hierarchical_probabilities(
        heldout_quality, yield_oof[heldout_mask], q_encoder, y_encoder, final_encoder
    )
    test_quality = blend_quality_probabilities(
        quality_test_average, ordinal_quality_test, best_ordinal_weight,
        q_encoder, graded_quality_order,
    )
    hierarchical_test += combine_hierarchical_probabilities(
        test_quality, yield_test_average, q_encoder, y_encoder, final_encoder
    ) / N_SPLITS
    heldout_score = f1_score(
        y_final[heldout_mask], hierarchical_oof[heldout_mask].argmax(1), average="macro"
    )
    quality_blend_rows.append({
        "heldout_fold": heldout_fold + 1,
        "ordinal_weight": best_ordinal_weight,
        "fit_f1": fit_score,
        "heldout_f1": heldout_score,
    })

# Cross-fit every calibration and final blend choice. A zero hierarchical weight is
# available, so an unhelpful component is never forced into the fitted blend.
calibrated_oof = np.zeros_like(direct_oof)
calibrated_test = np.zeros_like(direct_test)
calibration_rows = []
scale_rows = []
for heldout_fold in range(N_SPLITS):
    fit_mask = fold_ids != heldout_fold
    heldout_mask = fold_ids == heldout_fold
    calibrator = fit_calibrator(y_final[fit_mask], direct_oof[fit_mask], hierarchical_oof[fit_mask])
    calibrated_oof[heldout_mask] = apply_calibrator(
        direct_oof[heldout_mask], hierarchical_oof[heldout_mask], calibrator
    )
    calibrated_test += apply_calibrator(direct_test, hierarchical_test, calibrator) / N_SPLITS
    heldout_score = f1_score(y_final[heldout_mask], calibrated_oof[heldout_mask].argmax(1), average="macro")
    calibration_rows.append({
        "heldout_fold": heldout_fold + 1,
        "mode": calibrator["mode"],
        "direct_weight": calibrator["direct_weight"],
        "fit_f1": calibrator["fit_score"],
        "heldout_f1": heldout_score,
    })
    for class_index, class_name in enumerate(final_encoder.classes_):
        scale_rows.append({
            "heldout_fold": heldout_fold + 1,
            "class": class_name,
            "direct_scale": calibrator["direct_scales"][class_index],
            "hierarchical_scale": calibrator["hierarchical_scales"][class_index],
        })

raw_direct_f1 = f1_score(y_final, direct_oof.argmax(1), average="macro")
raw_hierarchical_f1 = f1_score(y_final, hierarchical_oof.argmax(1), average="macro")
honest_f1 = f1_score(y_final, calibrated_oof.argmax(1), average="macro")
print(f"\n[{VERSION}] Raw direct OOF: {raw_direct_f1:.4f}")
print(f"[{VERSION}] Raw hierarchical OOF: {raw_hierarchical_f1:.4f}")
print(f"[{VERSION}] Cross-fitted calibrated OOF: {honest_f1:.4f}")

pd.DataFrame(component_rows).to_csv(f"{OUT_DIR}/component_metrics.csv", index=False)
pd.DataFrame(quality_blend_rows).to_csv(f"{OUT_DIR}/quality_blend_folds.csv", index=False)
pd.DataFrame(calibration_rows).to_csv(f"{OUT_DIR}/calibration_folds.csv", index=False)
pd.DataFrame(scale_rows).to_csv(f"{OUT_DIR}/class_scales.csv", index=False)
save_class_metrics(y_final, calibrated_oof, final_encoder.classes_, f"{OUT_DIR}/oof_class_metrics.csv")
save_class_metrics(y_yield[is_graded], yield_oof[is_graded], y_encoder.classes_, f"{OUT_DIR}/yield_oof_class_metrics.csv")
pd.DataFrame({"feature": lgb_features, "importance": importance}).sort_values("importance", ascending=False).to_csv(
    f"{OUT_DIR}/lgb_feature_importance.csv", index=False
)
if SAVE_OOF:
    np.savez_compressed(
        f"{OUT_DIR}/oof_predictions.npz",
        y_true=y_final.astype("int8"),
        fold_id=fold_ids,
        direct=direct_oof.astype("float16"),
        hierarchical=hierarchical_oof.astype("float16"),
        quality=quality_oof.astype("float16"),
        ordinal_quality=ordinal_quality_oof.astype("float16"),
        yield_probability=yield_oof.astype("float16"),
        calibrated=calibrated_oof.astype("float16"),
    )

test_raw["LAST_GRADE"] = final_encoder.inverse_transform(calibrated_test.argmax(1))
output_path = f"{OUT_DIR}/260418.csv"
test_raw.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"[{VERSION}] Saved {output_path}; total time {elapsed(started)}")
