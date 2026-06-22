"""Nested, farm-safe reconstruction of carcass traits unavailable in test data."""

import gc

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold


TRAIT_COLUMNS = ["BACKFAT", "REA", "INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH"]
PROXY_FEATURE_NAMES = [
    *(f"proxy_{column.lower()}" for column in TRAIT_COLUMNS),
    "proxy_windex",
    "proxy_yield_margin_a",
    "proxy_yield_margin_c",
    "proxy_yield_grade_signal",
    "proxy_insfat_rank",
    "proxy_yuksak_rank",
    "proxy_fatsak_rank",
    "proxy_tissue_rank",
    "proxy_quality_worst_rank",
    "proxy_growth_high",
]


def _quality_ranks(predictions):
    insfat, yuksak, fatsak, tissue = [predictions[:, index] for index in [2, 3, 4, 5]]

    insfat_rank = np.select(
        [insfat >= 7.0, insfat >= 5.0, insfat >= 4.0, insfat >= 2.0],
        [0, 1, 2, 3],
        default=4,
    )
    rounded_yuksak = np.clip(np.rint(yuksak), 1, 7)
    yuksak_rank = np.select(
        [
            (rounded_yuksak >= 3) & (rounded_yuksak <= 5),
            np.isin(rounded_yuksak, [2, 6]),
            rounded_yuksak == 1,
            rounded_yuksak == 7,
        ],
        [0, 1, 2, 3],
        default=4,
    )
    rounded_fatsak = np.clip(np.rint(fatsak), 1, 7)
    fatsak_rank = np.select(
        [rounded_fatsak <= 4, rounded_fatsak == 5, rounded_fatsak == 6, rounded_fatsak == 7],
        [0, 1, 2, 3],
        default=4,
    )
    tissue_rank = np.clip(np.rint(tissue), 1, 5).astype("int8") - 1
    return [array.astype("float32") for array in [insfat_rank, yuksak_rank, fatsak_rank, tissue_rank]]


def derive_proxy_features(predictions, base_frame):
    """Create physical and rule-distance features from reconstructed traits."""
    predictions = np.asarray(predictions, dtype="float32").copy()
    clips = [(0, 60), (20, 200), (1, 9), (1, 7), (1, 7), (1, 5), (1, 9)]
    for index, (lower, upper) in enumerate(clips):
        predictions[:, index] = np.clip(predictions[:, index], lower, upper)

    result = pd.DataFrame(index=base_frame.index)
    for index, column in enumerate(TRAIT_COLUMNS):
        result[f"proxy_{column.lower()}"] = predictions[:, index]

    weight = pd.to_numeric(base_frame["WEIGHT"], errors="coerce").fillna(0).to_numpy(dtype="float32")
    sex = pd.to_numeric(base_frame["xgb_sex_code"], errors="coerce").fillna(2).to_numpy(dtype="int8")
    backfat, rea = predictions[:, 0], predictions[:, 1]
    intercept = np.select([sex == 0, sex == 1], [6.90137, 0.20108], default=11.06338)
    backfat_coef = np.select([sex == 0, sex == 1], [-0.9446, -2.18625], default=-1.25149)
    rea_coef = np.select([sex == 0, sex == 1], [0.31806, 0.29275], default=0.28238)
    weight_coef = np.select([sex == 0, sex == 1], [0.54952, 0.64099], default=0.56781)
    windex = (
        (intercept + backfat_coef * backfat + rea_coef * rea + weight_coef * weight)
        / np.maximum(weight, 1.0)
        * 100.0
    ).astype("float32")
    a_threshold = np.select([sex == 0, sex == 1], [61.83, 68.45], default=62.52)
    c_threshold = np.select([sex == 0, sex == 1], [59.70, 66.32], default=60.40)
    result["proxy_windex"] = windex
    result["proxy_yield_margin_a"] = (windex - a_threshold).astype("float32")
    result["proxy_yield_margin_c"] = (windex - c_threshold).astype("float32")
    result["proxy_yield_grade_signal"] = np.select(
        [windex >= a_threshold, windex >= c_threshold], [0, 1], default=2
    ).astype("float32")

    quality_ranks = _quality_ranks(predictions)
    for name, values in zip(
        ["proxy_insfat_rank", "proxy_yuksak_rank", "proxy_fatsak_rank", "proxy_tissue_rank"],
        quality_ranks,
    ):
        result[name] = values
    result["proxy_quality_worst_rank"] = np.maximum.reduce(quality_ranks).astype("float32")
    result["proxy_growth_high"] = (predictions[:, 6] >= 8.0).astype("float32")
    return result


def build_nested_proxy_features(
    x_train,
    x_validation,
    x_test,
    target_train,
    target_validation,
    farm_groups,
    categorical_features,
    task_type="GPU",
    inner_splits=2,
    random_seed=42,
    iterations=900,
):
    """Return inner-OOF proxies with one-model predictions for every split."""
    target_train = target_train[TRAIT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    target_validation = target_validation[TRAIT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_train = np.isfinite(target_train).all(axis=1) & (target_train > -90).all(axis=1)
    valid_validation = np.isfinite(target_validation).all(axis=1) & (target_validation > -90).all(axis=1)
    if valid_train.sum() < 10_000:
        raise ValueError(f"Too few valid rows for carcass proxy training: {valid_train.sum():,}")

    train_predictions = np.zeros((len(x_train), len(TRAIT_COLUMNS)), dtype="float32")
    validation_predictions = np.zeros((len(x_validation), len(TRAIT_COLUMNS)), dtype="float32")
    test_predictions = np.zeros((len(x_test), len(TRAIT_COLUMNS)), dtype="float32")
    validation_assignment = np.arange(len(x_validation)) % inner_splits
    test_assignment = np.arange(len(x_test)) % inner_splits
    missing_proxy_rows = ~valid_train.to_numpy()
    missing_positions = np.flatnonzero(missing_proxy_rows)
    missing_assignment = np.arange(len(missing_positions)) % inner_splits
    splitter = GroupKFold(n_splits=inner_splits)
    split_rows = np.flatnonzero(valid_train.to_numpy())
    split_groups = np.asarray(farm_groups)[split_rows]

    for inner_fold, (fit_local, hold_local) in enumerate(
        splitter.split(split_rows, groups=split_groups), start=1
    ):
        fit_rows = split_rows[fit_local]
        hold_rows = split_rows[hold_local]
        fit_target = target_train.iloc[fit_rows].to_numpy(dtype="float32")
        means = fit_target.mean(axis=0)
        scales = np.maximum(fit_target.std(axis=0), 1e-3)
        standardized_fit = (fit_target - means) / scales
        standardized_hold = (
            target_train.iloc[hold_rows].to_numpy(dtype="float32") - means
        ) / scales

        model = CatBoostRegressor(
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
            iterations=iterations,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=8.0,
            random_strength=0.5,
            random_seed=random_seed + inner_fold,
            task_type=task_type,
            allow_writing_files=False,
            verbose=100,
        )
        model.fit(
            x_train.iloc[fit_rows],
            standardized_fit,
            cat_features=categorical_features,
            eval_set=(x_train.iloc[hold_rows], standardized_hold),
            early_stopping_rounds=100,
            use_best_model=True,
        )
        train_predictions[hold_rows] = model.predict(x_train.iloc[hold_rows]) * scales + means
        validation_rows = np.flatnonzero(validation_assignment == inner_fold - 1)
        test_rows = np.flatnonzero(test_assignment == inner_fold - 1)
        validation_predictions[validation_rows] = (
            model.predict(x_validation.iloc[validation_rows]) * scales + means
        )
        test_predictions[test_rows] = model.predict(x_test.iloc[test_rows]) * scales + means
        assigned_missing = missing_positions[missing_assignment == inner_fold - 1]
        if len(assigned_missing):
            train_predictions[assigned_missing] = (
                model.predict(x_train.iloc[assigned_missing]) * scales + means
            )
        del model
        gc.collect()

    metrics = []
    validation_actual = target_validation.loc[valid_validation].to_numpy(dtype="float32")
    validation_predicted = validation_predictions[valid_validation.to_numpy()]
    for index, trait in enumerate(TRAIT_COLUMNS):
        metrics.append({
            "trait": trait,
            "mae": mean_absolute_error(validation_actual[:, index], validation_predicted[:, index]),
            "r2": r2_score(validation_actual[:, index], validation_predicted[:, index]),
            "support": int(valid_validation.sum()),
        })

    return (
        derive_proxy_features(train_predictions, x_train),
        derive_proxy_features(validation_predictions, x_validation),
        derive_proxy_features(test_predictions, x_test),
        metrics,
    )
