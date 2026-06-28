# Pipeline Development Notes

This document explains the final pipeline, `pipeline_v13_model.py`, for readers who want to understand how the model works and why v13 has its current structure.

The document is written in English, but Korean dataset terms are kept in parentheses where they map directly to labels, columns, or dataset terminology.

## 1. Domain Vocabulary

| English term | Korean dataset term | Meaning |
|:---|:---|:---|
| Hanwoo | 한우 | Korean native cattle |
| carcass grade | 도체등급 | slaughter grading outcome |
| final grade | 최종등급 | `LAST_GRADE`, the 16-class prediction target |
| quality grade | 육질등급 | meat quality grade: `1++`, `1+`, `1`, `2`, `3`, `등외` |
| yield grade | 육량등급 | carcass yield grade: `A`, `B`, `C` |
| out-of-grade | 등외 | outside the normal quality-grade classes |
| carcass traits | 도체형질 | measured slaughter traits such as backfat, ribeye area, intramuscular fat, color, texture, and maturity |
| lineage | 혈통 | parent/grandparent and KPN identity information |
| weather exposure | 기상 노출량 | accumulated weather conditions before slaughter |

The final prediction target is `LAST_GRADE`, a 16-class label:

```text
1++A, 1++B, 1++C,
1+A,  1+B,  1+C,
1A,   1B,   1C,
2A,   2B,   2C,
3A,   3B,   3C,
등외
```

## 2. Final v13 Summary

v13 combines a direct final-grade model with a hierarchical quality/yield model.

| Component | Role |
|:---|:---|
| `HanwooDataProcessorV12` | builds the test-safe feature table from train/test, weather, farm, lineage, and KPN data |
| Nested Proxy Reconstruction | predicts test-missing carcass traits (`도체형질`) as out-of-fold proxy features |
| LightGBM direct path | predicts the 16 final classes directly |
| CatBoost + Ordinal LGB quality path | predicts quality grade (`육질등급`) |
| Dual-C XGBoost yield path | predicts yield grade (`육량등급`) with special handling for C |
| Final blending/calibration | combines direct and hierarchical probabilities and adjusts rare-class probabilities |

Performance summary:

| Path | OOF Macro F1 | External validation |
|:---|---:|---:|
| Direct LightGBM | 0.2126 | - |
| Hierarchical path | 0.2384 | - |
| Final ensemble | 0.2475-0.2477 | 0.244 |

The small gap between OOF and external validation suggests that the farm-group validation design matches the test condition reasonably well.

## 3. Constraints That Shaped the Pipeline

### 3.1 Train/Test Farms Do Not Overlap

The train and test sets have effectively zero overlap in `FARM_UNIQUE_NO`. A random split can overestimate performance because cattle from similar farm conditions may appear in both train and validation folds.

v13 therefore uses farm-aware validation:

- cattle from the same farm stay in the same fold
- class ratios are preserved as much as possible
- direct target encoding by farm ID is avoided

### 3.2 Test Rows Do Not Have Measured Carcass Traits

The training set contains measured carcass traits (`도체형질`), but the test set does not.

| Column | Meaning |
|:---|:---|
| `BACKFAT` | backfat thickness (`등지방두께`) |
| `REA` | ribeye area / longissimus muscle area (`배최장근단면적`) |
| `INSFAT` | intramuscular fat / marbling score (`근내지방도`) |
| `YUKSAK` | meat color (`육색`) |
| `FATSAK` | fat color (`지방색`) |
| `TISSUE` | texture (`조직감`) |
| `GROWTH` | maturity (`성숙도`) |

These variables are strongly related to grade, but using the measured values directly would create a train/test mismatch. v13 uses proxy predictions instead.

### 3.3 Class Imbalance Matters

The label distribution is imbalanced.

- `3C` and `등외` are rare.
- Yield grade (`육량등급`) C is less frequent than A and B.
- Macro F1 gives rare classes the same weight as frequent classes.

This is why v13 has class-wise calibration and a dedicated Dual-C yield model.

### 3.4 Weather Features Must Be Time-Safe

Weather features must use only information before slaughter. v13 uses weather exposure summaries such as:

- previous-month heat days
- 3/6/12-month rolling THI
- 3/6/12-month heat-day counts
- short-term vs long-term temperature and THI trends

Weather is treated as cumulative exposure, not as a single-day value.

## 4. Development History

The earlier versions are best understood as a sequence of modeling constraints being discovered and addressed.

| Stage | Focus | What worked | What failed or changed |
|:---|:---|:---|:---|
| v1-v3 | baseline LightGBM and simple validation | direct 16-class modeling was easy to build | random/stratified validation was too optimistic because farms were not separated |
| v4-v6 | farm-aware validation and feature cleanup | `GroupKFold`, rolling weather, farm context, and lineage joins made validation more realistic | scores dropped, showing that earlier OOF scores were inflated |
| v7-v9 | CatBoost and hierarchical prediction | CatBoost helped with categorical features; direct and hierarchical paths made different errors | direct 16-class modeling alone was weak for rare classes |
| v10 | fold-safe KPN/station priors | leakage risk was reduced | low-support priors were noisy and not reliably helpful |
| v11-v12 | yield grade C recovery | C-specific modeling improved C recall | aggressive C detection reduced B recall |
| v13 | final architecture | proxy carcass traits + direct/hierarchical blending + Dual-C yield path | retained as final model |

The main lessons were:

- validation must reflect unseen farms
- test-missing variables need proxy reconstruction
- final grade has useful quality/yield structure
- rare classes need explicit handling
- generic stacking was less useful than structured blending

## 5. v13 Architecture

### 5.1 Data Processor

`HanwooDataProcessorV12` builds the feature table used by the final model.

Responsibilities:

- train/test preprocessing
- lineage (`혈통`) join by `CATTLE_NO`
- KPN join by `KPN_NO`
- weather join by station, year, and month
- farm area, herd count, density, and death-history joins by `FARM_UNIQUE_NO`
- slaughter month, slaughter season, and birth season features
- categorical cleanup for sex, region, station, and lineage IDs

Extended lineage features:

```text
FATHER_CATTLE_NO
MOTHER_ANIMAL_NO
F_GMOTHER_ANIMAL_NO
F_GFATHER_CATTLE_NO
M_GMOTHER_ANIMAL_NO
M_GFATHER_CATTLE_NO
```

These are mainly used by CatBoost because it can handle high-cardinality categorical IDs directly.

### 5.2 Target Decomposition

v13 predicts the target in two ways: directly as 16 final classes, and hierarchically as quality grade plus yield grade.

| Target | Meaning | Classes | Model |
|:---|:---|:---|:---|
| `y_final` | final carcass grade (`최종 도체등급`) | 16 classes | LightGBM direct path |
| `y_quality` | quality grade (`육질등급`) | `1++`, `1+`, `1`, `2`, `3`, `등외` | CatBoost + Ordinal LGB |
| `y_yield` | yield grade (`육량등급`) | `A`, `B`, `C` | Dual-C XGBoost |

This reflects how `LAST_GRADE` is structured: most final classes are a combination of quality grade and yield grade.

### 5.3 Feature Sets

| Feature set | Contents | Model | Reason |
|:---|:---|:---|:---|
| `lgb_features` | base numeric/categorical features plus fold-safe group counts | LightGBM, Ordinal LGB | stable for direct prediction and ordered boundaries |
| `cat_quality_features` | base features, extended lineage IDs, proxy carcass traits, group counts | CatBoost quality model | strong for high-cardinality categorical interactions |
| `xgb_yield_features` | numeric-heavy features, proxy carcass traits, group counts | XGBoost yield model | yield grade is strongly tied to numeric carcass signals |

`attach_fold_features()` creates group counts and cohort-normalized physical features inside each fold. Validation fold information is not used to compute training fold statistics.

### 5.4 Nested Proxy Reconstruction

Nested Proxy Reconstruction is the key v13 improvement.

Problem:

- measured carcass traits are available in train
- measured carcass traits are missing in test
- these traits are highly predictive of grade

Solution:

Inside each outer farm fold, v13 creates inner folds. A `CatBoostRegressor` is trained on inner-train rows and predicts carcass traits for inner-validation rows. This gives training rows proxy values from models that did not see those rows.

```text
Outer train
  |- Inner fold 1 train -> proxy model -> Inner fold 1 validation proxy
  |- Inner fold 2 train -> proxy model -> Inner fold 2 validation proxy

Outer validation/test
  |- inner proxy model prediction -> proxy features
```

Example proxy features:

```text
proxy_backfat
proxy_rea
proxy_insfat
proxy_yuksak
proxy_fatsak
proxy_tissue
proxy_growth
proxy_windex
proxy_yield_margin_a
proxy_yield_margin_c
proxy_quality_worst_rank
```

Proxy reconstruction metrics:

| Trait | Meaning | MAE | R2 | Interpretation |
|:---|:---|---:|---:|:---|
| `BACKFAT` | backfat thickness (`등지방두께`) | 3.61 | 0.233 | partially recoverable |
| `REA` | ribeye area (`배최장근단면적`) | 7.94 | 0.545 | relatively well reconstructed |
| `INSFAT` | intramuscular fat (`근내지방도`) | 1.45 | 0.336 | partially recovers a key quality signal |
| `YUKSAK` | meat color (`육색`) | 0.302 | 0.135 | weakly explained |
| `FATSAK` | fat color (`지방색`) | 0.148 | 0.105 | weakly explained |
| `TISSUE` | texture (`조직감`) | 0.726 | 0.318 | partially recoverable |
| `GROWTH` | maturity (`성숙도`) | 0.460 | 0.909 | very well reconstructed |

Color-related traits have low R2, likely because weather, lineage, and body-weight features do not fully explain meat color and fat color.

### 5.5 Direct LightGBM Path

The direct path predicts the 16 final classes in one model.

Role:

- learns the final label distribution directly
- captures quality/yield combination patterns without explicit decomposition
- provides a stable probability base for calibration

Limitations:

- weak rare-class recall
- does not explicitly model the quality/yield structure
- less natural than CatBoost for high-cardinality categorical features

### 5.6 CatBoost / Ordinal LGB Quality Path

The quality path predicts quality grade (`육질등급`).

CatBoost:

- handles sex, region, station, and lineage IDs as categorical features
- uses extended lineage and proxy carcass traits
- benefits from `INSFAT`, `TISSUE`, and `GROWTH` proxies

Ordinal LGB:

- uses the ordered structure of quality grades
- learns binary boundaries across ordered grades
- gives different errors from CatBoost, making blending useful

The two models are blended with fold-specific weights.

### 5.7 Dual-C XGBoost Yield Path

The yield path predicts yield grade (`육량등급`) A/B/C.

A single A/B/C model often predicts C as B. v13 decomposes yield prediction:

```text
Step 1: P(C) = C vs non-C
Step 2: P(A | non-C) = A vs B

P(A) = (1 - P(C)) * P(A | non-C)
P(B) = (1 - P(C)) * (1 - P(A | non-C))
P(C) = P(C)
```

Two C models are trained:

| Model | Setting | Purpose |
|:---|:---|:---|
| balanced-C | `balance_power = 0.25` | detects C conservatively and preserves B recall |
| aggressive-C | `balance_power = 0.50` | detects C aggressively and improves C recall |

Observed trade-off:

- balanced-C: C recall 0.2952, B recall 0.6364
- aggressive-C: C recall 0.5109, B recall 0.4836

v13 searches fold-specific blending weights between these two C models.

### 5.8 Hierarchical Probability and Final Ensemble

The hierarchical path combines quality and yield probabilities:

```text
P(final grade = quality + yield)
  = P(quality) * P(yield)
```

Examples:

```text
P(1++A) = P(quality = 1++) * P(yield = A)
P(2C)   = P(quality = 2)   * P(yield = C)
```

`등외` has no yield-grade suffix and is handled as a separate final class.

Final prediction:

1. Generate direct probabilities.
2. Generate hierarchical probabilities.
3. Apply class-wise scaling for rare-class calibration.
4. Search direct/hierarchical blending weights.
5. Average test probabilities across folds.
6. Select the highest-probability class as `LAST_GRADE`.

## 6. Output Files

Default output directory:

```text
outputs/v13/
```

| File | Description |
|:---|:---|
| `260418.csv` | prediction output |
| `component_metrics.csv` | fold-level direct/quality/yield/hierarchy metrics |
| `proxy_metrics.csv` | carcass-trait proxy reconstruction metrics |
| `quality_blend_folds.csv` | CatBoost/Ordinal quality blending results |
| `calibration_folds.csv` | final ensemble calibration results |
| `class_scales.csv` | class-wise scaling coefficients |
| `oof_class_metrics.csv` | final class-level OOF precision/recall/F1 |
| `yield_oof_class_metrics.csv` | yield-grade class-level OOF metrics |
| `lgb_feature_importance.csv` | LightGBM feature importance |

## 7. Model and Component Comparison

| Model/component | Role | Strengths | Weaknesses / risks | Final v13 decision |
|:---|:---|:---|:---|:---|
| LightGBM direct 16-class | predicts final grade directly | fast, stable, learns final label distribution | weak rare-class recall; limited categorical interaction handling | kept as the base axis of the final ensemble |
| CatBoost quality | predicts quality grade | strong with high-cardinality categorical features, lineage IDs, region/station interactions | higher training cost; GPU/environment dependency | kept as the core quality model |
| Ordinal LightGBM quality | learns ordered quality-grade boundaries | reflects ordinal grade structure | weaker categorical handling than CatBoost | kept as a complementary quality blender |
| XGBoost yield | predicts yield grade | strong numeric boundary learning | vulnerable to C-grade imbalance | expanded into Dual-C and kept |
| Dual-C XGBoost | separates C vs non-C and A vs B | improves C recall and controls B/C trade-off | blending can reduce B recall if too aggressive | core yield-path structure |
| Nested Proxy Reconstruction | reconstructs test-missing carcass traits | adds carcass-trait information without leakage | weak proxy traits can add noise | most important v13 improvement |
| Fold-safe group count | adds KPN/station/region/lineage frequency | target-free, low leakage risk, good coverage | simple frequency information only | kept as stable auxiliary features |
| KPN/station target prior | provides historical grade distribution | interpretable domain signal | noisy with low support and uneven coverage | used cautiously; not a core dependency |
| Class-wise scaling | adjusts rare-class probabilities | directly helps Macro F1 | can reduce frequent-class precision if overdone | applied only through CV-based calibration |
| General stacking | combines model probabilities through a meta-model | flexible in theory | overfits when model errors are highly correlated | not favored; structured blending is preferred |
