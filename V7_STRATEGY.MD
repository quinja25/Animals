# Hanwoo Grade Prediction v7 Analysis Report

## Executive Summary

v6 is currently capped at `0.212` Macro F1 on the leaderboard. The main constraint is not just model capacity, but a set of structural data issues:

- `0%` farm overlap between train and test makes farm-specific memorization useless at inference time.
- KPN SBV features are extremely sparse because the Excel file covers mostly older bulls, so the current genetic signal is weak.
- Several engineered features have leakage or time-alignment risk, especially target statistics, weather rollups, and farm-level aggregates.
- The current hierarchical grade approach can produce logically inconsistent combinations because quality and yield are modeled independently.

The practical path to v7 is to reduce leakage, improve feature honesty, and shift the model toward features that survive the test distribution: genetics that actually join, station-level signal, animal physiology, and properly aligned weather history.

## Current Findings

### 1. Validation and leaderboard

- v6 LB score: `0.212` Macro F1.
- Train size: `2.4M` rows.
- Test size: `452K` rows.
- Farm overlap: `0%`.

### 2. Grade distribution

- `LAST_GRADE` includes the `등외` class.
- The `등외` value appears as corrupted bytes in raw form and must be decoded as UTF-8.
- `등외` is rare: `5,540` rows, about `0.23%` of train.

### 3. Lineage and KPN coverage

- Only about `62%` of cattle in train/test have lineage records.
- About `40%` of cattle miss `KPN_NO` after the lineage join.
- Of the rows that do have KPN info, only `7.73%` match the Excel breeding values.
- The Excel file has `995` unique KPNs, while `1,366` appear in lineage.
- Match quality drops sharply for cattle born in `2017` and later.
- KPN grade averages and counts are missing for about `40-41%` of rows because of join failure.
- Historical KPN grade stats from training are still useful because about `98%` of test KPNs appear somewhere in train.

### 4. Station statistics

- Station grade averages are missing for about `7.4%` of test because some stations never appear in train.
- Station remains useful, but it should be treated as a generalization feature, not a farm proxy.

### 5. Feature importance

- The strongest features are `density`, `weight`, `weight per age`, `age`, `station`, and `station grade average`.
- KPN SBV features sit near the bottom because they are sparse and incomplete.

## Key Problems

### 1. KPN SBV features are too sparse

The current Excel-based KPN features are nearly unusable as direct predictors because roughly `95.5%` of the relevant values are missing. The file only covers older bulls, so it does not provide broad coverage for the current population.

### 2. Hierarchical grade modeling is inconsistent

The model predicts quality and yield independently, but those outputs are correlated. That allows invalid combinations such as `1++A` when the quality path should already imply `등외`.

There is also a concern that missing yield components default to `B` during extraction, which may bias the model toward that class.

### 3. Weather features may leak future information

The rolling weather features are built on monthly aggregates, but they are not tightly aligned to each animal’s slaughter date. That risks mixing future weather into historical windows.

### 4. Farm area and density logic is stale

- Farm area is deduplicated by keeping the first record rather than the most recent one.
- Density uses a static 2023 cattle count across all years.

Both choices reduce temporal fidelity and can distort farm condition estimates.

### 5. Station-level averages can leak across folds

Station grade averages are computed from the full training set. Since farms in the same station can land in different folds, this can leak target information into validation.

### 6. Validation environment is currently blocked

The validation script exists, but the current Python environment cannot run it because `pandas` is not installed.

## v7 Direction

### Priority 1: Make target statistics fold-safe

Rebuild `kpn_grade_avg`, `kpn_grade_cnt`, and station-level grade features inside each CV fold only. Use Bayesian smoothing so sparse groups shrink toward the fold mean instead of producing unstable estimates.

### Priority 2: Use only meaningful genetic coverage

Keep the KPN features only where the join is reliable. Expand beyond the four current SBV columns if the source supports it, but do not assume the Excel file will cover the full KPN universe.

### Priority 3: Fix the grade formulation

Move away from the current independent quality/yield extraction if it can produce invalid combinations. At minimum, enforce output consistency in post-processing. A direct 16-class formulation is a cleaner long-term option.

### Priority 4: Rebuild weather features around animal time

Weather should be tied to birth-to-slaughter lifecycle windows, not just slaughter-month station averages. That makes the feature set more biologically meaningful and less likely to leak future information.

### Priority 5: Correct farm area and density features

Use the latest available farm area record, and compute density with a time-appropriate livestock count rather than a fixed 2023 constant.

### Priority 6: Preserve stable predictors

Keep the features that are clearly working:

- `weight`
- `age`
- `weight per age`
- `station`
- `station grade average`
- valid station/weather summaries

## Recommended v7 Build Order

1. Recompute KPN and station target stats with fold awareness.
2. Fix weather time alignment and rebuild lifecycle weather features.
3. Correct farm area deduplication and density calculation.
4. Re-evaluate KPN features after join coverage is measured cleanly.
5. Test a direct 16-class formulation or a constrained hierarchical version.
6. Validate the revised pipeline on a working Python environment.

## Validation Note

The prepared validator at `scratch/validate_v6.py` could not run in the current terminal because the environment is missing `pandas`.

## Bottom Line

The v7 gain will come less from adding more raw features and more from making the existing features honest:

- no leakage in target statistics,
- no stale farm metadata,
- no future weather in historical windows,
- no weak genetic features treated as if they were reliable,
- no inconsistent grade outputs.

That is the clearest path to improving beyond the current `0.212` Macro F1 ceiling.
