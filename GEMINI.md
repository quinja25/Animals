# Project Standards & Conventions

## Pipeline & Submission Standards

To ensure consistency and compatibility with the evaluation system, all pipelines must adhere to the following standards for producing submission files.

### Submission File Requirements
- **Filename**: `260418.csv` (This corresponds to the registration number).
- **Encoding**: UTF-8 with BOM (`utf-8-sig`).
- **Required Column**: `LAST_GRADE`
- **Allowed Values**: `1++A`, `1++B`, `1++C`, `1+A`, `1+B`, `1+C`, `1A`, `1B`, `1C`, `2A`, `2B`, `2C`, `3A`, `3B`, `3C`, `등외`.
- **Target Encoding**:
  - `0: 1++A`, `1: 1++B`, `2: 1++C`
  - `3: 1+A`, `4: 1+B`, `5: 1+C`
  - `6: 1A`, `7: 1B`, `8: 1C`
  - `9: 2A`, `10: 2B`, `11: 2C`
  - `12: 3A`, `13: 3B`, `14: 3C`
  - `15: 등외`

### Pipeline Workflow
1. **Feature Engineering**: Standardized features include physical/physical (`WEIGHT`, `AGE`, `sex_code`), temporal (`abatt_year/month`), genetic (KPN stats), environmental (`stn`, weather stats), and farm management (`density`, `farm_size`).
2. **Validation Strategy**: Use `GroupKFold` based on `FARM_UNIQUE_NO` to ensure generalizability to unseen farms (0% farm overlap between train and test).
3. **Metric**: Primary optimization metric is **Macro F1-Score**.
4. **Output**: Save predictions to `submissions/{version}/260418.csv` (e.g., `submissions/v5/260418.csv`). The filename must be exactly `260418.csv` and the encoding must be `utf-8-sig`. Any auxiliary files like feature importance should also be stored in the same versioned directory.

## Data Mapping Schema

To avoid manual joining overhead, follow this standardized mapping logic:

| Source Dataset | Join Key(s) | Target Dataset | Logic / Pre-processing |
| :--- | :--- | :--- | :--- |
| `hanwoo_train/test` | `CATTLE_NO` | `hanwoo_lineage` | Use `str.strip()` on keys. 1:1 mapping after `drop_duplicates('CATTLE_NO')`. |
| `hanwoo_lineage` | `KPN_NO` | `KPN 유전능력 자료` | Extract breeding values (sbv) for weight, rea, backfat, and insfat. |
| `hanwoo_train/test` | `FARM_UNIQUE_NO`| `hanwoo_area` | Join to get `AREA` and yearly counts. |
| `hanwoo_area` | `FARM_UNIQUE_NO`| `hanwoo_death` | Aggregate `death` counts per farm before joining with `area`. |
| `hanwoo_train/test` | `stn`, `year`, `month` | `hanwoo_weather` | Aggregate weather to monthly means (`ta`, `rn`, `rhm`) before joining. |

### Join Requirements
- **Key Normalization**: Always apply `.astype(str).str.strip()` to `CATTLE_NO`, `KPN_NO`, and `FARM_UNIQUE_NO` before merging.
- **Deduplication**: `hanwoo_lineage` and `hanwoo_area` MUST be deduplicated by their primary keys (`CATTLE_NO` and `FARM_UNIQUE_NO` respectively) to prevent row explosion during joins.
- **Handling Unseen Farms**: Since there is 0% farm overlap between train and test, NEVER join on `FARM_UNIQUE_NO` for target encoding. Instead, join on `KPN_NO` or `stn` for historical performance stats.

## Technical Stack
- **Models**: LightGBM (Baseline), CatBoost, XGBoost.
- **Ensemble**: Weighted Average or Stacking based on OOF predictions.
- **Environment**: Python with standard data science stack (pandas, numpy, scikit-learn, lightgbm).
