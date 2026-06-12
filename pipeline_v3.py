"""
한우 도체 등급 예측 파이프라인 v2
─────────────────────────────────────────────────────
기존 대비 주요 개선 사항:
  1. 사육기간 전체 기상 집계 (출생~도축 전체 월 집계) → 핵심 차별화
  2. 도축전 1·2·3개월 기상 별도 피처화 (직전 환경 영향 반영)
  3. 농장별 Leave-One-Out Target Encoding (등급 수준 proxy)
  4. 샘플링 없이 전체 240만행 학습
  5. num_leaves=127, n_estimators=600 강화 모델
  6. 그램 2026 (Intel Ultra 7, 16GB RAM) 최적화

실행 방법:
  1. 아래 DATA_DIR, OUT_DIR 경로를 본인 환경에 맞게 수정
  2. python pipeline_v2.py
  3. submissions/260418.csv 확인

필요 패키지:
  pip install lightgbm scikit-learn pandas numpy
"""

import pandas as pd
import numpy as np
import gc
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── 경로 설정 (본인 환경에 맞게 수정) ───────────────────────────
DATA_DIR = "data"          # hanwoo_train.csv 등이 있는 폴더
OUT_DIR  = "submissions"   # 결과 저장 폴더
# ─────────────────────────────────────────────────────────────────

Path(OUT_DIR).mkdir(exist_ok=True)
T_START = time.time()

def elapsed():
    return f"{(time.time()-T_START)/60:.1f}분"

print("=" * 60)
print("  한우 도체 등급 예측 파이프라인 v2")
print("=" * 60)

# ─── 등급 관련 상수 ───────────────────────────────────────────────
GRADE_ORDER = [
    "1++A","1++B","1++C",
    "1+A", "1+B", "1+C",
    "1A",  "1B",  "1C",
    "2A",  "2B",  "2C",
    "3A",  "3B",  "3C",
    "등외"
]

# 등급을 0~6 점수로 변환 (target encoding용)
GRADE_SCORE = {
    "1++A":15,"1++B":14,"1++C":13,
    "1+A":12, "1+B":11, "1+C":10,
    "1A":9,   "1B":8,   "1C":7,
    "2A":6,   "2B":5,   "2C":4,
    "3A":3,   "3B":2,   "3C":1,
    "등외":0
}

SIDO_MAP = {s: i for i, s in enumerate([
    "강원특별자치도","경기도","경상남도","경상북도","광주광역시","대구광역시",
    "대전광역시","부산광역시","서울특별시","세종특별자치시","울산광역시",
    "인천광역시","전라남도","전라북도","전북특별자치도","전라북도특별자치도",
    "제주특별자치도","충청남도","충청북도"
])}

# ─── STEP 1: 기상 데이터 사전 집계 ───────────────────────────────
print(f"\n[STEP 1] 기상 데이터 월별 집계 중... ({elapsed()})")

weather = pd.read_csv(f"{DATA_DIR}/hanwoo_weather.csv", parse_dates=["date"])
weather["ta_avg"]  = (weather["ta_max"] + weather["ta_min"]) / 2
weather["THI"]     = ((1.8*weather["ta_avg"]+32)
                      - (0.55 - 0.0055*weather["rhm_avg"])
                      * (1.8*weather["ta_avg"]-26))
weather["year"]    = weather["date"].dt.year
weather["month"]   = weather["date"].dt.month

# 월별 집계 (도축월·출생월·도축전 N개월 join에 사용)
wm = weather.groupby(["stn","year","month"]).agg(
    ta_mean    = ("ta_avg",  "mean"),
    ta_max_mean= ("ta_max",  "mean"),
    ta_min_mean= ("ta_min",  "mean"),
    thi_mean   = ("THI",     "mean"),
    thi_max    = ("THI",     "max"),
    rhm_mean   = ("rhm_avg", "mean"),
    heat2d     = ("THI",     lambda x: (x>=80).sum()),   # 중등도 열스트레스
    heat3d     = ("THI",     lambda x: (x>=85).sum()),   # 심각 열스트레스
    cold_d     = ("ta_min",  lambda x: (x<-5).sum()),    # 한파일수
    rn_sum     = ("rn_day",  "sum"),
    ws_mean    = ("ws_davg", "mean"),
).reset_index()

# float32 변환으로 메모리 절약
float_cols = wm.select_dtypes("float64").columns
wm[float_cols] = wm[float_cols].astype("float32")
del weather; gc.collect()
print(f"  월별 집계 완료: {wm.shape} ({elapsed()})")

WEATHER_START = pd.Timestamp("2020-01-01")   # 기상 데이터 시작일

# ─── STEP 2: 농장 피처 준비 ───────────────────────────────────────
print(f"\n[STEP 2] 농장 피처 준비 중... ({elapsed()})")

area = pd.read_csv(f"{DATA_DIR}/hanwoo_area.csv",
                   dtype={"FARM_UNIQUE_NO": str})
# -99는 결측
for col in ["C2023","C2024","C2025","AREA"]:
    area[col] = area[col].replace(-99, np.nan)
area["avg_cattle"] = area[["C2023","C2024","C2025"]].mean(axis=1)
area["density"]    = area["avg_cattle"] / area["AREA"].replace(0, np.nan)
area = area[["FARM_UNIQUE_NO","avg_cattle","density"]]

death = pd.read_csv(f"{DATA_DIR}/hanwoo_death.csv",
                    usecols=["FARM_UNIQUE_NO"],
                    dtype={"FARM_UNIQUE_NO": str})
death_cnt = (death.groupby("FARM_UNIQUE_NO").size()
                  .reset_index(name="death_cnt"))
del death; gc.collect()
print(f"  농장 피처 준비 완료 ({elapsed()})")

# ─── STEP 3: 훈련 데이터 로드 ──────────────────────────────────────
print(f"\n[STEP 3] 훈련 데이터 로드 중... ({elapsed()})")

LOAD_COLS = ["sido","stn","ABATT_DATE","JUDGE_SEX","WEIGHT","AGE",
             "BIRTH_YMD","FARM_UNIQUE_NO","LAST_GRADE"]

train = pd.read_csv(
    f"{DATA_DIR}/hanwoo_train.csv",
    usecols=LOAD_COLS,
    dtype={"FARM_UNIQUE_NO": str},
    low_memory=False,
)
train["ABATT_DATE"] = pd.to_datetime(train["ABATT_DATE"], errors="coerce")
train["BIRTH_YMD"]  = pd.to_datetime(
    train["BIRTH_YMD"].astype(str), format="%Y%m%d", errors="coerce")
print(f"  로드 완료: {len(train):,}행 ({elapsed()})")

# ─── STEP 4: 농장별 Target Encoding (Leave-One-Out) ───────────────
print(f"\n[STEP 4] 농장별 Target Encoding 계산 중... ({elapsed()})")

train["grade_score"] = train["LAST_GRADE"].map(GRADE_SCORE).fillna(0).astype("float32")

# Leave-One-Out target encoding (데이터 누수 방지)
farm_sum   = train.groupby("FARM_UNIQUE_NO")["grade_score"].transform("sum")
farm_cnt   = train.groupby("FARM_UNIQUE_NO")["grade_score"].transform("count")
train["farm_grade_loo"]  = (farm_sum - train["grade_score"]) / (farm_cnt - 1).clip(lower=1)
train["farm_grade_mean"] = farm_sum / farm_cnt   # 추론시 사용할 일반 평균
train["farm_grade_cnt"]  = farm_cnt

# 농장별 등급 표준편차 (농장 품질 일관성)
farm_std = train.groupby("FARM_UNIQUE_NO")["grade_score"].std().reset_index()
farm_std.columns = ["FARM_UNIQUE_NO","farm_grade_std"]

del farm_sum, farm_cnt; gc.collect()
print(f"  Target Encoding 완료 ({elapsed()})")

# ─── STEP 5: 공통 피처 생성 함수 ─────────────────────────────────
def make_features(df, farm_grade_lookup, wm_monthly, is_train=True):
    """
    df: 훈련 or 테스트 DataFrame
    farm_grade_lookup: FARM_UNIQUE_NO → (mean, std, cnt) 딕셔너리
    wm_monthly: 월별 기상 집계 DataFrame
    """
    df = df.copy()

    # ── 기본 날짜 파생 ──
    df["abatt_year"]   = df["ABATT_DATE"].dt.year.fillna(0).astype("int16")
    df["abatt_month"]  = df["ABATT_DATE"].dt.month.fillna(0).astype("int8")
    df["abatt_season"] = df["abatt_month"].map(
        lambda m: 0 if m in [3,4,5] else 1 if m in [6,7,8]
                  else 2 if m in [9,10,11] else 3)
    df["birth_year"]   = df["BIRTH_YMD"].dt.year.fillna(0).astype("int16")
    df["birth_month"]  = df["BIRTH_YMD"].dt.month.fillna(0).astype("int8")
    df["birth_season"] = df["birth_month"].map(
        lambda m: 0 if m in [3,4,5] else 1 if m in [6,7,8]
                  else 2 if m in [9,10,11] else 3)

    # ── 범주형 인코딩 ──
    df["sex_code"]  = (df["JUDGE_SEX"]
                       .map({"암":0,"수":1,"거세":2}).fillna(-1).astype("int8"))
    df["sido_code"] = df["sido"].map(SIDO_MAP).fillna(-1).astype("int8")

    # ── 월령 파생 ──
    df["age_sq"]              = (df["AGE"] ** 2).astype("float32")
    df["age_sex"]             = df["AGE"] * df["sex_code"]  # 성별×월령 상호작용
    df["age_abatt_season"]    = df["AGE"] * df["abatt_season"]  # 도축계절×월령

    # ── 도축월 기상 join ──
    wm_s = (wm_monthly
            .rename(columns={c: f"s_{c}"
                              for c in wm_monthly.columns
                              if c not in ["stn","year","month"]})
            .rename(columns={"year":"abatt_year","month":"abatt_month"}))
    df = df.merge(wm_s, on=["stn","abatt_year","abatt_month"], how="left")

    # ── 출생월 기상 join ──
    wm_b = (wm_monthly
            .rename(columns={c: f"b_{c}"
                              for c in wm_monthly.columns
                              if c not in ["stn","year","month"]})
            .rename(columns={"year":"birth_year","month":"birth_month"}))
    df = df.merge(wm_b, on=["stn","birth_year","birth_month"], how="left")

    # ── 도축전 1·2·3개월 기상 join ──
    for lag, prefix in [(1,"p1"), (2,"p2"), (3,"p3")]:
        lag_date  = df["ABATT_DATE"] - pd.DateOffset(months=lag)
        lag_year  = lag_date.dt.year
        lag_month = lag_date.dt.month
        tmp = pd.DataFrame({
            "stn":       df["stn"].values,
            "year":      lag_year.values,
            "month":     lag_month.values,
        })
        tmp = tmp.merge(wm_monthly, on=["stn","year","month"], how="left")
        for col in [c for c in wm_monthly.columns if c not in ["stn","year","month"]]:
            df[f"{prefix}_{col}"] = tmp[col].values

    # ── 농장 피처 join ──
    df = df.merge(area,      on="FARM_UNIQUE_NO", how="left")
    df = df.merge(death_cnt, on="FARM_UNIQUE_NO", how="left")
    df["death_cnt"]  = df["death_cnt"].fillna(0)
    df["death_rate"] = df["death_cnt"] / (df["avg_cattle"].fillna(1) + 1)

    # ── 농장 Target Encoding (추론시) ──
    if not is_train:
        df = df.merge(farm_grade_lookup, on="FARM_UNIQUE_NO", how="left")
        df["farm_grade_loo"]  = df["farm_grade_mean"]  # 추론시엔 LOO=mean
        df["farm_grade_cnt"]  = df["farm_grade_cnt"].fillna(0)
        df["farm_grade_std"]  = df.get("farm_grade_std", pd.Series(dtype=float))
        df["farm_grade_mean"] = df["farm_grade_mean"].fillna(
            df["farm_grade_mean"].mean() if "farm_grade_mean" in df else 7.5)

    df = df.merge(farm_std, on="FARM_UNIQUE_NO", how="left")

    return df


# ─── STEP 6: 사육기간 전체 기상 집계 (핵심 피처) ─────────────────
def compute_rearing_weather(df, wm_monthly, chunk_size=80_000):
    """
    출생~도축 기간의 전체 월별 기상을 집계하여 피처 생성.
    메모리 효율을 위해 chunk_size 단위로 처리.
    """
    print(f"  사육기간 기상 집계 시작 (총 {len(df):,}행, 청크={chunk_size:,}) ...")
    results = []
    n_chunks = (len(df) + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        t0 = time.time()
        start = ci * chunk_size
        end   = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end][["stn","BIRTH_YMD","ABATT_DATE"]].copy()
        chunk["idx_orig"] = range(start, end)

        # 기상 데이터가 있는 기간으로 클립
        chunk["birth_eff"] = chunk["BIRTH_YMD"].clip(lower=WEATHER_START)

        # (stn, year, month) 키로 explode
        valid = chunk.dropna(subset=["birth_eff","ABATT_DATE"])
        valid = valid[valid["birth_eff"] <= valid["ABATT_DATE"]]

        if len(valid) == 0:
            results.append(pd.DataFrame({"idx_orig": range(start, end)}))
            continue

        valid["months"] = valid.apply(
            lambda r: pd.period_range(r["birth_eff"], r["ABATT_DATE"], freq="M").tolist(),
            axis=1
        )
        expanded = (valid[["idx_orig","stn","months"]]
                    .explode("months")
                    .dropna(subset=["months"]))
        expanded["year"]  = expanded["months"].dt.year
        expanded["month"] = expanded["months"].dt.month
        expanded = expanded.merge(
            wm_monthly, on=["stn","year","month"], how="left")

        # 사육기간 집계
        feat_agg = expanded.groupby("idx_orig").agg(
            rear_ta_mean       = ("ta_mean",     "mean"),
            rear_ta_max        = ("ta_max_mean",  "mean"),
            rear_ta_min        = ("ta_min_mean",  "mean"),
            rear_thi_mean      = ("thi_mean",     "mean"),
            rear_thi_max       = ("thi_max",      "max"),
            rear_rhm_mean      = ("rhm_mean",     "mean"),
            rear_heat2d_total  = ("heat2d",       "sum"),
            rear_heat3d_total  = ("heat3d",       "sum"),
            rear_cold_d_total  = ("cold_d",       "sum"),
            rear_rn_total      = ("rn_sum",       "sum"),
            rear_month_cnt     = ("ta_mean",      "count"),
        ).reset_index()

        # 여름(6~8월) 열스트레스 집계
        summer = expanded[expanded["month"].isin([6,7,8])]
        if len(summer):
            summer_agg = summer.groupby("idx_orig").agg(
                rear_summer_thi  = ("thi_mean","mean"),
                rear_summer_heat = ("heat2d",  "sum"),
            ).reset_index()
            feat_agg = feat_agg.merge(summer_agg, on="idx_orig", how="left")

        # 겨울(12,1,2월) 한파 집계
        winter = expanded[expanded["month"].isin([12,1,2])]
        if len(winter):
            winter_agg = winter.groupby("idx_orig").agg(
                rear_winter_ta   = ("ta_mean","mean"),
                rear_winter_cold = ("cold_d", "sum"),
            ).reset_index()
            feat_agg = feat_agg.merge(winter_agg, on="idx_orig", how="left")

        results.append(feat_agg)
        del expanded, valid, chunk; gc.collect()

        if (ci+1) % 5 == 0 or ci == 0:
            pct = (ci+1)/n_chunks*100
            print(f"    청크 {ci+1}/{n_chunks} ({pct:.0f}%) — {elapsed()}")

    all_feats = pd.concat(results, ignore_index=True)
    # idx_orig로 원본 df와 merge
    df_idx = pd.DataFrame({"idx_orig": range(len(df))})
    return df_idx.merge(all_feats, on="idx_orig", how="left").drop("idx_orig", axis=1)


# ─── STEP 7: 훈련 데이터 피처 생성 ───────────────────────────────
print(f"\n[STEP 7] 훈련 데이터 피처 생성 중... ({elapsed()})")

# 농장 target encoding 조회 테이블 (test에서 사용)
farm_grade_lookup = (train.groupby("FARM_UNIQUE_NO")
                          .agg(farm_grade_mean=("grade_score","mean"),
                               farm_grade_cnt =("grade_score","count"))
                          .reset_index())
farm_grade_lookup = farm_grade_lookup.merge(farm_std, on="FARM_UNIQUE_NO", how="left")

train = make_features(train, farm_grade_lookup, wm, is_train=True)

# 사육기간 전체 기상 집계
rear_feats_train = compute_rearing_weather(train, wm)
train = pd.concat([train.reset_index(drop=True), 
                   rear_feats_train.drop("idx_orig", axis=1, errors="ignore")
                                   .reset_index(drop=True)], axis=1)

print(f"  훈련 피처 생성 완료: {train.shape} ({elapsed()})")

# ─── STEP 8: 피처 컬럼 확정 ───────────────────────────────────────
DROP_COLS = {
    "sido","ABATT_DATE","BIRTH_YMD","FARM_UNIQUE_NO","JUDGE_SEX",
    "LAST_GRADE","grade_score",
}
FEATURE_COLS = [c for c in train.columns
                if c not in DROP_COLS and train[c].dtype != object]
print(f"\n  사용 피처 수: {len(FEATURE_COLS)}")
print(f"  피처 목록: {FEATURE_COLS}")

# ─── STEP 9: 타겟 인코딩 + 체크포인트 저장 ──────────────────────
import joblib, os
from sklearn.preprocessing import LabelEncoder

CKPT_DIR = f"{OUT_DIR}/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── X_train 체크포인트 ──
CKPT_X     = f"{CKPT_DIR}/X_train.npy"
CKPT_Y     = f"{CKPT_DIR}/y_train.npy"
CKPT_FEATS = f"{CKPT_DIR}/feature_cols.pkl"

le = LabelEncoder()
le.fit(GRADE_ORDER)

if os.path.exists(CKPT_X) and os.path.exists(CKPT_Y):
    print(f"\n[STEP 9] 체크포인트 발견 → X_train, y_train 로드 중...")
    X_train     = np.load(CKPT_X)
    y_train     = np.load(CKPT_Y)
    FEATURE_COLS = joblib.load(CKPT_FEATS)
    print(f"  로드 완료: X_train {X_train.shape}")
else:
    print(f"\n[STEP 9] 피처 행렬 생성 중...")
    y_train = le.transform(train["LAST_GRADE"].fillna("등외"))
    X_train = train[FEATURE_COLS].fillna(-999).astype("float32").values
    del train; gc.collect()
    # 체크포인트 저장
    np.save(CKPT_X, X_train)
    np.save(CKPT_Y, y_train)
    joblib.dump(FEATURE_COLS, CKPT_FEATS)
    print(f"  저장 완료: {CKPT_X}")

print(f"  X_train shape: {X_train.shape}")

# ─── STEP 10: LightGBM 학습 (Fold별 체크포인트) ──────────────────
print(f"\n[STEP 10] LightGBM 학습 중... ({elapsed()})")

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

N_FOLDS = 3   # ← 5→3으로 변경 (시간 단축)

LGB_PARAMS = dict(
    objective        = "multiclass",
    num_class        = len(GRADE_ORDER),
    metric           = "multi_logloss",
    n_estimators     = 400,      # ← 800→400 (시간 단축)
    learning_rate    = 0.05,
    num_leaves       = 63,       # ← 127→63 (시간 단축)
    min_child_samples= 30,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    class_weight     = "balanced",
    n_jobs           = -1,
    random_state     = 42,
    verbose          = -1,
)

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_preds  = np.zeros((len(y_train), len(GRADE_ORDER)), dtype="float32")
cv_scores  = []
best_iters = []

# ── 이미 완료된 fold 체크포인트 불러오기 ──
for fold in range(N_FOLDS):
    ckpt_fold  = f"{CKPT_DIR}/fold_{fold}.txt"
    ckpt_oof   = f"{CKPT_DIR}/oof_fold_{fold}.npy"
    ckpt_score = f"{CKPT_DIR}/score_fold_{fold}.pkl"

    tr_idx, val_idx = list(kf.split(X_train, y_train))[fold]

    if os.path.exists(ckpt_fold) and os.path.exists(ckpt_oof):
        # 이미 완료된 fold → 체크포인트에서 복원
        booster   = lgb.Booster(model_file=ckpt_fold)
        val_prob  = booster.predict(X_train[val_idx])
        fold_data = joblib.load(ckpt_score)
        fold_f1   = fold_data["f1"]
        best_iter = fold_data["best_iter"]
        oof_preds[val_idx] = np.load(ckpt_oof)
        cv_scores.append(fold_f1)
        best_iters.append(best_iter)
        print(f"  Fold {fold+1}/{N_FOLDS} ✅ 체크포인트 복원 | F1={fold_f1:.4f} | iter={best_iter}")
        continue

    # 새로 학습
    t_fold = time.time()
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    val_prob = model.predict_proba(X_val)
    val_pred = np.argmax(val_prob, axis=1)
    fold_f1  = f1_score(y_val, val_pred, average="macro")
    cv_scores.append(fold_f1)
    best_iters.append(model.best_iteration_)
    oof_preds[val_idx] = val_prob

    # Fold 체크포인트 저장
    model.booster_.save_model(ckpt_fold)
    np.save(ckpt_oof, val_prob)
    joblib.dump({"f1": fold_f1, "best_iter": model.best_iteration_}, ckpt_score)

    print(f"  Fold {fold+1}/{N_FOLDS} | F1={fold_f1:.4f} | iter={model.best_iteration_} "
          f"| {(time.time()-t_fold)/60:.1f}분 → 체크포인트 저장 ✅")

    del model, X_tr, X_val, y_tr, y_val; gc.collect()

oof_pred_labels = np.argmax(oof_preds, axis=1)
oof_f1          = f1_score(y_train, oof_pred_labels, average="macro")
best_iter_avg   = int(np.mean(best_iters) * 1.1)

print(f"\n  ★ OOF Macro-F1  : {oof_f1:.4f}")
print(f"  ★ CV 평균 F1    : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"  ★ Best Iter 평균: {int(np.mean(best_iters))} → 전체 학습용: {best_iter_avg}")

# ── 전체 데이터로 최종 모델 학습 ──
CKPT_FINAL = f"{CKPT_DIR}/final_model.txt"

if os.path.exists(CKPT_FINAL):
    print(f"\n  최종 모델 체크포인트 발견 → 로드 중...")
    final_booster = lgb.Booster(model_file=CKPT_FINAL)
    # 예측용 래퍼 (predict_proba 대신 predict 사용)
    class BoosterWrapper:
        def __init__(self, booster): self.booster = booster
        def predict_proba(self, X): return self.booster.predict(X)
    final_model = BoosterWrapper(final_booster)
    print(f"  로드 완료 ✅")
else:
    print(f"\n  전체 데이터로 최종 모델 학습 중... ({elapsed()})")
    final_params = {**LGB_PARAMS, "n_estimators": best_iter_avg}
    final_model_lgb = lgb.LGBMClassifier(**final_params)
    final_model_lgb.fit(X_train, y_train)
    # 체크포인트 저장
    final_model_lgb.booster_.save_model(CKPT_FINAL)
    print(f"  최종 모델 저장 완료: {CKPT_FINAL} ✅")

    class BoosterWrapper:
        def __init__(self, booster): self.booster = booster
        def predict_proba(self, X): return self.booster.predict(X)
    final_model = BoosterWrapper(final_model_lgb.booster_)
    del final_model_lgb

del X_train; gc.collect()
print(f"  최종 모델 학습 완료 ({elapsed()})")

# ─── STEP 11: 테스트셋 예측 ───────────────────────────────────────
print(f"\n[STEP 11] 테스트셋 예측 중... ({elapsed()})")

TEST_COLS = ["sido","stn","ABATT_DATE","JUDGE_SEX","WEIGHT","AGE",
             "BIRTH_YMD","FARM_UNIQUE_NO"]

test = pd.read_csv(
    f"{DATA_DIR}/test_hanwoo.csv",
    usecols=[c for c in TEST_COLS if c in
             pd.read_csv(f"{DATA_DIR}/test_hanwoo.csv", nrows=0).columns],
    dtype={"FARM_UNIQUE_NO": str},
    low_memory=False,
)
test["ABATT_DATE"] = pd.to_datetime(test["ABATT_DATE"], errors="coerce")
test["BIRTH_YMD"]  = pd.to_datetime(
    test["BIRTH_YMD"].astype(str), format="%Y%m%d", errors="coerce")

# LAST_GRADE 컬럼이 있으면 제거
if "LAST_GRADE" in test.columns:
    test = test.drop("LAST_GRADE", axis=1)

test = make_features(test, farm_grade_lookup, wm, is_train=False)

rear_feats_test = compute_rearing_weather(test, wm)
test = pd.concat([test.reset_index(drop=True),
                  rear_feats_test.drop("idx_orig", axis=1, errors="ignore")
                                 .reset_index(drop=True)], axis=1)

# 훈련에 없는 피처 채우기
for col in FEATURE_COLS:
    if col not in test.columns:
        test[col] = -999

X_test     = test[FEATURE_COLS].fillna(-999).astype("float32").values
pred_proba = final_model.predict_proba(X_test)
pred_label = le.inverse_transform(np.argmax(pred_proba, axis=1))

# ─── STEP 12: 저장 ─────────────────────────────────────────────────
print(f"\n[STEP 12] 결과 저장 중... ({elapsed()})")

# 제출 파일
pd.DataFrame({"LAST_GRADE": pred_label}).to_csv(
    f"{OUT_DIR}/260418.csv", index=False, encoding="utf-8-sig")

# 피처 중요도
fi = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": final_model.feature_importances_,
}).sort_values("importance", ascending=False)
fi.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

# OOF 예측 저장 (분석용)
pd.DataFrame(oof_preds, columns=le.classes_).to_csv(
    f"{OUT_DIR}/oof_predictions.csv", index=False)

# ─── 결과 요약 ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ★ OOF Macro-F1     : {oof_f1:.4f}")
print(f"  ★ CV 평균 F1       : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"  ★ 사용 피처 수     : {len(FEATURE_COLS)}개")
print(f"  ★ 총 소요 시간     : {elapsed()}")
print(f"  ★ 제출 파일        : {OUT_DIR}/260418.csv")
print(f"{'='*60}")

print(f"\n  예측 분포 (상위 등급 순):")
dist = pd.Series(pred_label).value_counts()
for grade in GRADE_ORDER:
    cnt = dist.get(grade, 0)
    print(f"    {grade:5s}: {cnt:6,}건 ({cnt/len(pred_label)*100:.1f}%)")

print(f"\n  피처 중요도 Top 20:")
print(fi.head(20).to_string(index=False))
