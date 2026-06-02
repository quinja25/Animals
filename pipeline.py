"""
한우 도체 등급 예측 파이프라인
- RAM 4GB 최적화 버전
- test에 없는 도체측정값(BACKFAT/REA/INSFAT 등) 사용 불가
- 기상+농장+혈통+기본정보로 예측
"""

import pandas as pd
import numpy as np
import warnings, gc
warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = "/sessions/happy-practical-brahmagupta/mnt/uploads"
OUT_DIR  = "/sessions/happy-practical-brahmagupta/mnt/Animals"

print("=" * 50)
print("한우 도체 등급 예측 파이프라인")
print("=" * 50)

# ─── STEP 0: 보조 통계 사전 계산 ───
print("\n[STEP 0] 부별/농장별 통계 계산...")

# 혈통 로드
lineage = pd.read_csv(f"{DATA_DIR}/hanwoo_lineage.csv",
                      usecols=["CATTLE_NO","FATHER_CATTLE_NO","KPN_NO"],
                      dtype=str)

# train에서 INSFAT/BACKFAT 컬럼만 읽어서 부별 통계 계산
mini = pd.read_csv(f"{DATA_DIR}/hanwoo_train.csv",
                   usecols=["CATTLE_NO","FARM_UNIQUE_NO","INSFAT","BACKFAT"],
                   dtype={"CATTLE_NO":str,"FARM_UNIQUE_NO":str,
                          "INSFAT":float,"BACKFAT":float})
mini["INSFAT"]  = mini["INSFAT"].replace(-99, np.nan)
mini["BACKFAT"] = mini["BACKFAT"].replace(-99, np.nan)
mini = mini.merge(lineage, on="CATTLE_NO", how="left")

# 부별 근내지방 평균 (유전력 프록시)
father_df = (mini.dropna(subset=["INSFAT","FATHER_CATTLE_NO"])
             .groupby("FATHER_CATTLE_NO")["INSFAT"].mean()
             .reset_index().rename(columns={"INSFAT":"father_insfat"}))

# KPN별 근내지방 평균
kpn_df = (mini.dropna(subset=["INSFAT","KPN_NO"])
           .groupby("KPN_NO")["INSFAT"].mean()
           .reset_index().rename(columns={"INSFAT":"kpn_insfat"}))

# 농장별 평균 근내지방/등지방 (농장 수준 품질 지표)
farm_q = mini.groupby("FARM_UNIQUE_NO").agg(
    farm_insfat=("INSFAT",  "mean"),
    farm_backfat=("BACKFAT", "mean"),
).reset_index()

del mini; gc.collect()
print(f"  부: {len(father_df):,}, KPN: {len(kpn_df):,}, 농장: {len(farm_q):,}")

# ─── STEP 1: 기상 집계 ───
print("\n[STEP 1] 기상 데이터 월별 집계...")

w = pd.read_csv(f"{DATA_DIR}/hanwoo_weather.csv", parse_dates=["date"])
w["ta_avg"] = (w["ta_max"] + w["ta_min"]) / 2
w["THI"]    = (1.8*w["ta_avg"]+32) - (0.55 - 0.0055*w["rhm_avg"]) * (1.8*w["ta_avg"]-26)
w["heat2"]  = (w["THI"] >= 80).astype(int)   # 중등도 이상
w["heat3"]  = (w["THI"] >= 85).astype(int)   # 심각
w["cold"]   = (w["ta_min"] < -5).astype(int)
w["year"]   = w["date"].dt.year
w["month"]  = w["date"].dt.month

wm = w.groupby(["stn","year","month"]).agg(
    ta_mean  = ("ta_avg",  "mean"),
    thi_mean = ("THI",     "mean"),
    thi_max  = ("THI",     "max"),
    rhm_mean = ("rhm_avg", "mean"),
    heat2d   = ("heat2",   "sum"),
    heat3d   = ("heat3",   "sum"),
    cold_d   = ("cold",    "sum"),
    rn_sum   = ("rn_day",  "sum"),
).reset_index()
for c in wm.select_dtypes("float64").columns:
    wm[c] = wm[c].astype("float32")
del w; gc.collect()

wm_s = (wm.rename(columns={c:f"s_{c}" for c in wm.columns if c not in ["stn","year","month"]})
          .rename(columns={"year":"abatt_year","month":"abatt_month"}))
wm_b = (wm.rename(columns={c:f"b_{c}" for c in wm.columns if c not in ["stn","year","month"]})
          .rename(columns={"year":"birth_year","month":"birth_month"}))
del wm; gc.collect()
print(f"  기상 집계 완료")

# ─── STEP 2: 농장 피처 ───
print("\n[STEP 2] 농장 피처...")

area  = pd.read_csv(f"{DATA_DIR}/hanwoo_area.csv",
                    dtype={"FARM_UNIQUE_NO":str})
death = pd.read_csv(f"{DATA_DIR}/hanwoo_death.csv",
                    usecols=["FARM_UNIQUE_NO"], dtype=str)
area["avg_cattle"] = area[["C2023","C2024","C2025"]].mean(axis=1)
area["density"]    = area["avg_cattle"] / area["AREA"].replace(0, np.nan)
area = area[["FARM_UNIQUE_NO","avg_cattle","density"]]
death_cnt = death.groupby("FARM_UNIQUE_NO").size().reset_index(name="death_cnt")
del death; gc.collect()

# ─── STEP 3: 공통 피처 생성 ───
GRADE_ORDER = ["1++A","1++B","1++C","1+A","1+B","1+C",
               "1A","1B","1C","2A","2B","2C","3A","3B","3C","등외"]
SIDO_MAP = {s:i for i,s in enumerate([
    "강원특별자치도","경기도","경상남도","경상북도","광주광역시","대구광역시",
    "대전광역시","부산광역시","서울특별시","세종특별자치시","울산광역시",
    "인천광역시","전라남도","전라북도","전라북도특별자치도","제주특별자치도",
    "충청남도","충청북도"])}

def make_features(df):
    df["ABATT_DATE"] = pd.to_datetime(df["ABATT_DATE"], errors="coerce")
    df["BIRTH_YMD"]  = pd.to_datetime(df["BIRTH_YMD"].astype(str), format="%Y%m%d", errors="coerce")

    df["abatt_year"]   = df["ABATT_DATE"].dt.year.fillna(0).astype(int)
    df["abatt_month"]  = df["ABATT_DATE"].dt.month.fillna(0).astype(int)
    df["abatt_season"] = df["abatt_month"].map(
        lambda m: 0 if m in [3,4,5] else 1 if m in [6,7,8] else 2 if m in [9,10,11] else 3)

    df["birth_year"]   = df["BIRTH_YMD"].dt.year.fillna(0).astype(int)
    df["birth_month"]  = df["BIRTH_YMD"].dt.month.fillna(0).astype(int)
    df["birth_season"] = df["birth_month"].map(
        lambda m: 0 if m in [3,4,5] else 1 if m in [6,7,8] else 2 if m in [9,10,11] else 3)

    df["sex_code"]  = df["JUDGE_SEX"].map({"암":0,"수":1,"거세":2}).fillna(-1).astype(int)
    df["sido_code"] = df["sido"].map(SIDO_MAP).fillna(-1).astype(int)

    # 기상 조인
    df = df.merge(wm_s, on=["stn","abatt_year","abatt_month"], how="left")
    df = df.merge(wm_b, on=["stn","birth_year","birth_month"],  how="left")

    # 농장
    df = df.merge(area,      on="FARM_UNIQUE_NO", how="left")
    df = df.merge(death_cnt, on="FARM_UNIQUE_NO", how="left")
    df["death_cnt"]  = df["death_cnt"].fillna(0)
    df["death_rate"] = df["death_cnt"] / (df["avg_cattle"].fillna(1) + 1)
    df = df.merge(farm_q,   on="FARM_UNIQUE_NO", how="left")

    # 혈통
    df["CATTLE_NO"] = df["CATTLE_NO"].astype(str)
    df = df.merge(lineage, on="CATTLE_NO", how="left")
    df = df.merge(father_df, on="FATHER_CATTLE_NO", how="left")
    df = df.merge(kpn_df,    on="KPN_NO",            how="left")

    return df

# ─── STEP 4: 학습 ───
print("\n[STEP 3] train 로드 및 피처 생성...")

LOAD_COLS = ["sido","stn","ABATT_DATE","JUDGE_SEX","WEIGHT","AGE",
             "BIRTH_YMD","CATTLE_NO","FARM_UNIQUE_NO","LAST_GRADE"]
train = pd.read_csv(f"{DATA_DIR}/hanwoo_train.csv", usecols=LOAD_COLS,
                    dtype={"CATTLE_NO":str,"FARM_UNIQUE_NO":str})
# 계층적 샘플링: 클래스 비율 유지하며 600K 샘플
N_SAMPLE = 600_000
train = (train.groupby("LAST_GRADE", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), int(N_SAMPLE * len(g)/len(train))+1), random_state=42))
              .reset_index(drop=True))
print(f"  샘플링 후 train 크기: {len(train):,}")
train = make_features(train)
gc.collect()

DROP = {"sido","ABATT_DATE","BIRTH_YMD","CATTLE_NO","FARM_UNIQUE_NO",
        "JUDGE_SEX","LAST_GRADE","FATHER_CATTLE_NO","MOTHER_ANIMAL_NO",
        "F_GMOTHER_ANIMAL_NO","F_GFATHER_CATTLE_NO","M_GMOTHER_ANIMAL_NO",
        "M_GFATHER_CATTLE_NO","KPN_NO"}
FEATURE_COLS = [c for c in train.columns
                if c not in DROP and train[c].dtype != object]

print(f"  피처 수: {len(FEATURE_COLS)}")
print(f"  피처: {FEATURE_COLS}")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder(); le.fit(GRADE_ORDER)
y = le.transform(train["LAST_GRADE"].fillna("등외"))
X = train[FEATURE_COLS].fillna(-999).astype("float32").values
del train; gc.collect()

print("\n[STEP 4] LightGBM 학습 (80/20 split)...")
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
del X; gc.collect()

model = lgb.LGBMClassifier(
    objective="multiclass", num_class=len(le.classes_),
    metric="multi_logloss", n_estimators=300, learning_rate=0.1,
    num_leaves=63, min_child_samples=50, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1,
)
model.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])

val_pred = model.predict(X_val)
val_f1   = f1_score(y_val, val_pred, average="macro")
print(f"\n  ✓ Validation Macro-F1 : {val_f1:.4f}")
print(f"  Best iteration       : {model.best_iteration_}")
del X_tr, X_val, y_tr, y_val; gc.collect()

# ─── STEP 5: 예측 ───
print("\n[STEP 5] test 예측...")
TEST_COLS = ["sido","stn","ABATT_DATE","JUDGE_SEX","WEIGHT","AGE",
             "BIRTH_YMD","CATTLE_NO","FARM_UNIQUE_NO","LAST_GRADE"]
test = pd.read_csv(f"{DATA_DIR}/test_hanwoo.csv", usecols=TEST_COLS,
                   dtype={"CATTLE_NO":str,"FARM_UNIQUE_NO":str})
test = make_features(test)

X_test     = test[FEATURE_COLS].fillna(-999).astype("float32").values
pred_label = le.inverse_transform(model.predict(X_test))

# 제출 파일
pd.DataFrame({"LAST_GRADE": pred_label}).to_csv(
    f"{OUT_DIR}/260418.csv", index=False, encoding="utf-8-sig")

# 피처 중요도
fi = pd.DataFrame({"feature":FEATURE_COLS, "importance":model.feature_importances_})
fi.sort_values("importance", ascending=False).to_csv(
    f"{OUT_DIR}/feature_importance.csv", index=False)

print(f"\n  제출 파일: {OUT_DIR}/260418.csv")
print(f"\n  예측 분포:")
print(pd.Series(pred_label).value_counts().sort_index().to_string())
print(f"\n  상위 15 피처:")
print(fi.sort_values("importance",ascending=False).head(15).to_string(index=False))
print(f"\n{'='*50}")
print(f"  최종 Validation Macro-F1: {val_f1:.4f}")
print(f"{'='*50}")
