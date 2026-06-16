
"""
한우 도체 등급 예측 파이프라인 v4 (Generalization Focus)
─────────────────────────────────────────────────────
v3 대비 주요 변경 사항:
  1. Unseen Farm 문제 해결: FARM_UNIQUE_NO 중복 0% 대응을 위해 농장 Target Encoding 제거
  2. 혈통 및 유전능력 강화: KPN별 등급 통계 + KPN 육종가(Breeding Value) 피처 추가
  3. 지역성 반영: stn(관측소)별 평균 등급 통계 추가 (지역 기후/환경 효과)
  4. 신뢰도 높은 검증: GroupKFold(groups=FARM_UNIQUE_NO) 도입으로 실제 리더보드 점수와 동기화
  5. 농장 속성 피처: 농장 ID 대신 사육 밀도, 폐사율, 규모 등 관리 지표 사용
"""

import pandas as pd
import numpy as np
import gc
import time
import warnings
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── 경로 설정 ───────────────────────────
# 현재 파일(pipeline_v4.py)의 위치를 기준으로 상위 폴더(루트)를 찾음
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "submissions", "v4")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ─── 상수 설정 ───────────────────────────
GRADE_ORDER = ["1++A","1++B","1++C","1+A","1+B","1+C","1A","1B","1C","2A","2B","2C","3A","3B","3C","등외"]
GRADE_SCORE = {g: 15-i for i, g in enumerate(GRADE_ORDER)}

def elapsed(start_time):
    return f"{(time.time() - start_time) / 60:.1f}분"

# ─── STEP 1: 보조 데이터 로드 및 전처리 ────────────────
print("\n[STEP 1] 보조 데이터 로드 및 정제...")
t_start = time.time()

# 1. 기상 데이터 월별 집계
weather = pd.read_csv(f"{DATA_DIR}/hanwoo_weather.csv", parse_dates=["date"])
weather["ta_avg"] = (weather["ta_max"] + weather["ta_min"]) / 2
weather["year"], weather["month"] = weather["date"].dt.year, weather["date"].dt.month
wm = weather.groupby(["stn","year","month"]).agg(
    ta_mean = ("ta_avg", "mean"),
    rn_sum  = ("rn_day", "sum"),
    rhm_mean= ("rhm_avg", "mean")
).reset_index().astype("float32")
del weather; gc.collect()

# 2. 농장 정보 (area, death)
area = pd.read_csv(f"{DATA_DIR}/hanwoo_area.csv").drop_duplicates('FARM_UNIQUE_NO')
death = pd.read_csv(f"{DATA_DIR}/hanwoo_death.csv")
death_cnt = death.groupby("FARM_UNIQUE_NO").size().reset_index(name="death_cnt")
area = area.merge(death_cnt, on="FARM_UNIQUE_NO", how="left").fillna(0)
area['density'] = area['AREA'] / (area['C2023'] + 1)
area['farm_size'] = pd.cut(area['C2023'], bins=[-1, 30, 100, 100000], labels=[0, 1, 2])
area['farm_size'] = area['farm_size'].cat.add_categories([-1]).fillna(-1).astype(int)
del death, death_cnt; gc.collect()

# 3. KPN 유전능력 (엑셀)
kpn_bv = pd.read_excel(f"{DATA_DIR}/KPN 유전능력 자료.xlsx")
kpn_cols = {
    'KPN명호': 'KPN_NO',
    '도체중 육종가': 'kpn_weight_bv',
    '등심단면적 육종가': 'kpn_rea_bv',
    '등지방두께 육종가': 'kpn_backfat_bv',
    '근내지방도 육종가': 'kpn_insfat_bv'
}
kpn_bv = kpn_bv[list(kpn_cols.keys())].rename(columns=kpn_cols)
# 엑셀 KPN 번호의 암호화 형식이 다를 수 있으므로 정합성 주의 필요

# 4. 혈통 데이터 (KPN_NO 확보용)
lineage = pd.read_csv(f"{DATA_DIR}/hanwoo_lineage_0612.csv", usecols=['CATTLE_NO', 'KPN_NO'])

print(f"  보조 데이터 준비 완료 ({elapsed(t_start)})")

# ─── STEP 2: 피처 엔지니어링 및 클렌징 ─────────────────────
def make_features_v4(df, is_train=True):
    df = df.copy()
    
    # 1. 날짜 변환 및 파생 피처 생성
    df["ABATT_DATE"] = pd.to_datetime(df["ABATT_DATE"], errors='coerce')
    df["BIRTH_YMD"]  = pd.to_datetime(df["BIRTH_YMD"].astype(str), format="%Y%m%d", errors='coerce')
    
    df["abatt_year"] = df["ABATT_DATE"].dt.year.fillna(0).astype(int)
    df["abatt_month"] = df["ABATT_DATE"].dt.month.fillna(0).astype(int)
    df["birth_year"] = df["BIRTH_YMD"].dt.year.fillna(0).astype(int)
    df["birth_month"] = df["BIRTH_YMD"].dt.month.fillna(0).astype(int)
    
    # 2. 성별 인코딩 (수치형 변환)
    df["sex_code"] = df["JUDGE_SEX"].map({"암":0, "수":1, "거세":2}).fillna(-1).astype(int)
    
    # 3. 조인: 기상 및 농장 속성
    # wm (기상) 조인 시 타입을 명시적으로 float32로 유지
    df = df.merge(wm.rename(columns={"ta_mean":"s_ta", "rn_sum":"s_rn", "rhm_mean":"s_rhm"}), 
                  left_on=["stn","abatt_year","abatt_month"], right_on=["stn","year","month"], how="left")
    
    df = df.merge(area[['FARM_UNIQUE_NO', 'density', 'farm_size', 'death_cnt']], on='FARM_UNIQUE_NO', how='left')
    
    # 4. 혈통 및 유전능력 매핑
    lineage_clean = lineage.drop_duplicates('CATTLE_NO')
    df = df.merge(lineage_clean, on='CATTLE_NO', how='left')
    df = df.merge(kpn_bv, on='KPN_NO', how='left')
    
    # 5. 수치형 변환 불가능한 원본 컬럼 제거
    # 'ABATT_DATE' 등 날짜 객체와 'JUDGE_SEX' 등 문자열 컬럼을 명시적으로 제거
    drop_cols = [
        "year", "month", "JUDGE_SEX", "sido", "sigungu", "eupmyeondong", 
        "CATTLE_NO", "ABATT_DATE", "BIRTH_YMD", "JUDGE_DATE"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    return df

# ─── STEP 3: 훈련 데이터 로드 및 피처 생성 ────────────────
print("\n[STEP 3] 훈련 데이터 구성 중...")
train = pd.read_csv(f"{DATA_DIR}/hanwoo_train.csv")
train = make_features_v4(train, is_train=True)

# Target Encoding 계산
train['grade_score'] = train['LAST_GRADE'].map(GRADE_SCORE)
kpn_stats = train.groupby('KPN_NO')['grade_score'].agg(['mean', 'count']).rename(columns={'mean':'kpn_grade_avg', 'count':'kpn_grade_cnt'}).reset_index()
stn_stats = train.groupby('stn')['grade_score'].mean().reset_index().rename(columns={'grade_score':'stn_grade_avg'})

# 통계 매핑
train = train.merge(kpn_stats, on='KPN_NO', how='left')
train = train.merge(stn_stats, on='stn', how='left')

# 모델 입력용 피처 선택 (수치형이 아닌 모든 컬럼 제외)
# 🚨 중요: 테스트 셋에 없는 등급 관련 직접 피처들을 반드시 제외해야 함 (Data Leakage 방지)
TARGET_RELATED_COLS = [
    'BACKFAT', 'REA', 'WINDEX', 'WGRADE', 'INSFAT', 'YUKSAK', 
    'FATSAK', 'TISSUE', 'GROWTH', 'COST_AMT', 'WEIGHT' # WEIGHT는 테스트에 있지만 보통 제외하거나 신중히 사용
]
NON_FEATURE_COLS = ["LAST_GRADE", "grade_score", "FARM_UNIQUE_NO", "KPN_NO"] + TARGET_RELATED_COLS

FEATURES = [c for c in train.columns if c not in NON_FEATURE_COLS and train[c].dtype in ['int32', 'int64', 'float32', 'float64']]
# WEIGHT는 테스트셋에도 있으므로 피처에 포함 (지침 확인 결과 사용 가능)
if 'WEIGHT' in train.columns:
    FEATURES.append('WEIGHT')

print(f"  피처 수: {len(FEATURES)}")
print(f"  사용 피처: {FEATURES}")

le = LabelEncoder()
y = le.fit_transform(train["LAST_GRADE"].fillna("등외"))
X = train[FEATURES].fillna(-999).astype("float32")
groups = train["FARM_UNIQUE_NO"]

print(f"  피처 수: {len(FEATURES)}")
print(f"  사용 피처: {FEATURES}")

# ─── STEP 4: 학습 (GroupKFold) ───────────────────────────────
print("\n[STEP 4] LightGBM 학습 (GroupKFold)...")
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": len(GRADE_ORDER),
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}

models = []
oof_f1 = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=1000)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    
    val_pred = model.predict(X_val)
    score = f1_score(y_val, val_pred, average="macro")
    oof_f1.append(score)
    models.append(model)
    print(f"  Fold {fold+1} Macro-F1: {score:.4f}")

print(f"\n  ★ CV Macro-F1 (GroupKFold): {np.mean(oof_f1):.4f} ± {np.std(oof_f1):.4f}")

# ─── STEP 5: 테스트 예측 ───────────────────────────────────────
print("\n[STEP 5] 테스트 예측 및 제출 파일 생성...")
test_orig = pd.read_csv(f"{DATA_DIR}/test_hanwoo.csv")
test = make_features_v4(test_orig, is_train=False)

# 훈련 셋 통계 매핑
test = test.merge(kpn_stats, on='KPN_NO', how='left')
test = test.merge(stn_stats, on='stn', how='left')

X_test = test[FEATURES].fillna(-999).astype("float32")

# 앙상블 (확률 평균)
test_probs = np.zeros((len(X_test), len(GRADE_ORDER)))
for model in models:
    test_probs += model.predict_proba(X_test) / n_splits

pred_labels = le.inverse_transform(np.argmax(test_probs, axis=1))
test_orig["LAST_GRADE"] = pred_labels

out_path = f"{OUT_DIR}/260418.csv"
test_orig.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"  ✅ 제출 파일 저장 완료: {out_path}")

# 피처 중요도 저장
fi = pd.DataFrame({'feature': FEATURES, 'importance': models[0].feature_importances_}).sort_values('importance', ascending=False)
fi.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
