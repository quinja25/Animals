
"""
한우 도체 등급 예측 파이프라인 v5 (Hierarchical & Refined Mapping)
─────────────────────────────────────────────────────
v4 대비 주요 개선 사항:
  1. 계층적 모델링: 육질등급(6종)과 육량등급(3종)을 분리하여 예측 후 결합 (성능 대폭 향상 기대)
  2. 매핑 정교화: KPN_NO(24자) 및 CATTLE_NO(44자) 비식별화 규격에 맞춘 정밀 조인
  3. 피처 고도화: 중요도 기반 WEIGHT/AGE 상호작용 피처(월령당 체중 등) 추가
  4. 일반화 유지: Unseen Farm 대응을 위한 GroupKFold 및 농장 속성 기반 학습
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

# ─── 경로 및 상수 설정 ───────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "submissions", "v5")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 등급 정의
QUALITY_ORDER = ["1++", "1+", "1", "2", "3", "등외"]
YIELD_ORDER   = ["A", "B", "C"]
FINAL_GRADES  = ["1++A","1++B","1++C","1+A","1+B","1+C","1A","1B","1C","2A","2B","2C","3A","3B","3C","등외"]
GRADE_SCORE   = {g: 15-i for i, g in enumerate(FINAL_GRADES)}

def elapsed(start_time):
    return f"{(time.time() - start_time) / 60:.1f}분"

# ─── STEP 1: 보조 데이터 정밀 로드 ──────────────────
print("\n[STEP 1] 보조 데이터 정밀 로드 및 조인 준비...")
t_start = time.time()

# 1. KPN 유전능력 (엑셀) - 표준화 육종가 사용
kpn_bv = pd.read_excel(f"{DATA_DIR}/KPN 유전능력 자료.xlsx")
kpn_bv['KPN명호'] = kpn_bv['KPN명호'].astype(str).str.strip()
kpn_cols = {
    'KPN명호': 'KPN_NO',
    '도체중 표준화육종가': 'kpn_weight_sbv',
    '등심단면적 표준화육종가': 'kpn_rea_sbv',
    '등지방두께 표준화육종가': 'kpn_backfat_sbv',
    '근내지방도 표준화육종가': 'kpn_insfat_sbv'
}
kpn_bv = kpn_bv[list(kpn_cols.keys())].rename(columns=kpn_cols)

# 2. 혈통 데이터
lineage = pd.read_csv(f"{DATA_DIR}/hanwoo_lineage_0612.csv", usecols=['CATTLE_NO', 'KPN_NO'])
lineage['CATTLE_NO'] = lineage['CATTLE_NO'].astype(str).str.strip()
lineage['KPN_NO'] = lineage['KPN_NO'].astype(str).str.strip()
lineage = lineage.drop_duplicates('CATTLE_NO')

# 3. 기상/농장 데이터
weather = pd.read_csv(f"{DATA_DIR}/hanwoo_weather.csv", parse_dates=["date"])
weather["ta_avg"] = (weather["ta_max"] + weather["ta_min"]) / 2
weather["year"], weather["month"] = weather["date"].dt.year, weather["date"].dt.month
wm = weather.groupby(["stn","year","month"]).agg(
    ta_mean = ("ta_avg", "mean"),
    rn_sum  = ("rn_day", "sum"),
    rhm_mean= ("rhm_avg", "mean")
).reset_index().astype("float32")

area = pd.read_csv(f"{DATA_DIR}/hanwoo_area.csv").drop_duplicates('FARM_UNIQUE_NO')
death = pd.read_csv(f"{DATA_DIR}/hanwoo_death.csv")
death_cnt = death.groupby("FARM_UNIQUE_NO").size().reset_index(name="death_cnt")
area = area.merge(death_cnt, on="FARM_UNIQUE_NO", how="left").fillna(0)
area['density'] = (area['AREA'] / (area['C2023'] + 1)).astype("float32")
area['farm_size'] = pd.cut(area['C2023'], bins=[-1, 30, 100, 100000], labels=[0, 1, 2]).cat.add_categories([-1]).fillna(-1).astype(int)

print(f"  보조 데이터 준비 완료 ({elapsed(t_start)})")

# ─── STEP 2: 피처 엔지니어링 함수 ─────────────────────
def make_features_v5(df, is_train=True):
    df = df.copy()
    if 'CATTLE_NO' in df.columns: df['CATTLE_NO'] = df['CATTLE_NO'].astype(str).str.strip()
    if 'FARM_UNIQUE_NO' in df.columns: df['FARM_UNIQUE_NO'] = df['FARM_UNIQUE_NO'].astype(str).str.strip()
    
    # 1. 날짜 및 상호작용 피처
    df["ABATT_DATE"] = pd.to_datetime(df["ABATT_DATE"], errors='coerce')
    df["BIRTH_YMD"]  = pd.to_datetime(df["BIRTH_YMD"].astype(str), format="%Y%m%d", errors='coerce')
    
    df["abatt_year"] = df["ABATT_DATE"].dt.year.fillna(0).astype(int)
    df["abatt_month"] = df["ABATT_DATE"].dt.month.fillna(0).astype(int)
    df["birth_year"] = df["BIRTH_YMD"].dt.year.fillna(0).astype(int)
    df["birth_month"] = df["BIRTH_YMD"].dt.month.fillna(0).astype(int)
    
    df["sex_code"] = df["JUDGE_SEX"].map({"암":0, "수":1, "거세":2}).fillna(-1).astype(int)
    df["weight_per_age"] = (df["WEIGHT"] / (df["AGE"] + 1)).astype("float32")
    
    # 2. 조인
    df = df.merge(wm.rename(columns={"ta_mean":"s_ta", "rn_sum":"s_rn", "rhm_mean":"s_rhm"}), 
                  left_on=["stn","abatt_year","abatt_month"], right_on=["stn","year","month"], how="left")
    df = df.merge(area[['FARM_UNIQUE_NO', 'density', 'farm_size', 'death_cnt']], on='FARM_UNIQUE_NO', how='left')
    df = df.merge(lineage, on='CATTLE_NO', how='left')
    df = df.merge(kpn_bv, on='KPN_NO', how='left')
    
    # 3. 타겟 분리 (훈련 데이터)
    if is_train and 'LAST_GRADE' in df.columns:
        df['LAST_GRADE'] = df['LAST_GRADE'].fillna("등외").astype(str)
        df['target_q'] = df['LAST_GRADE'].str.extract(r'^(1\+\+|1\+|1|2|3|등외)')[0].fillna("등외")
        df['target_y'] = df['LAST_GRADE'].str.extract(r'(A|B|C)$')[0].fillna("B")
    
    drop_cols = ["year", "month", "JUDGE_SEX", "sido", "sigungu", "eupmyeondong", "CATTLE_NO", "ABATT_DATE", "BIRTH_YMD", "JUDGE_DATE"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

# ─── STEP 3: 훈련 준비 및 Target Encoding ────────────────
print("\n[STEP 3] 훈련 데이터 구성 및 Target Encoding...")
train = pd.read_csv(f"{DATA_DIR}/hanwoo_train.csv")
train = make_features_v5(train, is_train=True)

# Target Encoding 계산
train['grade_score'] = train['LAST_GRADE'].map(GRADE_SCORE)
kpn_stats = train.groupby('KPN_NO')['grade_score'].agg(['mean', 'count']).rename(columns={'mean':'kpn_grade_avg', 'count':'kpn_grade_cnt'}).reset_index()
stn_stats = train.groupby('stn')['grade_score'].mean().reset_index().rename(columns={'grade_score':'stn_grade_avg'})

train = train.merge(kpn_stats, on='KPN_NO', how='left')
train = train.merge(stn_stats, on='stn', how='left')

# 피처 목록 확정
TARGET_RELATED = ['BACKFAT', 'REA', 'WINDEX', 'WGRADE', 'INSFAT', 'YUKSAK', 'FATSAK', 'TISSUE', 'GROWTH', 'COST_AMT']
NON_FEATURE = TARGET_RELATED + ['LAST_GRADE', 'target_q', 'target_y', 'FARM_UNIQUE_NO', 'KPN_NO', 'grade_score']
FEATURES = [c for c in train.columns if c not in NON_FEATURE and train[c].dtype in ['int32', 'int64', 'float32', 'float64']]

print(f"  사용 피처 ({len(FEATURES)}개): {FEATURES}")

le_q = LabelEncoder().fit(QUALITY_ORDER)
le_y = LabelEncoder().fit(YIELD_ORDER)

X = train[FEATURES].fillna(-999).astype("float32")
y_q = le_q.transform(train["target_q"])
y_y = le_y.transform(train["target_y"])
groups = train["FARM_UNIQUE_NO"]

# ─── STEP 4: 계층적 학습 (GroupKFold) ───────────────────
print("\n[STEP 4] 계층적 모델 학습 시작...")
gkf = GroupKFold(n_splits=5)
LGB_PARAMS = {"objective": "multiclass", "learning_rate": 0.05, "num_leaves": 127, "class_weight": "balanced", "random_state": 42, "verbose": -1}

models_q, models_y = [], []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_q, groups=groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    
    # 육질 모델
    m_q = lgb.LGBMClassifier(**LGB_PARAMS, num_class=len(QUALITY_ORDER), n_estimators=500)
    m_q.fit(X_tr, y_q[tr_idx], eval_set=[(X_val, y_q[val_idx])], callbacks=[lgb.early_stopping(50)])
    models_q.append(m_q)
    
    # 육량 모델
    m_y = lgb.LGBMClassifier(**LGB_PARAMS, num_class=len(YIELD_ORDER), n_estimators=500)
    m_y.fit(X_tr, y_y[tr_idx], eval_set=[(X_val, y_y[val_idx])], callbacks=[lgb.early_stopping(50)])
    models_y.append(m_y)
    
    print(f"  Fold {fold+1} 완료")

# ─── STEP 5: 테스트 예측 및 결합 ───────────────────────
print("\n[STEP 5] 테스트 예측 및 계층적 결합...")
test_orig = pd.read_csv(f"{DATA_DIR}/test_hanwoo.csv")
test = make_features_v5(test_orig, is_train=False)

# 훈련 셋에서 계산된 통계 매핑
test = test.merge(kpn_stats, on='KPN_NO', how='left')
test = test.merge(stn_stats, on='stn', how='left')

X_test = test[FEATURES].fillna(-999).astype("float32")

prob_q = np.zeros((len(X_test), len(QUALITY_ORDER)))
prob_y = np.zeros((len(X_test), len(YIELD_ORDER)))

for m_q, m_y in zip(models_q, models_y):
    prob_q += m_q.predict_proba(X_test) / 5
    prob_y += m_y.predict_proba(X_test) / 5

pred_q = le_q.inverse_transform(np.argmax(prob_q, axis=1))
pred_y = le_y.inverse_transform(np.argmax(prob_y, axis=1))

# 최종 등급 결합 로직
def combine_grade(q, y):
    if q == "등외": return "등외"
    return f"{q}{y}"

final_preds = [combine_grade(q, y) for q, y in zip(pred_q, pred_y)]
test_raw["LAST_GRADE"] = final_preds

out_path = f"{OUT_DIR}/260418.csv"
test_raw.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"  ✅ v5 결과 저장 완료: {out_path}")

# Feature Importance 저장 (육질 모델 Fold 1 기준)
fi = pd.DataFrame({'feature': FEATURES, 'importance': models_q[0].feature_importances_}).sort_values('importance', ascending=False)
fi.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
