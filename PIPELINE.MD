# 프로젝트 파이프라인

## 핵심 전제: 테스트셋 피처 확인

`test_hanwoo.csv`에는 등급 판정 기준 피처가 **없습니다.**

| 구분 | 테스트셋에 있음 | 테스트셋에 없음 |
|------|:-----------:|:-----------:|
| 육량 관련 | WEIGHT ✅ | BACKFAT ❌ · REA ❌ · WINDEX ❌ · WGRADE ❌ |
| 육질 관련 | — | INSFAT ❌ · YUKSAK ❌ · FATSAK ❌ · TISSUE ❌ · GROWTH ❌ |
| 가격 | — | COST_AMT ❌ |
| 사용 가능 | `WEIGHT` `JUDGE_SEX` `AGE` `sido` `sigungu` `eupmyeondong` `stn` `ABATT_DATE` `BIRTH_YMD` `CATTLE_NO` `FARM_UNIQUE_NO` | |

→ 등급 판정 공식을 직접 적용할 수 없으므로 **기상·농장·혈통 피처 엔지니어링이 성능의 핵심**

---

## 현재 구현 현황 (`pipeline.py`) ✅

### 전체 흐름

```
STEP 0  혈통·농장 품질 통계 사전 계산  (train의 INSFAT/BACKFAT 활용)
  ↓
STEP 1  기상 월별 집계  (THI·폭염·한파 지수 생성)
  ↓
STEP 2  농장 피처 생성  (사육밀도·폐사율)
  ↓
STEP 3  make_features()  (train/test 공통 피처 결합)
  ↓
STEP 4  LightGBM 학습  (600K 계층 샘플링, 80/20 split)
  ↓
STEP 5  test 예측 → submissions/260418.csv
```

### 피처 목록 (실제 사용)

| 그룹 | 피처 | 설명 |
|------|------|------|
| 기본 | `WEIGHT`, `AGE`, `sex_code`, `sido_code` | 도체 기본 정보 |
| 날짜 | `abatt_year/month/season`, `birth_year/month/season` | 도축·출생 시기 |
| 기상(도축월) | `s_ta_mean`, `s_thi_mean`, `s_thi_max`, `s_rhm_mean`, `s_heat2d`, `s_heat3d`, `s_cold_d`, `s_rn_sum` | 도축 시점 기상 |
| 기상(출생월) | `b_ta_mean`, `b_thi_mean`, `b_thi_max`, `b_rhm_mean`, `b_heat2d`, `b_heat3d`, `b_cold_d`, `b_rn_sum` | 출생 시점 기상 |
| 농장 | `avg_cattle`, `density`, `death_cnt`, `death_rate` | 농장 규모·관리 |
| 농장 품질 | `farm_insfat`, `farm_backfat` | 농장 평균 육질·육량 수준 |
| 혈통 | `father_insfat`, `kpn_insfat` | 아비·KPN별 평균 근내지방 |

### 기상 피처 산식

```
THI = (1.8×평균기온 + 32) − (0.55 − 0.0055×습도) × (1.8×평균기온 − 26)
heat2d : THI ≥ 80인 일수 (중등도 열 스트레스)
heat3d : THI ≥ 85인 일수 (심각한 열 스트레스)
cold_d : 최저기온 < -5°C인 일수
```

### 타겟 클래스 (16개)

`LabelEncoder`로 아래 순서대로 0~15 인코딩:

```
0: 1++A   1: 1++B   2: 1++C
3: 1+A    4: 1+B    5: 1+C
6: 1A     7: 1B     8: 1C
9: 2A    10: 2B    11: 2C
12: 3A   13: 3B   14: 3C
15: 등외
```

결측값(`NaN`)은 `등외`로 대체 후 인코딩.

### 모델 설정

```python
LGBMClassifier(
    objective='multiclass', num_class=16,
    metric='multi_logloss',
    n_estimators=300, learning_rate=0.1,
    num_leaves=63, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    class_weight='balanced',
)
# 학습/검증 분할: stratified 80/20
# early stopping: 50라운드
# 결측 피처값: -999으로 대체
# 샘플링: 600K 계층 샘플 (클래스 비율 유지)
```

### 검증 결과

| 항목 | 값 |
|------|-----|
| 평가 지표 | Validation Macro F1 |
| 출력 파일 | `submissions/feature_importance.csv` (피처 중요도 상위 15개 포함) |

> 실행 시 콘솔에 `Validation Macro-F1` 및 `Best iteration`이 출력됩니다.

### 현재 한계

| 한계 | 개선 방향 |
|------|-----------|
| 기상을 출생월·도축월 **2개 시점**만 사용 | 사육 전 기간 집계 |
| 240만 행 중 600K(25%)만 학습 | 전량 학습 또는 더 큰 샘플 |
| 단일 LightGBM | CatBoost·XGBoost 앙상블 |
| 시간 분할 검증 미적용 | 시계열 기반 CV |

---

## 파일 구조

```
Animals/
├── data/
│   ├── hanwoo_train.csv
│   ├── hanwoo_weather.csv
│   ├── hanwoo_area.csv
│   ├── hanwoo_death.csv
│   ├── hanwoo_lineage.csv
│   └── test_hanwoo.csv
├── hanwoo_eda.ipynb          # EDA ✅
├── pipeline.py               # 전체 파이프라인 (피처 엔지니어링 + 학습 + 예측) ✅
└── submissions/              # 제출 파일 출력 디렉토리
```

---

## 단계별 진행 로드맵

### 1단계: 탐색적 데이터 분석 (EDA) ✅
**파일:** `hanwoo_eda.ipynb`

- 각 데이터셋 분포, 결측값, 등급 분포 확인
- 피처 간 상관관계 및 도메인 이해
- 데이터 조인 키 매칭률 검증

---

### 2단계: 피처 엔지니어링
**파일:** `feature_engineering.py`

| 피처 유형 | 데이터 출처 | 설명 |
|-----------|-----------|------|
| **기상** | `hanwoo_weather.csv` | 사육기간(출생~도축) 동안 기상 집계 — 대회 핵심 주제 |
| **농장** | `hanwoo_area.csv` + `hanwoo_death.csv` | 두당 면적, 사육두수 변화율, 폐사율 |
| **혈통** | `hanwoo_lineage.csv` | 아비·KPN별 LAST_GRADE 통계 (target encoding) |

기상 피처 설계:

```python
# 사육 기간 전체 통계
'ta_max_mean', 'ta_min_mean', 'diurnal_range_mean'  # 기온 평균·일교차
'heat_days',                                         # 폭염 일수 (ta_max >= 33°C)
'cold_days',                                         # 한파 일수 (ta_min <= -12°C)
'rn_day_sum', 'rhm_avg_mean'                         # 강수량, 습도

# 계절별 분리 집계
'summer_ta_max_mean', 'winter_ta_min_mean'

# 도축 직전 구간
'pre30d_ta_max_mean', 'pre30d_heat_days'             # 도축 전 30일 기상
```

농장·혈통 피처:

| 피처 | 출처 | 설명 |
|------|------|------|
| 두당 면적 | area | 축사 면적 / 사육두수 → 사육 밀도 |
| 사육두수 변화율 | area | 연도별 증감률 → 농장 경영 상태 |
| 농장 폐사율 | death | 농장별 폐사 건수 / 사육두수 → 농장 관리 수준 |
| 아비별 평균 등급 | lineage + train | 자손 등급 통계 → 유전적 영향 |
| KPN별 등급 통계 | lineage + train | 종축별 성과 |

---

### 3단계: 베이스라인 모델
**파일:** `model_baseline.py`

- LightGBM multi-class (16클래스 직접 예측)
- Stratified 5-Fold CV, 평가지표: **Macro F1**
- 시간 기반 분할 검증 (2023년 학습 → 2024~2025년 검증) 병행

```python
params = {
    'objective': 'multiclass',
    'num_class': 16,
    'metric': 'multi_logloss',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'min_child_samples': 50,
}
```

---

### 4단계: 모델 고도화
**파일:** `model_advanced.py`

**전략 A: 육질·육량 분리 예측**
```
모델A → 육질등급 (5클래스: 1++, 1+, 1, 2, 3)
모델B → 육량등급 (3클래스: A, B, C)
최종   → 육질 + 육량 결합 (15클래스 + 등외)
```

**전략 B: 모델 비교**

| 모델 | 특징 | 우선순위 |
|------|------|---------|
| LightGBM | 빠른 학습, 대용량 최적 | 1순위 |
| CatBoost | 범주형(sido·sigungu) 자동 처리, 순서형 타겟 지원 | 2순위 |
| XGBoost | 안정적, 과적합 제어 | 비교용 |

**전략 C: Optuna 하이퍼파라미터 튜닝**
- `num_leaves`, `learning_rate`, `feature_fraction`, `lambda_l1` 등

---

### 5단계: 앙상블
**파일:** `model_ensemble.py`

- LightGBM + CatBoost Weighted Average
- OOF(Out-of-Fold) 예측 기반 Stacking
- 모델별 Macro F1 가중치 부여

---

### 6단계: 제출 파일 생성
**파일:** `submit.py`

```python
test['LAST_GRADE'] = best_model.predict(test_features)
test.to_csv('260418.csv', index=False, encoding='utf-8-sig')
```

- UTF-8 BOM(`utf-8-sig`) 형식으로 저장
- 파일명: `260418.csv` (접수번호)
- `LAST_GRADE` 값: `1++A` ~ `3C`, `등외` 중 하나

---

## 검증 방법 및 지표

검증 데이터셋: `data/test_hanwoo.csv`

- 평가 지표: **Macro F1-Score**
- 제출 형식: UTF-8 또는 UTF-8 BOM
- 파일명: `260418.csv`

검증 전략:

| 방법 | 설명 |
|------|------|
| Stratified 5-Fold CV | 클래스 불균형 고려한 교차검증 |
| 시간 기반 분할 | 2023년 학습 → 2024~2025년 검증 (시계열 누수 방지) |
| Confusion Matrix | 등급별 혼동 패턴 분석 |
| Macro F1 per class | 등급별 F1 분석으로 약한 클래스 파악 |
