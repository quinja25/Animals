# 프로젝트 파이프라인

## 파이프라인 버전별 통합 비교 (Comparison Table) 📊

| 버전 | 스크립트 | 주요 특징 | 데이터 규모 | 기상 피처 | 검증 전략 | Macro F1 |
|:---:|:---|:---|:---:|:---|:---|:---|
| **v1** | `pipeline.py` | 베이스라인 모델 | 600K (샘플) | 출생·도축월 (2개) | Stratified 80/20 | 0.42 (추정) |
| **v2** | `pipeline v2.ipynb` | 성능 최적화 | 2.4M (전량) | 전 기간 + Lag | Stratified 5-CV | 0.4778 (OOF) |
| **v3** | `pipeline_v3.py` | 과적합 발생 | 2.4M (전량) | 전 기간 + Lag | Stratified 5-CV | 0.10 (LB) |
| **v4** | `pipeline_v4.py` | Unseen Farm 대응 | 2.4M (전량) | 전 기간 + Lag | GroupKFold | 0.20 (LB) |
| **v5** | `pipeline_v5.py` | 계층적 모델링 | 2.4M (전량) | 전 기간 + Lag | GroupKFold | **0.209 (LB)** |
| **v6** | `pipeline_v6.py` | 모듈화 & 누적 기상 | 2.4M (전량) | 6/12개월 Rolling | GroupKFold | *진행 중* |

---

## 파이프라인 Post-Mortem 및 향후 전략 (v7) 🚀

### 1. 이전 버전 분석 및 문제점 확인

| 항목 | 잘한 점 (Success) | 문제점 (Pain Points) |
|:---|:---|:---|
| **v1~v2** | 전량 학습 및 5-Fold CV로 높은 검증 점수 확보 | 농장(Farm) ID의 훈련/테스트 중복 0%를 간과하여 리더보드 점수 급락 |
| **v4~v5** | **GroupKFold** 도입으로 Unseen Farm에 대한 일반화 성능 확보 | 여전히 단일 모델의 한계로 인해 Macro F1이 0.2 수준에 머물러 있음 |
| **v6** | **모듈화(`DataProcessor`)** 및 **누적 기상 피처** 도입으로 데이터 연계 자동화 및 도메인 지식 반영 | 대량의 조인 연산으로 인한 메모리 관리 필요성 대두 |

### 2. 차기 모델링 전략 (`pipeline_v7`)

앙상블로 가기 전, **단일 모델(Single Model)**로서의 성능을 극대화하기 위한 전략입니다.

#### A. 모델 알고리즘 다양화 (CatBoost 도입)
*   **문제**: LightGBM은 범주형 변수(sido, sigungu, stn) 처리 시 수치형으로 변환되거나 리프 분할에 한계가 있음.
*   **해결**: 범주형 피처 처리에 특화된 **CatBoost**를 단일 모델로 사용. 특히 `stn`(관측소)이나 `sex_code` 같은 카테고리 피처의 관계를 더 정교하게 학습.

#### B. 피처 엔지니어링 심화 (Feature Engineering 2.0)
*   **상호작용 피처 확대**: `weight_per_age` 외에 `age_sex_interaction` (월령별 성별 성장 곡선), `seasonal_growth` (출생 계절과 도축 계절의 조합) 추가.
*   **Target Encoding 노이즈 제거**: 현재 단순 평균(Mean) 방식에서, 데이터 수가 적은 그룹(KPN 등)에 대해 **전역 평균을 가중 결합하는 Smoothing** 기법 적용.

#### C. 하이퍼파라미터 튜닝 최적화 (Optuna)
*   현재 고정된 `num_leaves=127` 대신, **Optuna**를 사용하여 단일 모델의 최적 파라미터(Depth, Learning Rate, Bagging fraction)를 탐색.

#### D. 데이터 샘플링 전략 수정
*   현재 전량 학습 시 클래스 불균형(3C, 등외 등 소수 클래스) 문제를 해결하기 위해 **SMOTE** 또는 **Class Weight**의 정밀 조정을 통해 Macro F1 최적화.

---

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

## 현재 구현 현황 (`pipelines/pipeline.py`) ✅

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

---

## 버전별 모델링 차이점 (v1 vs v2)

| 항목 | pipeline.py (v1) | pipeline v2.ipynb (v2) |
| :--- | :--- | :--- |
| **데이터 규모** | 600K 행 (계층적 샘플링) | **240만 행 (전량 학습)** |
| **기상 피처** | 출생월 & 도축월 (2개 시점) | **사육 전 기간 집계 + 도축 전 1·2·3개월 Lag** |
| **농장 피처** | INSFAT/BACKFAT 단순 평균 | **LOO(Leave-One-Out) Target Encoding + 등급 표준편차** |
| **상호작용 피처** | 없음 | **월령^2, 월령×성별, 월령×도축계절 등 추가** |
| **검증 전략** | 단일 80/20 Hold-out | **Stratified 5-Fold Cross Validation** |
| **모델 설정** | `num_leaves=63`, `n_estimators=300` | **`num_leaves=127`, `n_estimators=800`** |
| **성능 (Macro F1)** | 약 0.42 (추정) | **0.4778 (OOF 기준)** |

### 통합 피처 레퍼런스 (Standard Feature Set)

모델 버전과 관계없이 파이프라인에서 공통적으로 활용되는 핵심 피처 리스트입니다.

| 분류 | 피처명 | 설명 |
|:---:|:---|:---|
| **신체/물리** | `WEIGHT` | 도체중량 (kg) |
| | `AGE` | 도축 월령 (개월) |
| | `sex_code` | 성별 인코딩 (암:0, 수:1, 거세:2) |
| | `weight_per_age` | 월령 대비 체중 (발육 효율 지표) |
| **날짜/시간** | `abatt_year/month` | 도축 연도 및 월 |
| | `birth_year/month` | 출생 연도 및 월 |
| **유전 (KPN)** | `kpn_weight_sbv` | 도체중 표준화 육종가 |
| | `kpn_rea_sbv` | 등심단면적 표준화 육종가 |
| | `kpn_backfat_sbv` | 등지방두께 표준화 육종가 |
| | `kpn_insfat_sbv` | 근내지방도 표준화 육종가 |
| | `kpn_grade_avg` | 아비 소(KPN)별 과거 자손 평균 등급 점수 |
| **환경/지역** | `stn` | 기상 관측소 번호 |
| | `stn_grade_avg` | 지역(관측소)별 과거 평균 등급 점수 |
| | `s_ta`, `s_rn`, `s_rhm` | 도축월 평균기온, 강수량, 습도 |
| **농장 관리** | `density` | 사육 밀도 (면적 / 사육두수) |
| | `farm_size` | 농장 규모 (소:0, 중:1, 대:2) |
| | `death_cnt` | 농장별 과거 폐사 건수 |

---

## 타겟 클래스 (16개)

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

## 점수 급락 원인 분석 (v3 실적 부진) 🚨

현재 모델의 리더보드 점수(0.1)가 검증 점수(0.47)보다 현저히 낮은 이유는 다음과 같습니다.

| 원인 | 상세 내용 |
|------|-----------|
| **농장 정보 불일치** | 훈련 데이터와 테스트 데이터의 **농장(FARM_UNIQUE_NO) 중복도가 0%**임. |
| **과적합(Overfitting)** | 모델이 특정 농장 ID의 과거 성과에 과도하게 의존하도록 학습됨 (테스트 셋에선 무용지물). |
| **피처 누수(Leakage)** | 훈련 셋 농장의 Target Encoding이 테스트 셋의 '새로운 농장'에는 적용되지 않아 전역 평균값으로 대체됨. |

---

## 차기 모델링 전략 (`pipeline_v4`) 🚀

"보지 못한 농장"에 대해서도 강인한 예측력을 갖추는 것이 핵심입니다.

### 1. Target Encoding 대상 변경
테스트 셋과 중복도가 높은 피처를 중심으로 통계량을 재산출합니다.
*   **KPN_NO (중복도 98%)**: 종축별 등급 성과 통계 추가.
*   **FATHER_CATTLE_NO (중복도 98%)**: 아비 소별 등급 성과 통계 추가.
*   **stn (중복도 87%) / sido**: 지역별/관측소별 평균 등급 통계 추가.

### 2. 검증(Validation) 전략 수정
*   **GroupKFold 도입**: `FARM_UNIQUE_NO`를 기준으로 그룹화하여 교차 검증을 수행.
*   특정 농장의 데이터가 훈련과 검증에 동시에 들어가지 않도록 분리하여, 리더보드와 유사한 환경(Unseen Farm)에서 성능 측정.

### 3. 피처 선택 및 정제
*   **농장 ID 직접 사용 지양**: 농장 ID 자체를 피처로 쓰기보다, 농장의 특성(사육두수, 밀도, 폐사율)을 사용.
*   **Area 데이터 정제**: `hanwoo_area.csv`의 중복 농장 레코드를 제거하여 데이터 왜곡 방지.

---

## 버전별 모델링 차이점 (v4 vs v5) 🏆

v5는 v4의 한계를 극복하고 예측 성능을 극대화하기 위해 설계되었습니다.

| 항목 | pipeline_v4 (Baseline) | **pipeline_v5 (Advanced)** |
| :--- | :--- | :--- |
| **모델 구조** | 단일 16클래스 분류 (Multi-class) | **계층적 모델 (육질 6종 + 육량 3종 분리 학습)** |
| **데이터 매핑** | 단순 조인 (중복/공백 이슈 가능성) | **문자열 정규화(`strip`) + 개체별 1:1 매핑 정밀 보장** |
| **핵심 피처** | 개별 수치 피처 중심 | **상호작용 피처 추가 (`weight_per_age` 등 발육 효율 지표)** |
| **학습 안정성** | 16개 클래스 불균형에 취약 | **과업 분리를 통해 각 모델의 전문성 및 정확도 향상** |
| **검증 전략** | GroupKFold (Farm ID 기준) | **동일 (Unseen Farm에 대한 일반화 유지)** |
| 예상 성능 | Macro-F1 0.19~0.20 | **Macro-F1 0.35~0.45 (비약적 상승 기대)** |

---

## 버전별 모델링 차이점 (v5 vs v6) 🚀

v6는 유지보수성과 기상 데이터 활용도를 극대화하기 위해 설계된 최신 파이프라인입니다.

| 항목 | pipeline_v5 (Advanced) | **pipeline_v6 (Production)** |
| :--- | :--- | :--- |
| **코드 구조** | 단일 스크립트 (Monolithic) | **모듈화 (`DataProcessor` 클래스 분리)** |
| **기상 피처** | 도축월 단일 시점 중심 | **Lifecycle Rolling (6/12개월 누적 평균/폭염)** |
| **데이터 조인** | 코드 내 수동 조인 | **자동화된 전역 매핑 (Fuzzy matching 포함)** |
| **기상 지표** | 평균 기온, 강수량 | **THI(가축열스트레스지수) 추가** |
| **모델링** | LightGBM 계층적 분류 | **동일 (LGBM Hierarchical) + CatBoost 확장 가능** |
| **특징** | 빠른 실험용 | **대규모 데이터 처리 및 운영 최적화** |

### v6의 핵심 개선 사항

1. **`HanwooDataProcessor` 도입**: 
   - 6개 이상의 CSV/Excel 파일을 자동으로 조인하고 전처리.
   - `CATTLE_NO`, `FARM_UNIQUE_NO`, `KPN_NO` 등 핵심 키의 공백 제거 및 타입 정규화 자동화.
2. **누적 기상 영향력(Cumulative Impact)**:
   - 한우의 전 사육 기간(Lifecycle) 동안의 기상 영향을 수치화.
   - 최근 6개월, 12개월간의 평균 기온, 열스트레스(THI), 폭염 일수 등을 피처로 추가하여 "기상에 따른 등급 변화"라는 주제에 가장 부합.
3. **THI (Temperature Humidity Index)**:
   - `0.8 * T + (RH/100) * (T - 14.3) + 46.4` 산식을 적용하여 단순 기온보다 가축이 체감하는 환경 스트레스를 정밀하게 반영.
4. **상세 진행 로그 및 검증 통계**:
   - 각 Fold마다 육질(Quality) 및 육량(Yield) 모델의 **Macro-F1 Score**를 실시간 출력.
   - 전체 학습 완료 후 5-Fold 평균 및 표준편차를 제공하여 모델의 안정성 확인 가능.

---

# 신규 데이터 매핑 방법


## 컬럼명 설명

| 컬럼명 | 설명 | SIZE | 비식별화 SIZE |
|---------|---------|---------|---------|
| KPN_NO | 아비소의 정액번호 | VARCHAR(5) | 24자 |
| CATTLE_NO | 개체식별번호 | VARCHAR(15) | 44자 |
| MOTHER_CATTLE_NO | 어미소의 개체식별번호 | VARCHAR(15) | 44자 |
| FARM_UNIQUE_NO | 농장식별번호 | VARCHAR(6) | 24자 |

## 참고사항

- `FATHER_CATTLE_NO`, `MOTHER_ANIMAL_NO`, `F_GMOTHER_ANIMAL_NO` 등
- `CATTLE_NO` 또는 `ANIMAL_NO`로 된 컬럼은 15자리 개체식별번호이며, 비식별화 시 44자 문자열로 변환됨.

## 데이터 매핑

### KPN 매핑

- KPN 유전능력자료의 `KPN_NO`는 `hanwoo_lineage.csv`의 KPN 번호와 매핑
- 관계: `1:N`

### 개체번호 매핑

- `hanwoo_lineage.csv` 와 `hanwoo_train.csv` 는 `cattle_no` 기준으로 매핑
- 관계: 경우에 따라 `N:N`

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
