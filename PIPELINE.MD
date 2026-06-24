# 프로젝트 파이프라인

## 파이프라인 버전별 통합 비교 (Comparison Table) 📊

| 버전 | 스크립트 | 주요 특징 | 데이터 규모 | 기상 피처 | 검증 전략 | Macro F1 |
|:---:|:---|:---|:---:|:---|:---|:---|
| **v1** | `pipeline.py` | 베이스라인 모델 | 600K (샘플) | 출생·도축월 (2개) | Stratified 80/20 | 0.42 (추정) |
| **v2** | `pipeline v2.ipynb` | 성능 최적화 | 2.4M (전량) | 전 기간 + Lag | Stratified 5-CV | 0.4778 (OOF) |
| **v3** | `pipeline_v3.py` | 과적합 발생 | 2.4M (전량) | 전 기간 + Lag | Stratified 5-CV | 0.10 (LB) |
| **v4** | `pipeline_v4.py` | Unseen Farm 대응 | 2.4M (전량) | 전 기간 + Lag | GroupKFold | 0.20 (LB) |
| **v5** | `pipeline_v5.py` | 계층적 모델링 | 2.4M (전량) | 전 기간 + Lag | GroupKFold | **0.209 (LB)** |
| **v6** | `pipeline_v6.py` | 모듈화 & 누적 기상 | 2.4M (전량) | 6/12개월 Rolling | GroupKFold | **0.212 (LB)** |
| **v7** | `pipeline_v7.py` | CatBoost 단일 모델, 정합성 개선 | 2.4M (전량) | 3/6/12개월 Rolling | StratifiedGroupKFold | 별도 기록 없음 |
| **v8** | `pipeline_v8.py` | LightGBM + CatBoost 직접 앙상블 | 2.4M (전량) | 3/6/12개월 Rolling | StratifiedGroupKFold | 0.2126 (OOF) |
| **v9** | `pipeline_v9.py` | 직접 + 계층 앙상블, 클래스 보정 | 2.4M (전량) | 3/6/12개월 Rolling | StratifiedGroupKFold | **0.239 (LB)** / 0.2430 (보정 OOF) |
| **v10** | `pipeline_v10.py` | Fold-safe 사전확률, 교차 적합 보정 | 2.4M (전량) | 3/6/12개월 Rolling | StratifiedGroupKFold + Cross-fit | Fold 1 계층 F1 0.1722로 중단 |
| **v11** | `pipeline_v11.py` | 역할 분리, 순서형 육질, C 우선 육량 | 2.4M (전량) | 3/6/12개월 Rolling + 코호트 | StratifiedGroupKFold + Cross-fit | **0.2449 (교차 적합 OOF)** |
| **v12** | `pipeline_v12.py` | 확장 혈통, 완화된 C 가중치, B specialist 검증 | 2.4M (전량) | Rolling + 코호트 + 혈통 | StratifiedGroupKFold + Joint Cross-fit | **0.2460 (교차 적합 OOF)** |
| **v13** | `pipeline_v13.py` | Nested 도체형질 복원, Dual-C 앙상블 | 2.4M (전량) | 예측 도체형질 + 혈통 + 기상 | Nested Farm CV + Joint Cross-fit | **0.2477 OOF / 0.244 LB** |
| **v14** | `pipeline_v14.py` | 직접 경계 복원, B/C 전문 모델, 드리프트 저항 분기, 지역 보정 | 2.4M (전량) | 전체 경로 + 안정 피처 경로 | Nested Farm CV + Nested calibration | 전체 검증 예정 |

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

## 버전별 모델링 차이점 (v7 vs v8)

v7은 범주형 변수 처리에 강한 CatBoost 단일 모델을 사용했고, v8은 직접 16클래스 예측에서 LightGBM과 CatBoost의 앙상블 효과를 검증했습니다.

| 항목 | pipeline_v7 | pipeline_v8 |
|:---|:---|:---|
| **모델 구조** | CatBoost 단일 16클래스 분류 | LightGBM + CatBoost 16클래스 앙상블 |
| **검증 방식** | `FARM_UNIQUE_NO` 기준 StratifiedGroupKFold | 동일 |
| **범주형 처리** | CatBoost 네이티브 범주형 처리 | LightGBM category + CatBoost 네이티브 처리 |
| **그룹 정보** | 범주형 ID와 일반 수치 피처 중심 | KPN·관측소·지역별 Fold 내부 등장 횟수 추가 |
| **OOF 결과** | 별도 기록 없음 | LightGBM 0.2126, 최적 블렌드는 LightGBM 단독 |

v8에서 CatBoost 직접 모델은 LightGBM과 충분히 다른 오류를 만들었지만, 단순 확률 평균만으로는 점수가 개선되지 않았습니다. 따라서 다음 버전에서는 모델 종류를 늘리기보다 타겟 구조를 직접 활용하는 방향으로 전환했습니다.

---

## v9 성능 향상 분석

v9의 리더보드 Macro F1은 **0.239**로, 이전 0.21 수준에서 유의미하게 상승했습니다.

### 1. 직접 모델과 계층 모델의 결합

v9은 두 개의 서로 다른 예측 경로를 사용합니다.

```text
직접 경로: LightGBM → 최종 16개 등급 확률

계층 경로: CatBoost 육질 모델(6개 클래스)
          × CatBoost 육량 모델(3개 클래스)
          → 최종 16개 등급 결합 확률
```

| 모델 | 원본 OOF Macro F1 | 역할 |
|------|-----------------:|------|
| 직접 LightGBM | 약 0.2126 | 육질·육량의 결합 관계를 직접 학습 |
| 계층형 CatBoost | 약 0.2083 | 육질과 육량의 구조적 규칙을 분리 학습 |

개별 원본 점수만 보면 계층 모델이 낮지만, 두 모델의 최종 예측은 전체 행의 약 **34.1%**에서 달랐습니다. 이 오류 다양성 때문에 계층 모델은 최종 기하평균 블렌드에서 **70%**의 비중을 차지했습니다.

### 2. 클래스별 확률 스케일 보정

원본 모델은 표본이 많은 B등급을 과대 예측하고 C등급을 과소 예측했습니다. v9은 OOF 확률에 클래스별 계수를 곱한 뒤 다시 정규화하여 Macro F1 의사결정 경계를 조정했습니다.

```python
보정확률[class] = 원본확률[class] * class_scale[class]
```

이 보정과 직접·계층 확률의 기하평균 블렌드를 결합하여 OOF Macro F1이 **0.2430**까지 상승했습니다.

### 3. v9 평가 시 주의점

v9의 0.2430은 동일한 OOF 데이터에서 클래스 스케일과 블렌드 가중치를 선택한 뒤 다시 계산한 값입니다. 따라서 순수한 미관측 데이터 성능보다 낙관적일 수 있습니다. 실제 리더보드 점수 0.239와의 차이가 크지는 않았지만, v10에서는 이 후처리 선택 과정도 교차 적합합니다.

### 4. 클래스별 잔여 문제

| 구간 | 관찰된 문제 |
|------|-------------|
| B등급 | 여전히 예측 비중이 높고 인접 육질등급 간 혼동이 큼 |
| C등급 | 재현율이 낮으며 특히 `1+C`, `1C`, `2C`, `3C`가 취약 |
| 등외 | F1은 높지만 표본이 5,540건으로 매우 적어 Fold별 변동 가능 |
| 중간 육질등급 | `1+`와 `1`, `1`과 `2` 사이 경계가 불명확 |

---

## v10 모델 설계

v10은 v9의 성공 요소를 유지하면서, 원본 모델의 순위 품질과 검증 신뢰도를 동시에 개선하도록 설계했습니다.

### 1. Fold-safe KPN·관측소 사전확률

테스트셋의 농장은 훈련셋과 전혀 겹치지 않지만, KPN과 관측소는 일부 공유됩니다.

| 그룹 키 | 테스트 행 기준 훈련셋 중복률 |
|---------|----------------------------:|
| `KPN_NO` | 약 59.4% |
| `stn` | 약 92.6% |
| `FARM_UNIQUE_NO` | 0% |

각 Fold의 훈련 부분에서만 KPN·관측소별 육질 6개 클래스와 육량 3개 클래스의 사전확률을 계산합니다.

```text
훈련 행   → 같은 농장의 모든 타겟을 제외한 Leave-One-Farm-Out 통계
검증 행   → 해당 Fold 훈련 데이터만 사용
테스트 행 → 해당 Fold 훈련 데이터만 사용
희소 그룹 → 전역 클래스 비율로 Bayesian smoothing
```

농장 단위 검증에서는 단순 Leave-One-Out만 적용하면 훈련 행이 같은 농장의 다른 개체를 통해 농장별 타겟 분포를 간접적으로 볼 수 있습니다. 검증 Fold에서는 해당 농장 전체가 사라지므로 훈련·검증 피처 분포가 달라집니다. v10은 훈련 통계에서도 같은 농장의 모든 레코드를 제외하여 이 불일치를 방지합니다.

Smoothing 계수는 100이며, 표본이 적은 KPN 또는 관측소가 극단적인 확률을 갖지 않도록 전역 분포로 수축합니다.

### 2. 모델별 피처 역할 분리

v10도 v9과 동일하게 직접 LightGBM과 계층형 CatBoost를 사용합니다. 다만 KPN·관측소의 육질·육량 사전확률은 구성 요소 타겟과 직접 대응하는 계층형 CatBoost에만 제공합니다. 직접 LightGBM은 v9에서 검증된 기본 피처와 그룹 등장 횟수만 사용하여 16클래스 결합 관계를 안정적으로 학습합니다.

| 경로 | 사용 피처 | 목적 |
|------|-----------|------|
| 직접 LightGBM | v9 기본 피처 + Fold-safe 그룹 횟수 | 검증된 직접 16클래스 성능 유지 |
| 계층형 CatBoost | 직접 경로 피처 + KPN·관측소 육질/육량 사전확률 | 구성 요소별 과거 분포 활용 |

### 3. 교차 적합 후처리

5개 모델 Fold 중 하나를 후처리 검증 Fold로 두고, 나머지 4개 Fold의 OOF 예측으로만 다음 값을 학습합니다.

- 직접 모델 클래스별 스케일
- 계층 모델 클래스별 스케일
- 산술평균 또는 기하평균 방식
- 직접·계층 모델 블렌드 가중치

이 과정을 5회 반복하여 모든 OOF 행이 자신을 보지 않은 보정기로 처리되도록 합니다.

```text
OOF Fold 1 ← Fold 2~5로 보정기 학습
OOF Fold 2 ← Fold 1, 3~5로 보정기 학습
...
OOF Fold 5 ← Fold 1~4로 보정기 학습
```

테스트 예측은 5개 보정기의 출력 확률을 평균합니다. 최종 보고 점수는 `Cross-fitted calibrated OOF`이며, v9의 동일 OOF 재최적화 점수보다 보수적이지만 리더보드 일반화 성능을 더 신뢰할 수 있습니다.

### 4. v10 출력 파일

| 파일 | 내용 |
|------|------|
| `submissions/v10/260418.csv` | 최종 제출 파일 |
| `fold_metrics.csv` | 직접·계층 모델의 Fold별 원본 Macro F1 |
| `calibration_folds.csv` | 보정 학습 점수, 미관측 Fold 점수, 블렌드 설정 |
| `class_scales.csv` | Fold 및 클래스별 확률 스케일 |
| `oof_class_metrics.csv` | 교차 적합 보정 후 클래스별 지표 |
| `oof_predictions.npz` | 원본·계층·보정 OOF 확률과 Fold 번호 |
| `feature_importance.csv` | 직접 LightGBM 피처 중요도 |

### 5. 실행 방법

```powershell
python pipelines/pipeline_v10.py
```

GPU CatBoost를 사용하려면 실행 전에 환경변수를 설정합니다.

```powershell
$env:CATBOOST_TASK_TYPE="GPU"
python pipelines/pipeline_v10.py
```

### 6. v10 평가 기준

v10은 아직 전체 학습이 완료되지 않았으므로 성능 수치를 확정하지 않습니다. 실행 후 다음 순서로 결과를 판단합니다.

1. `fold_metrics.csv`에서 Fold별 원본 점수 편차 확인
2. `calibration_folds.csv`에서 보정 학습 점수와 미관측 Fold 점수 차이 확인
3. 교차 적합 보정 OOF가 v9의 원본 OOF 약 0.21보다 개선되는지 확인
4. 클래스별 C등급 재현율 개선 여부 확인
5. 보정기별 블렌드 가중치와 클래스 스케일의 안정성 확인

---

## v11 모델 설계

v11은 동일한 데이터를 세 알고리즘에 반복 입력하는 방식이 아니라, 같은 Fold의 행을 유지하면서 타겟과 피처 역할을 분리합니다.

| 알고리즘 | 타겟 | 주요 피처 관점 |
|----------|------|----------------|
| LightGBM | 최종 16클래스 | v9에서 검증된 수치·기상·농장·그룹 횟수 |
| CatBoost | 육질 6클래스 및 등외 | 원본 KPN·관측소·지역 범주와 유전·환경 정보 |
| Ordinal LightGBM | 육질 순서 경계 4개 | `1++`, `1+ 이상`, `1 이상`, `2 이상` 누적 경계 |
| XGBoost | `C vs non-C`, 이후 `A vs B` | 수치형 체중·성별·월령 코호트와 육량 관련 유전 정보 |

세 모델은 동일한 농장 기준 Fold를 사용하므로 OOF 확률의 행 정렬과 검증 조건은 동일합니다. 그러나 서로 다른 타겟과 피처 공간을 사용하여 오류 상관을 낮춥니다.

### C vs non-C가 먼저인 이유

육량등급 A·B·C는 독립적인 명목 클래스가 아니라 하나의 잠재 육량지수에서 경계값으로 나뉘는 순서형 클래스입니다. v9 직접 모델에서 C등급 재현율은 대부분 3~8% 수준이었습니다. 세 클래스를 한 번에 학습하면 표본이 많은 B가 손실을 지배하고 C 경계를 충분히 학습하지 못할 수 있습니다.

v11은 다음 조건부 확률로 육량 확률을 복원합니다.

```text
1단계: p(C) = P(C vs non-C)
2단계: p(A | non-C) = P(A vs B | non-C)

p(A) = (1 - p(C)) × p(A | non-C)
p(B) = (1 - p(C)) × (1 - p(A | non-C))
p(C) = p(C)
```

첫 모델은 취약한 C 하단 경계에 집중하고, 두 번째 모델은 C를 제거한 뒤 A와 B의 상단 경계만 학습합니다.

### 모델별 신규 피처

XGBoost 육량 모델에는 다음과 같은 타겟 비사용 코호트 피처를 추가합니다.

- 동일 성별·2개월 월령 구간의 평균 및 중앙값 대비 체중 잔차
- 동일 코호트 내 체중 z-score와 IQR 위치
- 입력 월령과 출생·도축일 계산 월령의 차이
- 사육 개월당 폭염 노출량
- KPN 및 KPN 육종가 연결 여부
- KPN 육종가 결측 개수와 기상 피처 결측 개수

CatBoost는 원본 고카디널리티 범주를 유지하고, LightGBM은 v9 기준 피처를 유지합니다. 수동 타겟 사전확률은 v10 실험에서 직접 모델 성능을 떨어뜨렸으므로 v11 기본 구성에서 제외합니다.

### 단계별 성능 게이트

각 Fold에서 다음 지표를 `component_metrics.csv`에 저장합니다.

1. LightGBM 직접 16클래스 Macro F1
2. CatBoost 육질 Macro F1
3. Ordinal LightGBM 육질 Macro F1
4. XGBoost 육량 Macro F1 및 C등급 재현율
5. CatBoost/Ordinal 육질 × XGBoost 육량 계층 확률의 최종 Macro F1
6. 직접·계층 예측 불일치율
7. 진단용 최적 블렌드 Macro F1과 직접 모델 가중치

계층 경로가 도움이 되지 않으면 최종 블렌드 탐색에서 직접 모델 가중치 1.0을 선택할 수 있습니다. 최종 보고 점수는 전체 OOF에서 재사용한 가중치가 아니라, 나머지 네 Fold에서 보정기와 가중치를 학습한 교차 적합 OOF입니다.

CatBoost 육질과 Ordinal LightGBM 육질의 가중치도 별도로 교차 적합합니다. Ordinal 모델이 도움이 되지 않으면 `ordinal_weight=0`이 선택되어 기존 CatBoost 육질 경로로 자동 복귀합니다.

### 실행 순서

```powershell
$env:CATBOOST_TASK_TYPE="GPU"
$env:XGBOOST_DEVICE="cuda"

# LightGBM 기준선만 검증
$env:V11_SMOKE_STAGE="lgb"
python pipelines/pipeline_v11.py

# 첫 Fold의 세 구성요소 검증
$env:V11_SMOKE_STAGE="components"
python pipelines/pipeline_v11.py

# 전체 5-Fold
$env:V11_SMOKE_STAGE=""
python pipelines/pipeline_v11.py
```

첫 Fold 구성요소 검증에서 C 재현율과 계층 블렌드가 개선되지 않으면 전체 5-Fold 실행 전에 XGBoost 피처와 가중치를 조정합니다.

---

## v10 실패 분석 및 v11 최종 결과

### v10 중단 이유

v10 Fold 1의 직접 LightGBM은 Macro F1 `0.2131`로 정상적인 v9 기준선을 재현했습니다. 그러나 수동 KPN·관측소 육질/육량 사전확률을 받은 계층형 CatBoost는 최종 Macro F1 `0.1722`에 그쳤습니다.

| 경로 | v9 원본 OOF | v10 Fold 1 | 판단 |
|------|------------:|------------:|------|
| 직접 LightGBM | 0.2126 | 0.2131 | 정상 |
| 계층형 경로 | 0.2083 | 0.1722 | 큰 폭 하락 |

Leave-One-Farm-Out 처리만으로도 수동 타겟 사전확률의 분포 불일치와 과도한 의존을 완전히 해결하지 못했습니다. 따라서 v10 전체 실행을 중단하고, v11에서는 수동 타겟 사전확률을 기본 피처에서 제거했습니다.

### C 재현율의 정의

C 재현율은 실제 육량등급이 C인 개체 중 모델이 C로 예측한 비율입니다.

```text
C recall = true-positive C / all true C
```

예를 들어 실제 C 개체가 1,000마리일 때 재현율 0.108은 약 108마리, 재현율 0.511은 약 511마리를 C로 식별했다는 의미입니다. C 육량등급은 최종 클래스 `1++C`, `1+C`, `1C`, `2C`, `3C`에 공통으로 포함됩니다.

### C에 우선 집중한 이유와 A/B 처리

v9의 육량 구성요소 성능은 다음과 같았습니다.

| 육량등급 | Precision | Recall | F1 |
|----------|----------:|-------:|---:|
| A | 0.555 | 0.331 | 0.415 |
| B | 0.542 | 0.834 | 0.657 |
| C | 0.539 | 0.108 | 0.180 |

B는 이미 높은 재현율을 보였고 A는 중간 수준이었지만, C는 거의 선택되지 않았습니다. Macro F1은 세 클래스에 동일한 비중을 부여하므로 가장 약한 C 경계를 개선할 잠재 수익이 가장 컸습니다.

A와 B를 무시한 것은 아닙니다. v11은 먼저 `C vs non-C`를 학습하고, C가 아닌 A/B 표본만 사용해 `A vs B`를 학습합니다. 두 모델의 출력으로 A·B·C 전체 확률을 복원합니다.

### v11 5-Fold 구성요소 결과

아래 값은 최종 보정 전 구성요소 평균입니다.

| 지표 | v9 기준 | v11 5-Fold 평균 | 변화 |
|------|--------:|-----------------:|-----:|
| 직접 16클래스 F1 | 0.2126 | 0.2126 | 유지 |
| CatBoost 육질 F1 | 0.4626 | 0.4620 | 유사 |
| Ordinal 육질 F1 | 없음 | 0.4234 | 신규 |
| 육량 F1 | 0.4170 | **0.4786** | +0.0616 |
| C 재현율 | 0.1078 | **0.5109** | +0.4031 |
| 원본 계층 F1 | 0.2083 | **약 0.2336** | +0.0253 |
| 진단용 Fold 내 블렌드 F1 | 없음 | **0.2369** | 참고용 |

진단용 블렌드는 각 검증 Fold에서 직접 가중치를 선택한 낙관적 수치이므로 최종 성능으로 보고하지 않습니다. 최종 판단은 다른 네 Fold에서 가중치와 클래스 스케일을 학습한 교차 적합 OOF로 수행합니다.

### v11 최종 교차 적합 결과

| 지표 | v9 | v11 | 변화 |
|------|---:|----:|-----:|
| 원본 직접 OOF | 0.2126 | 0.2126 | 동일 |
| 원본 계층 OOF | 0.2083 | **0.2335** | +0.0252 |
| 최종 보정 OOF | 0.2430 | **0.2449** | +0.0019 |

v9의 `0.2430`은 전체 OOF에서 후처리 계수를 선택한 낙관적 수치인 반면, v11의 `0.2449`는 각 held-out Fold를 보지 않은 보정기로 계산했습니다. 따라서 절대 차이는 작지만 v11 결과의 검증 신뢰도가 더 높습니다.

총 실행 시간은 RTX 3070 기준 `221.3분`이었습니다.

### 교차 적합 가중치 안정성

| 항목 | 선택 결과 |
|------|-----------|
| CatBoost/Ordinal 육질 | 모든 Fold에서 Ordinal 가중치 0.6 |
| 직접/계층 결합 방식 | 모든 Fold에서 기하평균 |
| 직접 LightGBM 가중치 | 0.25~0.30 |
| 계층 경로 가중치 | 0.70~0.75 |
| held-out 최종 F1 범위 | 0.2411~0.2462 |

Fold 3의 held-out F1이 `0.2411`로 상대적으로 낮았지만 나머지 네 Fold는 `0.2452~0.2462` 범위였습니다. 가중치 선택이 매우 일관되므로 앙상블 구조는 안정적입니다.

### v11 육량 클래스별 결과

| 육량등급 | Precision | Recall | F1 | v9 F1 |
|----------|----------:|-------:|---:|------:|
| A | 0.4993 | 0.4875 | **0.4933** | 0.4148 |
| B | 0.5705 | 0.4836 | **0.5235** | 0.6566 |
| C | 0.3549 | 0.5109 | **0.4189** | 0.1796 |

C Hurdle은 C F1을 크게 높였고 A F1도 개선했습니다. 반면 B 재현율과 F1은 하락했습니다. Macro F1은 전체적으로 상승했지만, v12에서는 C의 개선을 유지하면서 B 성능을 일부 회복하는 것이 가장 명확한 과제입니다.

### 최종 16클래스 변화 요약

- 개선 폭이 큰 클래스: `1+C` +0.0297, `1++C` +0.0156, `1A` +0.0122, `3C` +0.0116
- 하락 폭이 큰 클래스: `1B` -0.0228, `1++A` -0.0169, `2A` -0.0156, `1+B` -0.0089
- `등외` F1: 0.6989 → 0.7031

v11의 이득은 주로 C 계열과 일부 A 계열에서 발생했고, 일부 B/A 클래스가 비용을 지불했습니다.

### v12 이후를 위한 유지 원칙

1. **기준선 보존**: v9 직접 LightGBM 피처와 고정 Farm Fold를 항상 재현합니다.
2. **모델별 역할 분리**: 알고리즘 수보다 타겟·피처·오류 구조의 다양성을 우선합니다.
3. **C Hurdle 유지 여부는 정밀도와 함께 판단**: v11 C precision은 0.3549, recall은 0.5109입니다. v12는 C F1을 유지하면서 B recall을 회복해야 합니다.
4. **수동 타겟 인코딩 기본 제외**: 사용하려면 Outer/Inner Farm GroupKFold와 smoothing을 포함한 별도 ablation으로 검증합니다.
5. **Ordinal 품질 모델은 0 가중치 허용**: 교차 적합에서 개선되지 않으면 자동 제외합니다.
6. **모든 후처리 교차 적합**: 전체 OOF에서 선택한 클래스 스케일이나 블렌드 점수를 최종 검증 점수로 사용하지 않습니다.
7. **증분 성능 기록**: 직접, 육질, 육량, 계층, 블렌드 순서로 Fold별 성능을 남깁니다.
8. **최종 OOF와 LB 구분**: OOF, 교차 적합 OOF, 리더보드 점수를 별도 열로 기록합니다.

### v12 후보 실험

- XGBoost C 모델의 `scale_pos_weight` 또는 C 확률 보정으로 C/B 경계 재조정
- B recall 0.4836을 회복하면서 C F1 0.4189를 유지하는 조건부 확률 보정
- Ordinal 품질 가중치의 Fold 간 변동 분석
- CatBoost/Ordinal 품질 결합을 클래스별로 다르게 적용할 필요가 있는지 검토
- 동일 OOF를 사용한 단순 모델 추가보다 현재 세 경로의 확률 보정 개선 우선

---

## v12 모델 설계

v11의 리더보드 개선 폭은 제한적이었지만 OOF 분석에서 두 가지 명확한 다음 단계가 확인됐습니다.

1. C Hurdle이 C를 크게 개선했지만 B F1이 0.6566에서 0.5235로 하락했습니다.
2. 기존 파이프라인은 테스트에서 높은 재사용률을 가진 부·조부모 혈통 식별자를 사용하지 않았습니다.

v12는 단순 모델 추가가 아니라 **새로운 유전 신호**와 **B/C 경계 복원**에 집중합니다.

### 1. 확장 혈통 피처

`hanwoo_lineage_0612.csv`에서 사용하지 않던 혈통 식별자의 테스트 적용 범위를 측정했습니다.

| 혈통 피처 | 테스트 결측률 | 테스트 행 중 훈련에서 관측 | 값이 있을 때 관측률 |
|-----------|------------:|--------------------------:|--------------------:|
| KPN | 40.6% | 59.4% | 99.99% |
| 부 개체번호 | 37.5% | 62.5% | 99.99% |
| 부계 조모 | 37.5% | 62.5% | 99.99% |
| 부계 조부 | 37.5% | 62.5% | 거의 100% |
| 모계 조부 | 37.5% | 62.5% | 99.99% |
| 모 개체번호 | 37.5% | 19.3% | 30.9% |
| 모계 조모 | 37.5% | 33.0% | 52.8% |

부·부계 조부모·모계 조부는 값이 존재할 경우 거의 모두 훈련에서 관측됩니다. 이 네 저·중카디널리티 식별자는 CatBoost 원본 범주형 피처로 사용합니다. 모 개체번호와 모계 조모는 고카디널리티이며 재사용률이 낮아 원본 범주로 사용하지 않고 Fold 내부 등장 횟수만 사용합니다.

직접 LightGBM은 v11 기준선을 보존하기 위해 원본 확장 혈통 ID를 받지 않습니다. XGBoost는 혈통 ID의 Fold-safe 등장 횟수만 사용합니다.

### 2. C 모델 가중치 완화

v11 C 분류기의 양성 가중치는 클래스 비율의 제곱근으로 약 2.0이었습니다. v12는 지수 `0.25`를 사용하여 약 1.4 수준의 완만한 가중치를 적용합니다. 목적은 C 순위 신호를 유지하면서 B로 돌아가야 할 경계 표본을 줄이는 것입니다.

### 3. B vs non-B specialist

v12는 동일한 XGBoost 수치 피처로 별도의 `B vs non-B` 모델을 학습합니다. 이 모델은 최종 앙상블에 직접 추가하지 않고 기존 Hurdle A/B/C 확률의 B 성분을 보정합니다.

```text
기존 Hurdle 확률: P(A), P(B), P(C)
B specialist:     P(B vs non-B)

보정 P(B) = (1-w) × Hurdle P(B) + w × Specialist P(B)
나머지 확률 = 기존 A:C 비율을 유지하며 1-P(B)에 배분
```

`w=0`을 후보에 포함하므로 B specialist가 도움이 되지 않으면 자동 제외됩니다.

### 4. B/C/육질 공동 교차 적합

각 held-out Fold에 대해 나머지 Fold의 OOF 일부만 사용하여 다음 값을 함께 탐색합니다.

- Ordinal 육질 가중치: 0.0~1.0
- B specialist 가중치: 0.0, 0.25, 0.5, 0.75
- C 확률 multiplier: 0.8, 0.9, 1.0

선택된 조합을 held-out Fold와 테스트에 적용한 뒤 기존 클래스 스케일 및 직접/계층 기하평균 보정을 다시 교차 적합합니다. B/C 보정과 최종 클래스 보정을 분리해서 최적화할 때 발생한 충돌을 줄이기 위한 구조입니다.

### 5. 출력 캐시

v12는 `test_components.npz`를 추가로 저장합니다.

```text
direct
quality
ordinal_quality
yield_probability
b_specialist
hierarchical
calibrated
```

향후 v13에서는 전체 모델을 다시 학습하지 않고 테스트 확률 후처리를 재현할 수 있습니다.

### 6. v12 성공 기준

| 지표 | v11 | v12 목표 |
|------|----:|---------:|
| 원본 직접 OOF | 0.2126 | 기준선 유지 |
| 원본 계층 OOF | 0.2335 | 0.235 이상 |
| 교차 적합 최종 OOF | 0.2449 | **0.247 이상** |
| B F1 | 0.5235 | 0.56 이상 |
| C F1 | 0.4189 | 0.39 이상 유지 |

### 7. 실행 방법

첫 Fold 구성요소 검증:

```powershell
$env:CATBOOST_TASK_TYPE="GPU"
$env:XGBOOST_DEVICE="cuda"
$env:V12_SMOKE_STAGE="components"
python pipelines/pipeline_v12.py
```

전체 실행:

```powershell
$env:V12_SMOKE_STAGE=""
$env:SAVE_OOF="1"
python pipelines/pipeline_v12.py
```

확장 혈통이 CatBoost 메모리 사용량을 증가시키므로 첫 Fold smoke test에서 GPU 메모리와 `diagnostic_best_blend_f1`을 확인한 뒤 전체 실행합니다.

### 8. v12 최종 결과

| 지표 | v11 | v12 | 변화 |
|------|----:|----:|-----:|
| 원본 직접 OOF | 0.2126 | 0.2126 | 동일 |
| 원본 계층 OOF | 0.2335 | **0.2354** | +0.0019 |
| 교차 적합 최종 OOF | 0.2449 | **0.2460** | +0.0011 |
| 실행 시간 | 221.3분 | 213.7분 | -7.6분 |

v12는 목표 `0.247`에는 도달하지 못했지만 모든 주요 OOF 지표를 개선했습니다. 직접 LightGBM이 동일하므로 개선은 확장 혈통을 사용한 육질 경로와 완화된 C 학습 가중치를 사용한 육량 경로에서 발생했습니다.

### 9. v12 육량 결과

| 육량등급 | v11 F1 | v12 Precision | v12 Recall | v12 F1 |
|----------|--------:|--------------:|-----------:|-------:|
| A | 0.4933 | 0.5031 | 0.4944 | **0.4987** |
| B | 0.5235 | 0.5624 | 0.6364 | **0.5971** |
| C | 0.4189 | 0.4350 | 0.2952 | **0.3517** |
| Macro | 0.4786 | — | — | **0.4825** |

B F1은 크게 회복됐고 A도 소폭 개선됐습니다. C F1은 하락했지만 전체 육량 Macro F1은 상승했습니다. v13에서는 현재 두 극단 사이에서 C F1을 일부 회복하면서 B F1을 유지하는 것이 과제입니다.

### 10. B specialist 및 공동 보정 판단

교차 적합 선택 결과는 모든 Fold에서 다음과 같았습니다.

```text
B specialist weight = 0.0
C multiplier = 1.0
```

즉 별도 B specialist와 후처리 C multiplier는 held-out 성능에 기여하지 않아 자동 제외됐습니다. B 회복은 specialist가 아니라 C 모델의 학습 가중치를 약 2.0에서 약 1.4로 낮춘 효과입니다. v13에서는 B specialist 학습 비용을 제거해야 합니다.

### 11. 혈통 및 육질 결합 결과

CatBoost 육질 F1 평균은 v11 `0.4620`에서 v12 약 `0.4643`으로 상승했습니다. 교차 적합 Ordinal 가중치는 v11의 일관된 0.6에서 v12의 `0.0~0.4`로 감소했습니다. 확장 혈통을 받은 CatBoost 육질 모델이 강화되면서 Ordinal 경로 의존도가 줄어든 것으로 해석됩니다.

최종 보정은 모든 Fold에서 기하평균을 선택했고 직접 LightGBM 가중치는 `0.15~0.25`였습니다. v11의 `0.25~0.30`보다 계층 경로 비중이 더 높아졌습니다.

### 12. v13 인계 사항

1. B specialist 제거: 모든 Fold에서 가중치 0으로 선택됨.
2. 확장 혈통 유지: CatBoost 육질 평균 F1 개선.
3. C balance power `0.25` 유지 후 확률 온도만 미세 탐색.
4. C/B 후처리는 최종 클래스 보정과 반드시 공동 검증.
5. CatBoost feature importance를 저장하여 어떤 혈통 ID가 실제 기여했는지 확인.
6. 목표는 B F1 `0.58+`, C F1 `0.38+`, 최종 교차 적합 OOF `0.248+`.

---

## v13 모델 설계

v11과 v12의 리더보드 점수가 동일하여, B/C 후처리만으로는 큰 성능 향상을 만들기 어렵다는 결론을 내렸습니다. v13은 테스트에 없는 실제 등급판정 중간 형질을 훈련 데이터에서 복원하는 다단계 모델입니다.

### 1. 복원 대상

하나의 GPU CatBoost MultiRMSE 모델이 다음 7개 형질을 동시에 예측합니다.

| 영역 | 형질 |
|------|------|
| 육량 | `BACKFAT`, `REA` |
| 육질 | `INSFAT`, `YUKSAK`, `FATSAK`, `TISSUE`, `GROWTH` |

각 형질은 단위가 다르므로 Inner Fold 훈련 데이터의 평균과 표준편차로 정규화한 후 동일한 MultiRMSE 손실에서 학습합니다. 예측 후 원래 단위로 복원합니다.

### 2. Nested Farm CV

실제 형질을 훈련 행에 그대로 제공하면 테스트에서는 예측값만 존재하는 심각한 누수가 발생합니다. v13은 각 Outer Farm Fold 안에서 다시 2개의 Inner Farm Fold를 생성합니다.

```text
Outer training farms
  ├─ Inner model 1: 절반 Farm 학습 → 다른 절반 proxy 예측
  └─ Inner model 2: 반대 절반 학습 → 첫 절반 proxy 예측

Outer validation/test
  └─ 두 Inner 모델의 예측 평균
```

따라서 최종 등급 모델이 보는 모든 훈련 proxy도 해당 개체와 농장을 보지 않은 모델의 예측입니다. Outer validation과 test 행은 두 Inner 모델 중 하나에 결정적으로 배정해 단일 모델 예측을 사용합니다. 두 모델 평균은 검증/test proxy만 과도하게 평활화하여 학습 OOF proxy와 분포가 달라질 수 있으므로 사용하지 않습니다.

### 3. 파생 도메인 피처

복원된 `BACKFAT`, `REA`, 실제 `WEIGHT`, 성별을 공식 육량지수 산식에 대입하여 다음 피처를 생성합니다.

- `proxy_windex`
- A 경계까지 거리
- C 경계까지 거리
- 예측 A/B/C 신호

육질 측면에서는 복원된 근내지방도·육색·지방색·조직감을 공식 구간에 매핑합니다.

- 항목별 예측 육질 rank
- 항목 중 최악의 예측 rank
- 고성숙도 예측 flag

원본 7개 연속 예측값도 함께 사용하여 최종 트리가 경계 오차를 자체 보정할 수 있게 합니다.

### 4. Dual-C 앙상블

v11의 aggressive-C와 v12의 balanced-C OOF 확률을 50:50으로 섞었을 때 육량 Macro F1은 `0.4880`, 원본 계층 F1은 `0.2371`로 두 버전보다 높았습니다.

v13은 B specialist를 제거하고 두 C 모델을 학습합니다.

| C 모델 | balance power | 역할 |
|--------|--------------:|------|
| Balanced C | 0.25 | B 정밀도와 재현율 보존 |
| Aggressive C | 0.50 | C 재현율 보존 |

두 C 확률의 가중치는 다른 Outer Fold의 OOF에서 `0.0~1.0` 범위로 교차 적합합니다. A/B 조건부 모델은 공유합니다.

### 5. Proxy 적용 범위

복원 피처는 세 경로에 모두 제공됩니다.

- 직접 LightGBM: 최종 16클래스의 물리적 중간 신호
- CatBoost/Ordinal 육질: 복원된 품질 형질과 규칙 거리
- XGBoost 육량: 복원된 backfat·REA·yield index

직접 LightGBM은 v12의 기존 피처 집합을 그대로 사용하며 proxy 피처를 받지 않습니다. 따라서 재구성 실험이 실패해도 direct 기준선이 훼손되지 않습니다. Proxy 피처는 CatBoost 품질 모델과 XGBoost 육량 모델에만 추가됩니다. Fold별 `proxy_metrics.csv`에 각 형질의 MAE와 R²를 저장합니다.

### 6. v13 실행 게이트

첫 Fold에서 다음을 확인합니다.

| 지표 | 최소 기준 |
|------|----------:|
| BACKFAT/REA R² | 양수, 가능하면 0.10 이상 |
| INSFAT R² | 양수 |
| 직접 F1 | v12 0.213 부근 이상 |
| 육량 F1 | 0.488 이상 |
| 진단용 계층 F1 | 0.238 이상 |
| 진단용 최종 블렌드 | 0.240 이상 |

Proxy R²가 음수이거나 직접/계층 F1이 v12보다 낮으면 전체 5-Fold를 실행하지 않습니다.

### 7. 실행 방법

```powershell
cd C:\Users\jaeyo\Projects\Animals\pipelines
conda activate hanwoo
$env:CATBOOST_TASK_TYPE="GPU"
$env:XGBOOST_DEVICE="cuda"
$env:V13_SMOKE_STAGE="components"
$env:SAVE_OOF="1"
python pipeline_v13.py
```

Nested proxy 모델 때문에 v13은 v12보다 오래 실행됩니다. 첫 Fold smoke 결과를 검토한 뒤 `V13_SMOKE_STAGE=""`로 전체 실행합니다.

### 8. v13 최종 결과

| 지표 | v12 | v13 | 변화 |
|------|----:|----:|----:|
| 원본 직접 OOF | 0.2126 | 0.2126 | 0.0000 |
| 원본 계층 OOF | 0.2354 | 0.2381 | +0.0027 |
| 교차 적합 보정 OOF | 0.2460 | 0.2477 | +0.0017 |
| 리더보드 | 약 0.241 | **0.244** | 개선 |

직접 LightGBM 기준선을 보존하면서 proxy를 받은 육질·육량 경로만 개선됐습니다. GROWTH와 REA 복원력은 높았지만 YUKSAK과 FATSAK 복원력은 낮아 최종 상승폭은 제한됐습니다.

### 9. v14 전 세 가지 진단

#### 교차 적합 스태커

v13의 direct, hierarchy, quality, ordinal, yield 확률을 입력으로 사용한 교차 적합 선형 스태커는 `0.2024`에 그쳤습니다. v13과 과거 버전의 단순 교차 적합 블렌드도 v11 결합 `0.2480`, v12 결합 `0.2481`로 v13 대비 약 `0.0004`만 개선됐습니다. 기존 모델들의 오류 상관이 높으므로 v14에서는 일반 스태킹을 사용하지 않습니다.

#### Train/Test 적대적 검증

| 피처 집합 | AUC |
|----------|----:|
| v13 전체 test-safe 피처 | 0.9794 |
| 동물·시간 핵심 피처만 사용 | 0.5409 |

분포 차이는 주로 `stn`, `sigungu`, `death_cnt`, `density`, `farm_size`와 지역 기상에서 발생했습니다. Train Farm 65,941개와 Test Farm 12,515개 사이의 공통 Farm은 0개입니다.

#### 시간·농장 검증 감사

2023~2025년 train/test 비율과 농장별 행 수 분포는 거의 같았습니다. v13 연도별 OOF는 2023년 `0.2292`, 2024년 `0.2549`, 2025년 `0.2543`이었습니다. 관측소 분포 재가중 OOF는 `0.2477`, 시군구 재가중 OOF는 `0.2487`로 원래 OOF와 거의 같습니다.

따라서 높은 적대적 AUC만으로 지역 피처를 모두 제거하면 안 됩니다. v14는 전체 모델을 유지하면서 지역·농장·기상 피처를 제외한 드리프트 저항 분기를 추가하고, 교차 적합에서 기여가 확인될 때만 결합합니다.

### 10. v14 인계 원칙

1. 일반 스태커는 제외합니다.
2. C-vs-rest가 놓치는 B/C 경계를 별도 전문 모델로 학습합니다.
3. 도체형질 회귀값뿐 아니라 공식 경계 거리를 직접 복원합니다.
4. 드리프트 저항 분기는 지역·농장 프로필·기상 피처를 제거합니다.
5. 지역 보정은 다른 Fold에서만 학습하고 표본이 적은 지역은 전역 보정으로 축소합니다.
6. 신규 분기는 가중치 0을 허용하며 v13 원본 경로를 강제로 훼손하지 않습니다.

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
