# 주제2: 기상에 따른 한우 등급 예측

> 기상청 빅데이터 콘테스트 — 축산물품질평가원 & 기상청 제공 데이터

---

## 환경 설정

### 1. Conda 설치 (Miniforge)

**macOS (Homebrew)**

```bash
brew install miniforge
conda init zsh
```

**Windows**

PowerShell에서 winget으로 설치합니다 (Windows 10 1709 이상 기본 내장):

```powershell
winget install CondaForge.Miniforge3
```

설치 완료 후 전체 경로로 conda를 초기화합니다 (설치 직후에는 `conda` 명령어가 PATH에 없음):

```powershell
& "$env:USERPROFILE\miniforge3\Scripts\conda.exe" init powershell
```

> PowerShell 실행 정책 오류가 뜨면 먼저 실행합니다:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

초기화 후 PowerShell을 **완전히 닫고 새로 열면** `conda` 명령어를 바로 사용할 수 있습니다.

### 2. 가상환경 생성 및 활성화

```bash
conda create -n hanwoo python=3.11 jupyter pandas matplotlib seaborn -y
conda activate hanwoo
```

### 3. 한국어 폰트 설정

노트북은 OS를 자동 감지해 폰트를 선택합니다.

| OS | 사용 폰트 | 설치 필요 여부 |
|----|-----------|:------------:|
| Windows | 맑은 고딕 (Malgun Gothic) | ❌ 기본 내장 |
| macOS | AppleGothic | ❌ 기본 내장 |
| Linux | NanumGothic | ✅ 별도 설치 필요 |

**Linux에서 NanumGothic 설치:**

```bash
conda install -c conda-forge font-nanum -y
# 설치 후 폰트 캐시 초기화
python -c "import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)"
```

**폰트가 여전히 깨지는 경우** (폰트 캐시 초기화):

```python
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
```

### 4. 추가 패키지 설치

```bash
pip install "numpy<2" lightgbm catboost xgboost scikit-learn
```

> 현재 Conda용 CatBoost 1.2.5는 NumPy 2.x와 바이너리 ABI가 맞지 않을 수 있습니다. `numpy.dtype size changed` 오류가 발생하면 `conda install -n hanwoo -c conda-forge "numpy=1.26.4" -y`로 NumPy 버전을 맞춥니다.

### 5. Jupyter 노트북 실행

```bash
jupyter notebook hanwoo_eda.ipynb
```

또는 VS Code에서 `hanwoo_eda.ipynb`를 열고 우측 상단 커널 선택에서 `hanwoo`를 선택합니다.

### 6. 파이프라인 실행

현재 최신 실험 파이프라인은 `pipelines/pipeline_v14.py`입니다. 프로젝트 루트에서 실행합니다.

```powershell
python pipelines/pipeline_v14.py
```

CatBoost GPU 학습이 가능한 환경에서는 다음과 같이 실행할 수 있습니다.

```powershell
$env:CATBOOST_TASK_TYPE="GPU"
$env:XGBOOST_DEVICE="cuda"
python pipelines/pipeline_v14.py
```

전체 학습 전에 첫 번째 LightGBM Fold만 검증하려면 다음 smoke test를 실행합니다.

```powershell
$env:V14_SMOKE_STAGE="lgb"
python pipelines/pipeline_v14.py
```

첫 Fold의 LightGBM, CatBoost, XGBoost 구성요소를 모두 검증하려면 다음을 실행합니다.

```powershell
$env:V14_SMOKE_STAGE="components"
python pipelines/pipeline_v14.py
$env:V14_SMOKE_STAGE=""
```

실행 완료 시 `submissions/v14/`에 제출 파일과 검증 산출물이 생성됩니다.

| 파일 | 설명 |
|------|------|
| `260418.csv` | 최종 제출 파일 |
| `component_metrics.csv` | Fold별 직접·육질·육량·계층 모델 Macro F1 |
| `quality_blend_folds.csv` | Ordinal·B specialist·C multiplier 선택 결과 |
| `calibration_folds.csv` | 교차 적합 보정 성능과 블렌드 설정 |
| `class_scales.csv` | Fold별 클래스 확률 보정 계수 |
| `oof_class_metrics.csv` | 클래스별 OOF 정밀도·재현율·F1 |
| `oof_predictions.npz` | 직접·계층·보정 OOF 확률과 Fold 번호 |
| `test_components.npz` | 테스트 구성요소별 원본 확률 캐시 |
| `lgb_feature_importance.csv` | LightGBM 피처 중요도 |

> v10은 240만 건 전체 데이터에 대해 5-Fold로 LightGBM 1개와 CatBoost 2개를 학습하므로 실행 시간이 길고 메모리 사용량이 큽니다. NVIDIA GPU가 있는 환경에서는 반드시 `CATBOOST_TASK_TYPE=GPU` 사용을 권장합니다.

---

## 모델 개발 현황

| 버전 | 핵심 모델 | 검증 전략 | 성능 |
|:---:|------|------|------|
| v6 | 계층형 LightGBM, 누적 기상 피처 | Farm GroupKFold | LB 0.212 |
| v7 | CatBoost 단일 모델, 데이터 정합성 개선 | StratifiedGroupKFold | 별도 기록 없음 |
| v8 | LightGBM + CatBoost 직접 16클래스 앙상블 | StratifiedGroupKFold | OOF 0.2126 |
| v9 | 직접 LightGBM + 계층형 CatBoost + 클래스 보정 | StratifiedGroupKFold | **LB 0.239**, 보정 OOF 0.2430 |
| v10 | Fold-safe KPN·관측소 사전확률 + 교차 적합 보정 | StratifiedGroupKFold | Fold 1 계층 F1 0.1722로 중단 |
| v11 | LightGBM 직접 + CatBoost/Ordinal 육질 + XGBoost 순서형 육량 | StratifiedGroupKFold + Cross-fit | **교차 적합 OOF 0.2449**, LB 제출 예정 |
| v12 | 확장 혈통 + 완화된 C 가중치 + B specialist 검증 | StratifiedGroupKFold + Cross-fit | **교차 적합 OOF 0.2460**, LB 0.241 수준 |
| v13 | Nested 도체형질 복원 + Dual-C 앙상블 | Nested Farm CV + Cross-fit | **원본 계층 OOF 0.2381**, **교차 적합 OOF 0.2477**, **LB 0.244** |
| v14 | 직접 경계 복원 + B/C 전문 모델 + 드리프트 저항 분기 + 지역 보정 | Nested Farm CV + Nested calibration | 구현 완료, 전체 검증 예정 |

v9의 성능 향상은 단순한 모델 용량 증가보다 다음 세 요소에서 발생했습니다.

- 최종 16개 등급을 직접 예측하는 모델과 육질·육량을 분리하는 계층 모델의 오류 다양성
- 불균형 클래스에 대한 클래스별 확률 스케일 보정
- 테스트 농장과 훈련 농장의 중복이 0%인 조건을 반영한 농장 그룹 교차검증

v10은 이 구조를 유지하면서 KPN 및 기상 관측소의 과거 육질·육량 분포를 Fold 내부에서만 계산하고, 후처리 역시 교차 적합하여 v9의 낙관적인 OOF 평가를 줄이는 것을 목표로 합니다. 상세 내용은 [PIPELINE.MD](PIPELINE.MD)를 참고하세요.

---

## 데이터 개요

| 파일명 | 설명 | 행 수 |
|--------|------|------:|
| `hanwoo_train.csv` | 도축·등급판정 훈련 데이터 | 2,408,699 |
| `hanwoo_weather.csv` | 기상 관측 데이터 | 973,248 |
| `hanwoo_area.csv` | 농장별 사육두수·면적 | 91,896 |
| `hanwoo_death.csv` | 폐사 기록 | 321,389 |
| `hanwoo_lineage.csv` | 혈통 정보 | 1,809,455 |

---

## 1. 훈련 데이터 (`hanwoo_train.csv`)

도축 및 등급판정 결과 데이터. **예측 대상(target)은 `LAST_GRADE`**.

| 칼럼명 | 타입 | 설명 | 비고 |
|--------|------|------|------|
| `sido` | string | 시·도 | 17개 시도 (강원특별자치도, 경기도, …) |
| `sigungu` | string | 시·군·구 | |
| `eupmyeondong` | string | 읍·면·동 | |
| `stn` | int | 기상 관측소 번호 | `hanwoo_weather.csv`의 `stn`과 조인 키 |
| `ABATT_DATE` | string | 도축일자 | YYYY-MM-DD |
| `JUDGE_DATE` | string | 등급판정일자 | YYYY-MM-DD |
| `JUDGE_SEX` | string | 판정 성별 | `암`, `수`, `거세` |
| `WEIGHT` | int | 도체중량 (kg) | |
| `BACKFAT` | float | 등지방두께 (mm) | 등심단면 2/3 부분 측정, 불가식 지방의 분포 |
| `REA` | float | 배최장근단면적 (cm²) | 등심단면 배최장근을 cm² 단위로 측정 |
| `WINDEX` | float | 육량지수 | 성별·품종별 산식으로 산출 (아래 참고) |
| `WGRADE` | string | 육량등급 | `A`, `B`, `C`, `D` / `-99.0`=결측 |
| `INSFAT` | float | 근내지방도 (마블링) | 1~9 (No.1~No.9) / `-99`=결측. 연도·풍미·보수성·다즙성 관련 |
| `YUKSAK` | float | 육색 | 1~7 (No.1~No.7) / `-99`=결측. 소비자 선택 중요 요소 |
| `FATSAK` | float | 지방색 | 1~7 (No.1~No.7) / `-99`=결측. 도체 상품가치에 중요 |
| `TISSUE` | float | 조직감 | 1~5 (No.1~No.5) / `-99`=결측. 고기의 탄력성·보수성·결·광택 상태 |
| `GROWTH` | float | 성숙도 | 1~9 / `-99`=결측. 가축의 생리적 연령, 조직의 연도·성상변화·풍미 관련 |
| `COST_AMT` | float | 경락가격 (원) | `-99`=결측 |
| `AGE` | int | 월령 (개월) | |
| `BIRTH_YMD` | int | 출생일자 | YYYYMMDD |
| `CATTLE_NO` | string | 개체번호 (암호화) | `hanwoo_lineage.csv`의 `CATTLE_NO`와 조인 키 |
| `FARM_UNIQUE_NO` | string | 농장 고유번호 (암호화) | `hanwoo_area.csv`, `hanwoo_death.csv`와 조인 키 |
| `LAST_GRADE` | string | **최종 등급 (예측 대상)** | `1++A`~`1++C`, `1+A`~`1+C`, `1A`~`1C`, `2A`~`2C`, `3A`~`3C`, `등외` |

### 최종 등급 구조

최종 등급 = **육질등급(1++, 1+, 1, 2, 3)** + **육량등급(A, B, C)**

- **육량등급**: 도체로부터 얻을 수 있는 살코기의 양을 등급화 (도체중량, 등지방두께, 배최장근단면적으로 산출)
- **육질등급**: 근내지방도와 육질의 품질을 종합적으로 판정 (근내지방도, 육색, 지방색, 조직감, 성숙도)

---

## 2. 기상 데이터 (`hanwoo_weather.csv`)

기상청 관측소별 일별 기상 데이터. 2020-01-01 ~ 기간 포함.

| 칼럼명 | 타입 | 설명 | 단위 |
|--------|------|------|------|
| `stn` | int | 기상 관측소 번호 | `hanwoo_train.csv`의 `stn`과 조인 키 |
| `date` | string | 관측 일자 | YYYY-MM-DD |
| `ta_max` | float | 일 최고기온 | °C |
| `rn_day` | float | 일 강수량 | mm |
| `ta_min` | float | 일 최저기온 | °C |
| `rhm_avg` | float | 일 평균 상대습도 | % |
| `ws_davg` | float | 일 평균 풍속 | m/s |

---

## 3. 사육면적 데이터 (`hanwoo_area.csv`)

농장별 연도별 사육두수 및 축사 면적.

| 칼럼명 | 타입 | 설명 | 비고 |
|--------|------|------|------|
| `FARM_UNIQUE_NO` | string | 농장 고유번호 (암호화) | 조인 키 |
| `C2023` | float | 2023년 사육두수 | 마리 |
| `C2024` | float | 2024년 사육두수 | 마리 |
| `C2025` | float | 2025년 사육두수 | 마리 |
| `AREA` | float | 축사 면적 | m² |

---

## 4. 폐사 데이터 (`hanwoo_death.csv`)

농장별 개체 폐사 기록.

| 칼럼명 | 타입 | 설명 | 비고 |
|--------|------|------|------|
| `FARM_UNIQUE_NO` | string | 농장 고유번호 (암호화) | 조인 키 |
| `BIRTH_YMD` | int | 출생일자 | YYYYMMDD |
| `DEAD_YMD` | int | 폐사일자 | YYYYMMDD |
| `DEAD_REASON` | string | 폐사 원인 | `-99`=결측 |

---

## 5. 혈통 데이터 (`hanwoo_lineage.csv`)

개체별 3대 혈통 정보 (부모 + 조부모).

| 칼럼명 | 타입 | 설명 | 비고 |
|--------|------|------|------|
| `CATTLE_NO` | string | 개체번호 (암호화) | 조인 키 |
| `KPN_NO` | string | KPN 번호 (암호화) | 한우 종축 등록번호 |
| `FATHER_CATTLE_NO` | string | 부(아비) 개체번호 | |
| `MOTHER_ANIMAL_NO` | string | 모(어미) 개체번호 | |
| `F_GMOTHER_ANIMAL_NO` | string | 부계 조모(할머니) 개체번호 | |
| `F_GFATHER_CATTLE_NO` | string | 부계 조부(할아버지) 개체번호 | |
| `M_GMOTHER_ANIMAL_NO` | string | 모계 조모(할머니) 개체번호 | |
| `M_GFATHER_CATTLE_NO` | string | 모계 조부(할아버지) 개체번호 | |

---

## 등급판정 기준 상세

### 육량지수 산식

성별·품종별 산정된 육량지수 산식에 따라 3개 등급으로 구분.

**한우:**

| 성별 | 육량지수 산식 |
|------|-------------|
| 암 | `[6.90137 − 0.9446 × 등지방두께(mm) + 0.31806 × 배최장근단면적(cm²) + 0.54952 × 도체중량(kg)] ÷ 도체중량(kg) × 100` |
| 수 | `[0.20108 − 2.18625 × 등지방두께(mm) + 0.29275 × 배최장근단면적(cm²) + 0.64099 × 도체중량(kg)] ÷ 도체중량(kg) × 100` |
| 거세 | `[11.06338 − 1.25149 × 등지방두께(mm) + 0.28238 × 배최장근단면적(cm²) + 0.56781 × 도체중량(kg)] ÷ 도체중량(kg) × 100` |

### 육량등급 판정 기준 (한우)

| 성별 | A등급 | B등급 | C등급 |
|------|-------|-------|-------|
| 암 | 61.83 이상 | 59.70 이상 ~ 61.83 미만 | 59.70 미만 |
| 수 | 68.45 이상 | 66.32 이상 ~ 68.45 미만 | 66.32 미만 |
| 거세 | 62.52 이상 | 60.40 이상 ~ 62.52 미만 | 60.40 미만 |

### 육질등급 판정 기준

각 항목중 가장 낮은 등급을 **최저 등급**으로 적용한 뒤, 성숙도를 반영하여 **최종 등급** 결정.

| 항목 | 1++ | 1+ | 1 | 2 | 3 |
|------|-----|-----|---|---|---|
| 근내지방도 | No.7⁺ ~ No.9⁺⁺ | No.5⁺⁺ ~ No.7⁰ | No.4⁰ ~ No.5⁺ | No.2⁰ ~ No.3⁺⁺ | No.1⁰ ~ No.1⁺⁺ |
| 육색 | No.3 ~ No.5 | No.2, No.6 | No.1 | No.7 | 그 외 |
| 지방색 | No.1 ~ No.4 | No.5 | No.6 | No.7 | 그 외 |
| 조직감 | No.1 | No.2 | No.3 | No.4 | No.5 |

**성숙도 반영:**
- 성숙도 1 → **상**, 2~7 → **점**, 8~9 → **결격**
- 성숙도 8~9인 경우 최저등급에서 1개 등급 하락

**예외 규칙:**
- 육색과 지방색이 모두 No.7에 해당하여 2등급인 경우 → 육질등급을 **3등급**으로 판정
- 근내지방도와 육색·지방색·조직감의 평가결과가 2개 등급 이상 차이나는 경우 → 성숙도를 적용하지 않고 최저 등급을 최종 등급으로 함

### 조직감 구분 기준

| 번호 | 구분 기준 |
|------|----------|
| 1 | 탄력성과 지방의 결이 매우 좋으며 수분이 알맞게 침출되고 결이 매우 곱고 섬세하며 고기의 광택이 매우 좋은 것 |
| 2 | 탄력성과 지방의 결이 좋으며 수분이 다소 알맞게 침출되고 결이 곱고 고기의 광택이 좋은 것 |
| 3 | 탄력성과 지방의 결이 보통이며 수분의 침출정도가 약간 많거나 적고 결과 고기의 광택이 보통인 것 |
| 4 | 탄력성과 지방의 결이 포장에 비해 좋지 않은 수준이며 고기의 표면이 건조하거나 수분이 결이 침출되고 결과 광택이 보통에 비해 좋지 않은 것 |
| 5 | 탄력성과 지방의 결이 나쁘며 고기의 표면이 매우 건조하거나 수분이 아주 많이 침출되고 결과 광택이 매우 좋지 않은 것 |

---

## 데이터 조인 관계

```
hanwoo_train (메인)
  ├── stn ──────────── hanwoo_weather.stn (+ date로 기상 매칭)
  ├── FARM_UNIQUE_NO ─ hanwoo_area.FARM_UNIQUE_NO
  ├── FARM_UNIQUE_NO ─ hanwoo_death.FARM_UNIQUE_NO
  └── CATTLE_NO ────── hanwoo_lineage.CATTLE_NO
```

## 결측값 표기

- `-99` 또는 `-99.0`: 해당 항목 결측 (COST_AMT, WGRADE, INSFAT, YUKSAK, FATSAK, TISSUE, GROWTH, DEAD_REASON 등)

---

## 대회 심사 기준

> 출처: [기상청 빅데이터 콘테스트](https://bd.kma.go.kr/contest/info_04.do)

### 심사 일정

| 단계 | 일정 | 방식 |
|------|------|------|
| 1차 심사 (예선) | 2026-07-08 | 서류 평가 |
| 2차 심사 (본선) | 2026-08-05 | 발표 평가 |

### 평가 항목 (6개)

| 항목 | 설명 |
|------|------|
| 모델 예측 정확성 | 예측 모델의 성능 및 정확도 |
| 합리성 | 분석 방법론의 논리적 타당성 |
| 데이터 분석능력 | 데이터 전처리·탐색·활용 능력 |
| 활용성 | 실제 현장 적용 가능성 |
| 창의성 | 독창적인 접근 방식 |
| 부합성 | 주제와의 적합도 |

### 주제2 검증 방식

주제2(한우)는 **별도의 자동 검증 시스템을 운영하지 않으며, 자율형식의 분석 과제**로 운영됩니다.
따라서 자체적으로 검증 방법을 설계하고 결과를 보고서에 포함해야 합니다.

---

## 프로젝트 파이프라인

단계별 진행 방법, 피처 엔지니어링 전략, 모델 구축 계획은 [PIPELINE.MD](PIPELINE.MD)를 참고하세요.

---

## Repository Structure

This public repository keeps the final runnable model at the project root:

- `pipeline_v13_model.py`: final model pipeline.
- `PIPELINE.MD`: pipeline history, modeling decisions, validation notes, and archived experiment summary.
- `archive/`: older pipelines, temporary scripts, scratch analysis, and superseded helper files.
- `eda_additions/`: generated EDA figures used to support feature and validation decisions.
- `kpn_eda/`: generated KPN genetic-value EDA figures.

Generated data, trained model artifacts, virtual environments, submissions, and local tool caches are excluded through `.gitignore`.

## Submission Standards

All submission-producing pipelines should follow these rules:

- Filename: `260418.csv`.
- Encoding: UTF-8 with BOM (`utf-8-sig`).
- Required column: `LAST_GRADE`.
- Allowed labels: `1++A`, `1++B`, `1++C`, `1+A`, `1+B`, `1+C`, `1A`, `1B`, `1C`, `2A`, `2B`, `2C`, `3A`, `3B`, `3C`, `등외`.
- Recommended output location: `submissions/{version}/260418.csv`.

Target label encoding used during modeling:

| Code | Grade |
|---:|:---|
| 0 | `1++A` |
| 1 | `1++B` |
| 2 | `1++C` |
| 3 | `1+A` |
| 4 | `1+B` |
| 5 | `1+C` |
| 6 | `1A` |
| 7 | `1B` |
| 8 | `1C` |
| 9 | `2A` |
| 10 | `2B` |
| 11 | `2C` |
| 12 | `3A` |
| 13 | `3B` |
| 14 | `3C` |
| 15 | `등외` |

## Data Join Standards

Use normalized string keys before joining:

- `CATTLE_NO`
- `KPN_NO`
- `FARM_UNIQUE_NO`

Recommended normalization:

```python
df[key] = df[key].astype(str).str.strip()
```

Primary joins:

| Source | Key | Target | Notes |
|:---|:---|:---|:---|
| `hanwoo_train/test` | `CATTLE_NO` | `hanwoo_lineage` | Deduplicate lineage by `CATTLE_NO` before joining. |
| `hanwoo_lineage` | `KPN_NO` | KPN genetic data | Use breeding-value fields where coverage is valid. |
| `hanwoo_train/test` | `FARM_UNIQUE_NO` | `hanwoo_area` | Use farm area and yearly farm context. |
| `hanwoo_area` | `FARM_UNIQUE_NO` | `hanwoo_death` | Aggregate death counts before joining. |
| `hanwoo_train/test` | `stn`, year, month | `hanwoo_weather` | Aggregate weather to monthly summaries before joining. |

Because train and test farms have no farm overlap, do not use direct `FARM_UNIQUE_NO` target encoding for final generalization. Use fold-safe historical statistics through broader keys such as `KPN_NO`, station (`stn`), region, or other non-leaking context.

## EDA Artifacts

Additional EDA figures:

- `eda_additions/01_sex_quality_distribution.png`: sex or judge-sex quality-grade distribution.
- `eda_additions/02_train_test_distribution_drift.png`: train/test drift by year and key fields.
- `eda_additions/03_missingness_by_quality.png`: missingness patterns by quality grade.
- `eda_additions/04_external_join_coverage.png`: coverage of weather, area, death, and lineage joins.
- `eda_additions/05_farm_context_high_quality.png`: farm size, density, and high-grade relationship.
- `eda_additions/06_grade_distribution_from_final_writeup.png`: final, quality, and yield grade imbalance from the final proposal.
- `eda_additions/07_carcass_trait_grade_correlation_heatmap.png`: carcass-trait correlation heatmap against quality and yield grades.
- `eda_additions/08_sex_quality_composition_from_final_writeup.jpg`: sex-wise quality-grade composition from the final proposal.

KPN EDA figures:

- `kpn_eda/01_core_breeding_value_distribution.png`: core breeding-value distributions.
- `kpn_eda/02_core_breeding_value_correlation.png`: correlation among genetic breeding values.
- `kpn_eda/03_core_accuracy_distribution.png`: distribution of breeding-value accuracy.
- `kpn_eda/04_inbreeding_and_pedigree_missingness.png`: inbreeding and pedigree missingness.
- `kpn_eda/05_genetic_profile_and_composite_index.png`: genetic profile and composite index example.
- `kpn_eda/06_kpn_match_coverage.png`: KPN matching coverage against lineage and train data.

## Final Proposal Summary

The final proposal frames the task as a weather and livestock data fusion problem for predicting carcass grade before shipment. Weather matters because high-temperature, high-humidity exposure raises heat stress, reduces intake and weight gain, and can suppress intramuscular fat deposition.

Key EDA findings reflected in the final writeup:

- Final grade labels are imbalanced, especially `3C` and `등외`.
- Yield grade `C` is underrepresented relative to `A` and `B`, which motivates dedicated C-grade handling.
- Backfat thickness and yield index are the strongest yield-grade indicators.
- Intramuscular fat is the strongest quality-grade indicator, while tissue texture is negatively related to quality grade.
- Sex is structurally important: steer and female cattle have much higher high-quality-grade share than male cattle.

The final modeling objective is to predict 16 final carcass-grade classes using weather exposure, farm context, lineage/genetic information, and animal-level physical features while preserving unseen-farm validation realism.
