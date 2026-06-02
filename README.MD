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
pip install lightgbm scikit-learn
```

### 5. Jupyter 노트북 실행

```bash
jupyter notebook hanwoo_eda.ipynb
```

또는 VS Code에서 `hanwoo_eda.ipynb`를 열고 우측 상단 커널 선택에서 `hanwoo`를 선택합니다.

### 6. 파이프라인 실행

`pipeline.py` 상단의 경로를 로컬 환경에 맞게 수정한 뒤 실행합니다.

```python
# pipeline.py 상단
DATA_DIR = "data"   # CSV 파일이 있는 디렉토리
OUT_DIR  = "submissions"  # 결과 파일 저장 디렉토리
```

```bash
python pipeline.py
```

실행 완료 시 `submissions/` 폴더에 제출 파일이 생성됩니다.

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
