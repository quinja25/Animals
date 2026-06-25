# 문서 추가용 EDA 후보

현재 HWPX 문서에는 등급 분포, 도체형질-등급 상관, 아비별 근내지방, 월별 도축/도체형질 추이, THI 노출률 분석이 포함되어 있다. 아래 항목은 문서에 없는 보완 EDA다.

## 1. 성별에 따른 육질등급 구성비
- 거세/암/수 등 성별별 육질등급 분포 차이를 확인하여 `JUDGE_SEX`를 핵심 범주형 변수로 사용하는 근거를 제시한다.

## 2. Train/Test 분포 drift
- 도축월, 시도, 성별의 train/test 분포 차이를 확인하여 검증 전략과 보정 필요성을 설명한다.

## 3. 결측 패턴과 등급 관계
- 도체형질 및 경락가격의 결측률이 특정 육질등급에 편중되는지 확인하여 결측치 처리 전략의 근거로 사용한다.

## 4. 외부 데이터 조인 커버리지
- weather, area, death, lineage 데이터가 train에 어느 정도 매칭되는지 제시하여 파생변수 활용 가능성과 한계를 명확히 한다.

## 5. 농장 규모·밀도·폐사율과 고등급 비율
- 농장 환경 변수가 육질 등급과 연결되는지 확인하여 농장 피처를 모델에 포함하는 근거를 제공한다.

## 생성 파일
- `eda_additions/01_sex_quality_distribution.png`
- `eda_additions/02_train_test_distribution_drift.png`
- `eda_additions/03_missingness_by_quality.png`
- `eda_additions/04_external_join_coverage.png`
- `eda_additions/05_farm_context_high_quality.png`
