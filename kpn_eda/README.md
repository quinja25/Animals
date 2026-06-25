# KPN 유전능력 자료 EDA 이미지 및 설명

## 01. 주요 형질 표준화육종가 분포
파일: `kpn_eda/01_core_breeding_value_distribution.png`

KPN별 핵심 형질(12개월체중, 도체중, 등심단면적, 등지방두께, 근내지방도)의 표준화육종가 분포를 비교한다. 0을 기준으로 각 형질의 우수/열위 방향을 직관적으로 확인할 수 있으며, 형질별 변동 폭이 큰 항목은 KPN 선택에 따른 차별성이 큰 변수로 해석할 수 있다.

## 02. 주요 형질 표준화육종가 상관관계
파일: `kpn_eda/02_core_breeding_value_correlation.png`

핵심 유전형질 간 동행 또는 trade-off 관계를 확인한다. 예를 들어 도체중, 등심단면적, 근내지방도, 등지방두께가 같은 방향으로 움직이는지 또는 일부 형질이 상충하는지 파악해 복합 유전지수 설계의 근거로 사용할 수 있다.

## 03. 주요 형질 유전능력 정확도 분포
파일: `kpn_eda/03_core_accuracy_distribution.png`

각 형질 육종가의 정확도 분포를 비교한다. 정확도가 낮은 형질은 모델 피처로 사용할 때 노이즈가 클 수 있으므로, 정확도 가중 평균이나 신뢰도 필터링 전략의 근거가 된다.

## 04. 근교계수 및 혈통 키 결측률
파일: `kpn_eda/04_inbreeding_and_pedigree_missingness.png`

근교계수 분포와 아비/조부/외조부 결측률을 함께 확인한다. 근교계수는 유전적 다양성과 관련된 위험 신호로 활용할 수 있고, 혈통 키 결측률은 조부·외조부 기반 확장 피처를 만들 때의 커버리지 한계를 보여준다.

## 05. 유전 프로파일 및 복합지수
파일: `kpn_eda/05_genetic_profile_and_composite_index.png`

근내지방도와 도체중을 축으로 KPN의 유전 프로파일을 시각화하고, 색상으로 등지방두께를 표시했다. 오른쪽의 복합지수는 `(근내지방도 + 도체중 + 등심단면적 - 등지방두께) / 4`로 계산했으며, 육질과 중량을 동시에 고려하는 KPN 선별 지표의 예시다.

## 06. KPN 유전능력 자료 매칭 커버리지
파일: `kpn_eda/06_kpn_match_coverage.png`

KPN 유전능력 엑셀의 `KPN명호`가 현재 `hanwoo_lineage.csv` 및 train 데이터의 `KPN_NO`와 얼마나 직접 연결되는지 확인한다. 현재 train 기준 직접 매칭률은 0.0%로, 유전능력값을 train에 바로 붙이기 전에 KPN 키 정합성 문제를 먼저 해결해야 한다.

## 생성 이미지
- `kpn_eda/01_core_breeding_value_distribution.png`
- `kpn_eda/02_core_breeding_value_correlation.png`
- `kpn_eda/03_core_accuracy_distribution.png`
- `kpn_eda/04_inbreeding_and_pedigree_missingness.png`
- `kpn_eda/05_genetic_profile_and_composite_index.png`
- `kpn_eda/06_kpn_match_coverage.png`
