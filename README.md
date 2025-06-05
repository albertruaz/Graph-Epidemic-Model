# 네트워크 기반 전염병 확산 시뮬레이션 모델

이 프로젝트는 그래프 이론을 기반으로 한 전염병 확산 시뮬레이션 도구입니다. 다양한 방역 대책의 효과를 정량적으로 분석하고 비교할 수 있는 기능을 제공합니다.

## 📋 목차

- [특징](#특징)
- [프로젝트 구조](#프로젝트-구조)
- [설치](#설치)
- [사용법](#사용법)
- [설정 파일](#설정-파일)
- [실행 스크립트](#실행-스크립트)
- [결과 분석](#결과-분석)
- [주요 지표](#주요-지표)

## ✨ 특징

- **네트워크 기반 모델링**: 실제 사회 네트워크 구조를 반영한 전염병 확산 시뮬레이션
- **다양한 방역 대책 분석**: 135개의 서로 다른 방역 대책 조합 테스트 지원
- **통계적 분석**: 여러 iteration을 통한 평균값과 표준편차 계산
- **시각화**: 히트맵을 통한 직관적인 결과 비교
- **공정한 실험 설계**: 동일한 초기 조건으로 모든 방역 대책 테스트

## 📁 프로젝트 구조

```
graph-epidemic-model/
├── config/                     # 설정 파일들
│   ├── config.json             # 기본 시뮬레이션 설정
│   ├── config_test.json        # 단일 실행 테스트 설정
│   └── config_test_statistics.json  # 통계 분석용 설정
├── models/                     # 시뮬레이터 모델들
│   ├── epidemic_simulator.py   # 기본 시뮬레이터
│   └── epidemic_simulator_test.py  # 테스트용 시뮬레이터
├── utils/                      # 유틸리티 함수들
│   ├── utils.py               # 공통 유틸리티
│   └── generate_heatmap.py    # 히트맵 생성 도구
├── saved_matrix/              # 네트워크 매트릭스 저장소
├── run.py                     # 기본 시뮬레이션 실행
├── run_test.py               # 방역 대책 비교 실행
├── run_test_statistics.py    # 통계적 분석 실행
├── environment.yml           # Conda 환경 파일
├── requirements.txt          # pip 패키지 목록
└── README.md                # 프로젝트 문서
```

## 🔧 설치

### Conda 환경 사용 (권장)

```bash
# 환경 생성 및 활성화
conda env create -f environment.yml
conda activate gem

# 또는 기존 환경에 설치
conda install --file requirements.txt
```

### pip 사용

```bash
pip install -r requirements.txt
```

### 필요한 패키지

- Python 3.10.16
- numpy 2.2.5
- pandas 2.2.3
- matplotlib 3.10.0
- seaborn 0.13.2
- networkx 3.4.2
- python-dateutil 2.9.0
- tqdm 4.67.1
- pillow 11.1.0
- pyyaml

## 🚀 사용법

### 1. 기본 시뮬레이션 실행

```bash
python run.py
```

- 단일 조건에서 전염병 확산 시뮬레이션 실행
- SIR 모델 기반 동적 분석
- 네트워크 시각화 및 애니메이션 생성

### 2. 방역 대책 비교 분석

```bash
python run_test.py
```

- 여러 방역 대책의 효과 비교
- 각 대책별 개별 결과 저장
- 빠른 비교 분석용

### 3. 통계적 분석 (권장)

```bash
python run_test_statistics.py
```

- 100회 반복 실험을 통한 통계적 분석
- 135개 방역 대책 조합 테스트
- 평균값과 표준편차 계산
- 자동 히트맵 생성

## ⚙️ 설정 파일

### config.json (기본 설정)

```json
{
  "defaults": {
    "N1": 1,
    "N2": 2,
    "N3": 3, // 접촉률 매개변수
    "tau": 0.3, // 감염 확률
    "alpha": 0.1, // 회복률
    "xi": 0.02, // 사망률
    "T": 100, // 시뮬레이션 스텝
    "seed": 42, // 랜덤 시드
    "init_infected_count": 4, // 초기 감염자 수
    "save_matrix": "random1" // 사용할 네트워크 매트릭스
  }
}
```

### config_test_statistics.json (통계 분석용)

```json
{
  "defaults": {
    "iterations": 100,           // 반복 실험 횟수
    "methods": [[2,4,8], ...],  // 테스트할 방역 대책들
    "limit_starting_step": 4,    // 전염 시작 제한 스텝
    "init_infected_method": "random"  // 초기 감염자 선택 방법
  }
}
```

## 📊 실행 스크립트

### run.py

- **용도**: 기본 시뮬레이션 실행
- **특징**: 단일 조건 분석, 상세한 시각화

### run_test.py

- **용도**: 방역 대책 비교 (단일 실행)
- **특징**: 빠른 비교, 개별 결과 저장

### run_test_statistics.py

- **용도**: 통계적 분석 (권장)
- **특징**:
  - 100회 반복 실험
  - 고정된 초기 감염자 조합으로 공정한 비교
  - 자동 히트맵 생성
  - 평균값과 표준편차 계산

## 📈 결과 분석

### 생성되는 파일들

1. **시뮬레이션 결과 파일**

   - `simulation_results_iterations_YYYYMMDD_HHMMSS.txt`
   - 각 방역 대책별 평균값과 표준편차

2. **히트맵 파일들**

   - `heatmap_Infection_Coverage_Ratio_mean_by_N1.png`
   - `heatmap_Infection_Duration_mean_by_N1.png`
   - `heatmap_Duration_Coverage_Ratio_mean_by_N1.png`

3. **요약 통계**
   - `summary_statistics_iterations.txt`
   - 전체 결과에 대한 통계적 요약

### 히트맵 해석

- **X축**: N2 (접촉 유형 2의 강도)
- **Y축**: N3 (접촉 유형 3의 강도)
- **색상**: 각 지표의 값 (밝을수록 높은 값)
- **N1별 분리**: 접촉 유형 1의 강도별로 별도 히트맵

## 📊 주요 지표

### 1. Infection Coverage Ratio (감염 커버리지 비율)

- **정의**: 전체 인구 중 감염된 비율
- **범위**: 0.0 ~ 1.0
- **해석**: 낮을수록 방역 효과가 좋음

### 2. Infection Duration (감염 지속 기간)

- **정의**: 전염병이 지속되는 스텝 수
- **단위**: 시뮬레이션 스텝
- **해석**: 짧을수록 빠른 종료

### 3. Duration/Coverage Ratio (지속기간/커버리지 비율)

- **정의**: 감염 지속 기간을 커버리지로 나눈 값
- **해석**: 감염 확산의 효율성 지표

### 4. R0 (기초감염재생산수)

- **정의**: 1명의 감염자가 평균적으로 감염시키는 사람 수
- **임계값**: R0 > 1이면 전염병 확산, R0 < 1이면 소멸

## 🔬 실험 설계

### 공정한 비교를 위한 설계

1. **고정된 초기 감염자**: 모든 방역 대책이 동일한 초기 조건에서 테스트
2. **다중 반복**: 100회 반복을 통한 통계적 신뢰성 확보
3. **체계적 조합**: N1(0~2) × N2(0~4) × N3(0~8)의 135개 조합 테스트

### 방역 대책 매개변수

- **N1**: 가까운 접촉 (가족, 동거인)
- **N2**: 중간 접촉 (직장, 학교)
- **N3**: 먼 접촉 (사회 활동, 모임)

값이 작을수록 강한 방역 조치를 의미합니다.
