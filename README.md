# League of Legends 데이터 분석 파이프라인

이 저장소는 **리그 오브 레전드(LoL)** 경기 데이터를 수집, 전처리, 분석하는 Python 스크립트와 모듈을 포함합니다.  
프로젝트의 목표는 팀 음성 채팅과 게임 내 이벤트가 경기 결과와 어떻게 연관되는지 연구하는 것입니다.  
실시간 음성 인식, 경기 데이터 수집, 발화 구간 감지, 데이터셋 생성, 머신러닝 모델링 및 시각화 등의 기능을 제공합니다.

## 주요 구성 요소

### 1. 데이터 수집 (`data_collector.py`)
- LoL Live Client API를 통해 실시간 경기 데이터 수집
- 게임 시작/종료 감지 및 이벤트 로그 저장
- 데이터 압축 및 이메일 전송 기능 포함

### 2. 전처리 파이프라인 (`preprocessing/`)
- **`voice_processor.py`**: 오디오 파일에서 발화/비발화 구간 감지
- **`feature_extractor.py`**: 이벤트 로그에서 주요 특징 추출
- **`event_classifier.py`**: 챔피언 처치, 오브젝트 파괴 등 이벤트 분류 및 골드 보상 계산
- **`dataset_builder.py`**: 모델 학습용 CSV 데이터셋 생성
- **`utils.py`**: 시간 변환, 골드 누적 관리 등 유틸리티 함수 제공

### 3. 모델링 (`model_min.py`, `model_event.py`, `modeling_sna.py`)
- 머신러닝 모델을 이용한 승률 예측 및 중요 특징 분석
- 이벤트 기반 데이터 분석 및 시각화
- 네트워크 분석을 통한 발화 패턴 연구

### 4. Discord 봇 (`bot.py`)
- 음성 채널에서 발화 감지 및 데이터 수집
- 멀티 서버 환경에서 안정적인 데이터 기록

### 5. 보조 스크립트
- **`Recording_test/`**: 오디오 녹음 및 패키징 테스트 코드

## 실행
```bash
python data_collector.py        # 실시간 경기 데이터 수집 시작
python preprocessing/main.py    # 전처리 파이프라인 실행
python model_event.py           # 이벤트 기반 분석 실행
```
- **배포**: 실제 환경에서는 모든 환경에서 실행할 수 있도록 cx_Freeze Module을 사용하여, exe로 빌드하여 배포함. 

## 기술 스택

* **언어**: Python 3.8+
* **데이터 처리**: pandas, numpy
* **머신러닝**: scikit-learn, xgboost
* **음성 처리**: librosa, pYIN 알고리즘
* **시각화**: matplotlib, networkx
* **기타**: Discord.py, Riot Games API

---
