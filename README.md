# 🛡️ Employee Risk Anomaly Detection System

**이상 행위 탐지 기반 내부감사 고도화 시스템**  
업무시간 외 로그인, 민감 리소스 접근, 시퀀스 이상 탐지, 분류모델, 대시보드, 리포트 자동화, 모델 재현성 관리까지 구현했습니다.

---

## 📁 프로젝트 구조

```bash
employee_risk_anomaly_detection/
├── .github/workflows/ci-deploy-reporting.yaml   # CI/CD: reporting 서비스 자동 배포
├── alerting/                                     # 알림 전송 및 억제 로직
│   └── alerting.py
├── dags/                                         # Airflow DAGs
│   ├── dag_anomaly_detection.py
│   ├── dag_classification.py
│   ├── dag_sequence_detection.py
│   ├── dag_reporting_and_dashboard.py
│   └── dag_combined_pipeline.py
├── data/
│   └── db.py                                     # DBClient
├── detector/
│   └── detector.py                               # IsolationForest 기반 RiskAnomalyDetector
├── features/                                     # 피처 엔지니어링
│   ├── feature_extractor.py
│   ├── classification_features.py
│   ├── sequence_features.py
├── governance/                                   # 운영 안정성 · 거버넌스
│   ├── audit_logger.py
│   ├── governance.py
│   └── mlflow_tracker.py
├── k8s/                                          # Kubernetes 매니페스트
│   ├── k8s_anomaly_manifest.yaml
│   └── reporting/reporting-manifests.yaml
├── models/                                       # ML 모델 정의
│   ├── classification.py
│   └── sequence_detector.py
├── reporting/                                    # 리포트 자동 생성 + Streamlit API
│   └── report_automation.py
├── config.yaml                                   # 시스템 설정파일
├── config.py                                     # 설정 로더
├── pipeline.py                                   # 전체 파이프라인 실행 (통합 버전)
├── metrics.py                                    # Prometheus 메트릭 수집 서버
├── scheduler.py                                  # APScheduler 재학습 스케줄러
├── main.py                                       # CLI 실행 진입점
├── requirements.txt                              # 의존성 목록
```

## ✅ 주요 구성 요소 요약

### 1. 감사지원 특화 피처 및 이상 탐지 모델 (Unsupervised + Sequence 기반)

- **IsolationForest 기반 이상 탐지**
- **업무시간 외 로그인 패턴 비율**
  - `off_hours_login_ratio` 피처 사용
- **민감 리소스 접근 가중치 점수**
  - `weighted_resource_score` 계산
- **업무 프로세스 시퀀스 이상 탐지**
  - Word2Vec + LSTM 기반 시퀀스 모델
  - 관련 모듈:  
    - `features/sequence_features.py`  
    - `models/sequence_detector.py`

---

### 2. 분류 모델 도입 (Supervised + XAI)

- **RandomForest 기반 분류 모델**
  - 구현 파일: `models/classification.py`
- **내부 감사 라벨링 데이터 기반 학습 (`is_risk`)**
- **설명 가능한 AI (XAI)**
  - SHAP 기반 피처 중요도 시각화
  - 메서드: `classification.explain()`

---

### 3. 리포트 및 대시보드 자동화

- **PPT/Excel 리포트 자동 생성**
  - 라이브러리: `python-pptx`, `openpyxl`
- **인터랙티브 대시보드**
  - Streamlit 기반 UI
  - 구현 위치: `reporting/report_automation.py`
- **다양한 알림 채널 연동**
  - Slack, Email(SES/SMTP), SMS(PagerDuty) 등
- **Grafana/Prometheus 기반 운영 대시보드**

---

### 4. 운영 안정성 및 거버넌스

- **모델 버전 관리**
  - MLflow 기반 Registry
  - 구현 위치: `governance/mlflow_tracker.py`
- **모델 성능 모니터링 및 임계치 경보**
  - Precision/Recall 등 주요 메트릭 평가
  - 관련 모듈:  
    - `metrics.py`  
    - `governance/performance_monitor.py`
- **감사 추적 로깅**
  - 탐지 및 알림 기록 저장
  - 구현 위치: `governance/audit_logger.py`
- **자동 재학습 스케줄링**
  - APScheduler 기반 이벤트/시간 기반 재학습
  - 모듈: `scheduler.py`
