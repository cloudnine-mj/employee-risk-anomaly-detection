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