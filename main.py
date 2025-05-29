"""
setup_logging(): 기본 로그 포맷 설정
main():
- 설정 로드 (config.yaml)
- Prometheus 메트릭 서버 기동
- 자동 재학습 스케줄러 초기화
- 탐지 파이프라인 실행
- 스케줄러 실행 유지 및 종료 처리
"""

import argparse
import logging
import sys
from config import load_config
from metrics import start_metrics_server
from pipeline import run_pipeline
from scheduler import RetrainScheduler


def setup_logging():
    # 기본 로깅 설정을 수행, 로그에 타임스탬프, 레벨, 메시지를 출력하도록 구성함.
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]


def main():
    # 로깅 설정 초기화
    setup_logging()

    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='리스크 이상 탐지 엔트리포인트')
    parser.add_argument('--config', type=str, default='config.yaml', help='설정 파일 경로')
    args = parser.parse_args()

    # 설정 파일 로드
    cfg = load_config(args.config)
    logging.info('설정 로드 완료: %s', args.config)

    # Prometheus 메트릭 서버 시작 (metrics_port 설정이 있을 경우)
    metrics_port = cfg.get('metrics_port')
    if metrics_port is not None:
        start_metrics_server(metrics_port)

    # 자동 재학습 스케줄러 초기화 (enable_retrain_scheduler 설정이 있을 경우)
    if cfg.get('enable_retrain_scheduler'):
        def retrain_func():
            run_pipeline(cfg)

        sched = RetrainScheduler(
            retrain_func,
            periodic_interval_hours=cfg.get('retrain_hourly', 24),
            event_threshold=cfg.get('retrain_event_threshold', 5),
            event_window_minutes=cfg.get('retrain_event_window', 10)
        )
        logging.info('RetrainScheduler 초기화 완료')

    # 이상 탐지 파이프라인 실행
    run_pipeline(cfg)

    # 스케줄러가 동작 중이면 메인 스레드 유지
    try:
        if cfg.get('enable_retrain_scheduler'):
            logging.info('스케줄러가 실행 중입니다. 종료하려면 Ctrl+C를 누르세요.')
            while True:
                pass
    except KeyboardInterrupt:
        # 스케줄러 종료 처리
        if cfg.get('enable_retrain_scheduler'):
            sched.shutdown()
        logging.info('프로그램을 종료합니다.')

if __name__ == '__main__':
    main()
