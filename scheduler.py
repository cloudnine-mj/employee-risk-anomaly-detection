# RetrainScheduler: 주기적·이벤트 기반 자동 재학습 구현
# record_event()로 이벤트 기록 후 threshold 초과 시 즉시 재학습 트리거
# shutdown()으로 스케줄러 종료

import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Callable, Optional, List

class RetrainScheduler:
    """
    자동 재학습 스케줄러
    - 주기적 재학습: 일정 시간 간격으로 retrain 함수를 호출
    - 이벤트 기반 재학습: 최근 window 내 anomaly 이벤트 수가 threshold 초과 시 즉시 retrain
    """
    def __init__(
        self,
        retrain_func: Callable[[], None],
        periodic_interval_hours: Optional[float] = None,
        event_threshold: Optional[int] = None,
        event_window_minutes: Optional[int] = None
    ):
        """
        :param retrain_func: 모델 재학습을 수행하는 함수
        :param periodic_interval_hours: 주기적 재학습 주기 (시간 단위)
        :param event_threshold: 이벤트 기반 재학습 임계치 (이상탐지 이벤트 수)
        :param event_window_minutes: 이벤트 기반 윈도우 크기 (분)
        """
        self.retrain_func = retrain_func
        self.scheduler = BackgroundScheduler()
        self.events: List[datetime] = []
        self.event_threshold = event_threshold
        self.event_window = timedelta(minutes=event_window_minutes) if event_window_minutes else None

        # 설정된 경우 주기적 재학습 작업 추가
        if periodic_interval_hours and periodic_interval_hours > 0:
            self.scheduler.add_job(
                self._trigger_retrain,
                'interval',
                hours=periodic_interval_hours,
                id='periodic_retrain'
            )
            logging.info(f"Scheduled periodic retrain every {periodic_interval_hours} hours")

        self.scheduler.start()
        logging.info("RetrainScheduler started")

    def record_event(self):
        # 이상 이벤트 발생 시 호출하여 기록하고, 이벤트 기반 임계치 초과 시 재학습 트리거
        now = datetime.utcnow()
        self.events.append(now)
        if self.event_window and self.event_threshold:
            # 윈도우 내 이벤트만 유지
            cutoff = now - self.event_window
            self.events = [t for t in self.events if t >= cutoff]
            if len(self.events) >= self.event_threshold:
                logging.info(
                    f"Event-based retrain triggered: {len(self.events)} events in last {self.event_window}"
                )
                self._trigger_retrain()
                self.events.clear()

    def _trigger_retrain(self):
        # 내부적으로 재학습 함수를 호출
        try:
            logging.info("Starting model retraining...")
            self.retrain_func()
            logging.info("Model retraining completed")
        except Exception as e:
            logging.error(f"Retraining failed: {e}")

    def shutdown(self):
        # 스케줄러 종료
        self.scheduler.shutdown()
        logging.info("RetrainScheduler stopped")

# Example usage
if __name__ == '__main__':
    import argparse
    from datetime import timedelta
    from pipeline import run_pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    # 초기 설정 로드 및 재학습 함수 정의
    from config import load_config
    cfg = load_config(args.config)
    def retrain(): run_pipeline(cfg)

    # 스케줄러 실행 (매 24시간, 5회 이상 이벤트 시 10분 window)
    scheduler = RetrainScheduler(
        retrain_func=retrain,
        periodic_interval_hours=24,
        event_threshold=5,
        event_window_minutes=10
    )
    try:
        # 주기 실행 및 이벤트 기록
        while True:
            # 애플리케이션 로직에서 이상 발생 시 record_event 호출
            pass
    except KeyboardInterrupt:
        scheduler.shutdown()
