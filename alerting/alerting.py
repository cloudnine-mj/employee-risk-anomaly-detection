# AlertmanagerClient: Alertmanager HTTP API로 알림 전송
# FlappingSuppressor: 짧은 시간 내 반복 알림 억제
# Deduplicator: 지정 윈도우 내 중복 알림 억제
# MuteList: 특정 라벨 조합 알림 완전 억제
# filter_alerts(): 위 3단계 억제 로직 순차 적용


import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any

class AlertmanagerClient:
    # Prometheus Alertmanager에 알림을 보내는 클라이언트
    def __init__(self, base_url: str, timeout: int = 5):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logging.info(f"AlertmanagerClient initialized with URL: {self.base_url}")

    def send_alerts(self, alerts: List[Dict[str, Any]]):
        # Alertmanager에 알림 목록을 보내기. 각 알림 사전은 Alertmanager API 사양을 따라야 함.
        if not alerts:
            logging.info("No alerts to send.")
            return
        url = f"{self.base_url}/api/v1/alerts"
        try:
            resp = requests.post(url, json=alerts, timeout=self.timeout)
            resp.raise_for_status()
            logging.info(f"Sent {len(alerts)} alerts to Alertmanager")
        except Exception as e:
            logging.error(f"Failed to send alerts to Alertmanager: {e}")
            raise

class FlappingSuppressor:
    def __init__(self, window: timedelta, threshold: int):
        self.window = window
        self.threshold = threshold
        self.history: Dict[str, List[datetime]] = {}

    def should_suppress(self, alert_id: str) -> bool:
        now = datetime.utcnow()
        times = self.history.get(alert_id, [])
        # keep only within window
        times = [t for t in times if now - t <= self.window]
        times.append(now)
        self.history[alert_id] = times
        if len(times) > self.threshold:
            logging.warning(f"Suppressing flapping alert '{alert_id}': {len(times)} occurrences within {self.window}")
            return True
        return False

class Deduplicator:
    def __init__(self, dedup_window: timedelta):
        self.dedup_window = dedup_window
        self.last_seen: Dict[str, datetime] = {}

    def is_duplicate(self, alert_id: str) -> bool:
        now = datetime.utcnow()
        last = self.last_seen.get(alert_id)
        if last and now - last <= self.dedup_window:
            logging.info(f"Duplicate alert '{alert_id}' suppressed (last at {last})")
            return True
        self.last_seen[alert_id] = now
        return False

class MuteList:
    def __init__(self, mutes: List[Dict[str, Any]]):
        self.mutes = mutes

    def is_muted(self, labels: Dict[str, Any]) -> bool:
        for mute in self.mutes:
            if all(labels.get(k) == v for k, v in mute.items()):
                logging.info(f"Muted alert with labels {labels}")
                return True
        return False

def filter_alerts(
    alerts: List[Dict[str, Any]],
    flapper: FlappingSuppressor,
    deduper: Deduplicator,
    mute_list: MuteList
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for alert in alerts:
        labels = alert.get('labels', {})
        alert_id = labels.get('alertname', '')
        if mute_list.is_muted(labels):
            continue
        if deduper.is_duplicate(alert_id):
            continue
        if flapper.should_suppress(alert_id):
            continue
        filtered.append(alert)
    return filtered


if __name__ == '__main__':
    # Setup
    base_url = 'http://alertmanager.local:9093'
    client = AlertmanagerClient(base_url)
    flapper = FlappingSuppressor(timedelta(minutes=5), threshold=3)
    deduper = Deduplicator(timedelta(minutes=10))
    mutes = MuteList([{'alertname':'RiskAnomaly','severity':'info'}])

    alerts = [
        {'labels':{'alertname':'RiskAnomaly','severity':'warning'}, 'annotations':{'desc':'...'}, 'startsAt':datetime.utcnow().isoformat()+'Z'}
    ]
    filtered = filter_alerts(alerts, flapper, deduper, mutes)
    client.send_alerts(filtered)