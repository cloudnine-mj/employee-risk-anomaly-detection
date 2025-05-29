from prometheus_client import start_http_server, Counter, Histogram, Gauge
from datetime import datetime
import logging
import sys

# Prometheus
REQUEST_LATENCY = Histogram(
    'risk_detection_latency_seconds',
    'Time taken for risk detection pipeline'
)
ERROR_COUNT = Counter(
    'risk_detection_errors_total',
    'Total number of errors in detection pipeline'
)
ANOMALIES_TOTAL = Counter(
    'risk_detection_anomalies_total',
    'Total number of anomalies detected'
)
LAST_RUN = Gauge(
    'risk_detection_last_run_timestamp',
    'Unix timestamp of last successful run'
)

def start_metrics_server(port: int = 8000):
    """
    Start the Prometheus HTTP metrics server on the given port.
    """
    # Configure a simple logger to print start message
    handler = logging.StreamHandler(sys.stdout)
    logging.getLogger().addHandler(handler)
    start_http_server(port)
    logging.info(f"Prometheus metrics server started on port {port}")

def instrumented(func):
    """
    Decorator to instrument a function with Prometheus metrics.
    Measures execution time, error count, anomaly count, and last run timestamp.
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.utcnow()
        try:
            result = func(*args, **kwargs)
        except Exception:
            ERROR_COUNT.inc()
            logging.exception("Error in instrumented function")
            raise
        duration = (datetime.utcnow() - start_time).total_seconds()
        REQUEST_LATENCY.observe(duration)
        LAST_RUN.set_to_current_time()
        # If result is a DataFrame with 'anomaly' column, count anomalies
        try:
            anomaly_count = int(result['anomaly'].sum())
            ANOMALIES_TOTAL.inc(anomaly_count)
        except Exception:
            pass
        return result
    return wrapper

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Start metrics server')
    parser.add_argument('--port', type=int, default=8000, help='Metrics HTTP server port')
    args = parser.parse_args()
    start_metrics_server(args.port)
    print("Metrics server is running. Press Ctrl+C to stop.")
