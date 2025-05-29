from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd

from config import load_config
from reporting.report_automation import ReportGenerator, run_dashboard

# DAG 기본 설정
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    dag_id='reporting_and_dashboard',
    default_args=default_args,
    description='월간 리포트 생성 및 대시보드 실행',
    schedule_interval='@monthly',
    catchup=False,
    max_active_runs=1,
    tags=['reporting','dashboard']
) as dag:

    def generate_monthly_reports(**ctx):
        cfg = load_config('/path/to/config.yaml')
        # 이상 탐지 요약 데이터 로드
        df_anom = pd.read_csv(cfg['output_csv'], index_col=0)
        df_seq  = pd.read_csv(cfg['sequence_output_csv'], index_col=0) if cfg.get('enable_sequence_detection') else pd.DataFrame()
        df_cls  = pd.read_csv(cfg['classification_output_csv'], index_col=0) if cfg.get('enable_classification') else pd.DataFrame()

        # 요약 데이터프레임 생성
        summary = pd.DataFrame({
            'total_anomalies': [len(df_anom)],
            'sequence_anomalies': [len(df_seq)],
            'classified_risks': [df_cls['prediction'].sum() if not df_cls.empty else 0]
        })

        gen = ReportGenerator(template_pptx=cfg.get('pptx_template'))
        # PPTX/Excel 리포트 생성
        gen.generate_pptx(summary, cfg.get('monthly_pptx', 'monthly_report.pptx'))
        gen.generate_excel(summary, cfg.get('monthly_excel', 'monthly_report.xlsx'))

    def launch_dashboard(**ctx):
        cfg = load_config('/path/to/config.yaml')
        run_dashboard(cfg.get('sequence_output_csv', 'sequence_anomalies.csv'))

    task_generate = PythonOperator(
        task_id='generate_monthly_reports',
        python_callable=generate_monthly_reports
    )

    task_dashboard = PythonOperator(
        task_id='launch_streamlit_dashboard',
        python_callable=launch_dashboard
    )

    task_generate >> task_dashboard