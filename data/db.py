import pandas as pd
import sqlalchemy
from datetime import datetime

class DBClient:
    """
    직원 활동 로그를 가져오는 데이터베이스 클라이언트
    사용법:
        db = DBClient(db_uri)
        df = db.fetch(start_datetime, end_datetime)
    """
    def __init__(self, db_uri: str):
        # Initialize SQLAlchemy engine
        self.engine = sqlalchemy.create_engine(db_uri)

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        query = (
            """
            SELECT
                user_id,
                login_time,
                action_type,
                resource_id,
                amount
            FROM employee_activity_logs
            WHERE login_time BETWEEN :start AND :end
            """
        )
        df = pd.read_sql(
            query,
            con=self.engine,
            params={'start': start, 'end': end}
        )
        return df

if __name__ == '__main__':
    from datetime import datetime, timedelta
    import argparse

    parser = argparse.ArgumentParser(description='Fetch employee activity logs')
    parser.add_argument('--db-uri', required=True, help='SQLAlchemy DB URI')
    parser.add_argument('--days', type=int, default=1, help='Lookback window in days')
    args = parser.parse_args()

    end = datetime.utcnow()
    start = end - timedelta(days=args.days)
    db = DBClient(args.db_uri)
    df = db.fetch(start, end)
    print(df.head())