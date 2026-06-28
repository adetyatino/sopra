import os
import io
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sqlalchemy import create_engine

# Perubahan: Menambahkan default_args untuk manajemen retry otomatis 
# dan catchup=False untuk mencegah backfill tidak diinginkan
default_args = {'retries': 1}

with DAG(
    'daily_orders_etl',
    default_args=default_args,
    schedule_interval='0 2 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    def extract(**kwargs):
        # Perubahan: Menggunakan to_json() agar data bisa dilewatkan antar task via XCom 
        # (Script 1 tidak memiliki mekanisme transfer data)
        orders = [
            {'order_id': 'ORD-001', 'customer_id': 'C100', 'amount': 100.0},
            {'order_id': 'ORD-002', 'customer_id': 'C101', 'amount': 0.0},
            {'order_id': 'ORD-001', 'customer_id': 'C100', 'amount': 100.0},
        ]
        df = pd.DataFrame(orders)
        return df.to_json()

    def transform(**kwargs):
        # Perubahan: Menambahkan xcom_pull untuk menerima data dari task 'extract'
        ti = kwargs['ti']
        json_data = ti.xcom_pull(task_ids='extract')
        orders_df = pd.read_json(io.StringIO(json_data))
        
        # Perubahan: Menambahkan validasi file path agar tidak error saat file tidak ditemukan
        file_path = '/opt/airflow/data/customers.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan di {file_path}")
            
        customers = pd.read_csv(file_path)
        
        # Perubahan: Menggunakan how='left' untuk audit (menjaga semua baris order) 
        # dan suffixes untuk menangani kolom duplikat agar tidak error
        result = orders_df.merge(customers, on='customer_id', how='left', suffixes=('', '_drop'))
        
        # Perubahan: Membersihkan kolom duplikat hasil merge agar data bersih untuk keperluan audit
        cols_to_drop = [col for col in result.columns if '_drop' in col]
        result = result.drop(columns=cols_to_drop)
        
        return result.to_json()

    def load_db(**kwargs):
        # Perubahan: Menambahkan penanganan jika data kosong dan menggunakan koneksi DB yang sesuai standar
        ti = kwargs['ti']
        json_data = ti.xcom_pull(task_ids='transform')
        if not json_data:
            return
        df = pd.read_json(io.StringIO(json_data))
        
        # Perubahan: Menggunakan if_exists='replace' agar data audit selalu update dan bersih tanpa duplikasi akumulatif
        engine = create_engine('postgresql+psycopg2://airflow:airflow@postgres/airflow')
        df.to_sql('orders', engine, if_exists='replace', index=False)

    # Perubahan: Menggunakan sintaks bitshift (>>) yang merupakan standar Airflow untuk mendefinisikan dependency task
    PythonOperator(task_id='extract', python_callable=extract) >> \
    PythonOperator(task_id='transform', python_callable=transform) >> \
    PythonOperator(task_id='load', python_callable=load_db)