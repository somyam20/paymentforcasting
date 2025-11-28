import psycopg2
import psycopg2.extras
from config.settings import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

def get_conn():
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
        raise RuntimeError("Database configuration incomplete")
    
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_name TEXT PRIMARY KEY,
            s3_url TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS project_aliases (
            project_name TEXT,
            customer_key TEXT,
            alias TEXT,
            PRIMARY KEY (project_name, customer_key)
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()

def save_project(project_name: str, s3_url: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO projects (project_name, s3_url) VALUES (%s, %s) ON CONFLICT (project_name) DO UPDATE SET s3_url = EXCLUDED.s3_url",
        (project_name, s3_url),
    )
    conn.commit()
    cur.close()
    conn.close()

def get_project_url(project_name: str) -> str | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT s3_url FROM projects WHERE project_name = %s", (project_name,))
    r = cur.fetchone()
    cur.close()
    conn.close()
    return r[0] if r else None

def upsert_alias(project_name: str, customer_key: str, alias: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO project_aliases (project_name, customer_key, alias) VALUES (%s,%s,%s) ON CONFLICT (project_name, customer_key) DO UPDATE SET alias = EXCLUDED.alias",
        (project_name, customer_key, alias),
    )
    conn.commit()
    cur.close()
    conn.close()

def get_alias(project_name: str, customer_key: str) -> str | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT alias FROM project_aliases WHERE project_name = %s AND customer_key = %s",
        (project_name, customer_key),
    )
    r = cur.fetchone()
    cur.close()
    conn.close()
    return r[0] if r else None