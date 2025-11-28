import asyncpg
from config.settings import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

async def get_conn():
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
        raise RuntimeError("Database configuration incomplete")
    
    return await asyncpg.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

async def init_db():
    conn = await get_conn()
    await conn.execute(
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
    await conn.close()

async def save_project(project_name: str, s3_url: str):
    conn = await get_conn()
    await conn.execute(
        "INSERT INTO projects (project_name, s3_url) VALUES ($1, $2) ON CONFLICT (project_name) DO UPDATE SET s3_url = EXCLUDED.s3_url",
        project_name, s3_url
    )
    await conn.close()

async def get_project_url(project_name: str) -> str | None:
    conn = await get_conn()
    r = await conn.fetchrow("SELECT s3_url FROM projects WHERE project_name = $1", project_name)
    await conn.close()
    return r['s3_url'] if r else None

async def upsert_alias(project_name: str, customer_key: str, alias: str):
    conn = await get_conn()
    await conn.execute(
        "INSERT INTO project_aliases (project_name, customer_key, alias) VALUES ($1,$2,$3) ON CONFLICT (project_name, customer_key) DO UPDATE SET alias = EXCLUDED.alias",
        project_name, customer_key, alias
    )
    await conn.close()

async def get_alias(project_name: str, customer_key: str) -> str | None:
    conn = await get_conn()
    r = await conn.fetchrow(
        "SELECT alias FROM project_aliases WHERE project_name = $1 AND customer_key = $2",
        project_name, customer_key
    )
    await conn.close()
    return r['alias'] if r else None