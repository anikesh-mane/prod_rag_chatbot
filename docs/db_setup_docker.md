# 🚀 Docker Setup Guide: Redis, PostgreSQL, and Milvus

This document contains complete Docker commands to set up:

- Redis
- PostgreSQL
- Milvus (with dependencies)

All setups include:
- Persistent storage
- Secure passwords
- Host networking option (for Lightning / VM environments)

---

# 🟥 1️⃣ Redis Setup

## 🔹 Pull Image

```bash
docker pull redis:8.4-alpine
```

## 🔹 Run Redis (Host Networking - Recommended for VM/Lightning)

```bash
docker run -d \
  --name redis-server \
  --network host \
  redis:8.4-alpine \
  redis-server \
  --requirepass StrongPassword123 \
  --bind 0.0.0.0
```

## 🔹 Verify

```bash
ss -lntp | grep 6379
```
or
```python
import redis

# Update these values
HOST = "localhost"
PORT = 6379
PASSWORD = "StrongPassword123"  # or None if no password

r = redis.Redis(
    host=HOST,
    port=PORT,
    password=PASSWORD,
    decode_responses=True
)

# Test connection
print("Ping:", r.ping())

# Get Redis configuration to know how many DBs exist
config = r.config_get("databases")
num_dbs = int(config["databases"])

print("\nAvailable Logical Databases:")
for i in range(num_dbs):
    print("-", i)
```

## 🔹 Connect

```bash
redis-cli -h localhost -p 6379 -a StrongPassword123
```

---

# 🟦 2️⃣ PostgreSQL Setup

## 🔹 Pull Image

```bash
docker pull postgres:18-alpine3.22
```

## 🔹 Create Persistent Volume

```bash
docker volume create pgdata
```

## 🔹 Run PostgreSQL (Host Networking)

```bash
docker run -d \
  --name postgres-db \
  --network host \
  -e POSTGRES_USER=ragbot \
  -e POSTGRES_PASSWORD=StrongPassword123 \
  -e POSTGRES_DB=ragbot \
  -v pgdata:/var/lib/postgresql/data \
  postgres:18-alpine3.22
```

## 🔹 Verify

```bash
ss -lntp | grep 5432
```
or
```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://ragbot:StrongPassword123@localhost:5432/postgres"

async def main():
    engine = create_async_engine(DATABASE_URL)

    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT datname FROM pg_database WHERE datistemplate = false;")
        )
        
        print("Databases:")
        for row in result:
            print("-", row[0])

    await engine.dispose()

asyncio.run(main())
```

## 🔹 Connect

```bash
psql -h localhost -U ragbot -d mydb
```

---

# 🟨 3️⃣ Milvus Setup (Standalone)

Milvus requires:
- etcd
- MinIO
- Milvus service

We use Docker Compose for clean setup.

## 🔹 Create docker-compose.yml

```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.25
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2024-12-18T13-15-44Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.6.11
    command: ["milvus", "run", "standalone"]
    security_opt:
    - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MQ_TYPE: woodpecker
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "bash", "-c", "echo > /dev/tcp/localhost/19530"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

## 🔹 Start Milvus

```bash
docker compose up -d
```

## 🔹 Verify

```bash
ss -lntp | grep 19530
```
or
```python
from pymilvus import connections, db

# Connect to Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# List databases
databases = db.list_database()

print("Databases:")
for database in databases:
    print("-", database)

```

Milvus default port: **19530**

---

# 🔐 Security Recommendations

- Always use strong passwords
- Avoid exposing database ports publicly
- Prefer host networking only inside secured VM
- For production: use private VPC networking

---

# 📌 Default Ports Summary

| Service      | Default Port |
|-------------|--------------|
| Redis       | 6379         |
| PostgreSQL  | 5432         |
| Milvus      | 19530        |
| etcd        | 2379         |
| MinIO       | 9000         |

---

# ✅ Quick Health Check Commands

```bash
docker ps
```

```bash
docker logs <container-name>
```

```bash
ss -lntp
```

---

# 🎯 Notes

- Use `--network host` for Lightning / cloud VM environments
- Use Docker bridge networking for local development
- Always attach volumes for persistence
- For production, consider managed database services

---

🚀 Setup Complete

