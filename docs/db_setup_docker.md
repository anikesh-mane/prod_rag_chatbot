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
docker pull redis:7
```

## 🔹 Create Persistent Volume

```bash
docker volume create redisdata
```

## 🔹 Run Redis (Host Networking - Recommended for VM/Lightning)

```bash
docker run -d \
  --name redis-server \
  --network host \
  -v redisdata:/data \
  redis:7 \
  redis-server \
  --requirepass StrongPassword123 \
  --appendonly yes \
  --bind 0.0.0.0
```

## 🔹 Verify

```bash
ss -lntp | grep 6379
```

## 🔹 Connect

```bash
redis-cli -h localhost -p 6379 -a StrongPassword123
```

---

# 🟦 2️⃣ PostgreSQL Setup

## 🔹 Pull Image

```bash
docker pull postgres:16
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
  -e POSTGRES_USER=anikesh \
  -e POSTGRES_PASSWORD=StrongPassword123 \
  -e POSTGRES_DB=mydb \
  -v pgdata:/var/lib/postgresql/data \
  postgres:16
```

## 🔹 Verify

```bash
ss -lntp | grep 5432
```

## 🔹 Connect

```bash
psql -h localhost -U anikesh -d mydb
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
version: '3.9'

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: milvus-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    volumes:
      - ./volumes/etcd:/etcd
    network_mode: host

  minio:
    image: minio/minio:latest
    container_name: milvus-minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data
    volumes:
      - ./volumes/minio:/minio_data
    network_mode: host

  milvus:
    image: milvusdb/milvus:v2.3.9
    container_name: milvus-standalone
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: localhost:2379
      MINIO_ADDRESS: localhost:9000
    volumes:
      - ./volumes/milvus:/var/lib/milvus
    depends_on:
      - etcd
      - minio
    network_mode: host
```

## 🔹 Start Milvus

```bash
docker compose up -d
```

## 🔹 Verify

```bash
ss -lntp | grep 19530
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

