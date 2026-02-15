# Multilingual RAG Chatbot for Channel Sales  
**Trying to build VIBE CODED Prod grade LLM System with Feedback-Driven Learning, Cost Control & Zero-Downtime Embedding Migration**

[Roadmap](docs/nextsteps.md)
[DB Setup](docs/db_setup_docker.md)

---

## 1. Project Overview

This project implements a **production-grade conversational AI system** for **Channel Sales and Operations**.  
The chatbot answers questions over internal documents (pricing, partner policies, sales playbooks, tickets, etc.), supports **multiple languages**, and continuously improves via **user feedback-driven fine-tuning**.

The system is designed to mirror a **real enterprise deployment**:

- Retrieval-Augmented Generation (RAG)  
- Multilingual support  
- Token-level cost tracking  
- Low-latency API serving  
- Embedding versioning and backfill  
- Feedback loop for model improvement  
- Cloud-deployable microservices  

---

## 2. High-Level Architecture

```
User (Any Language)
|
Language Detection
|
Translation (if needed)
|
Query Embedding
|
Vector Search (FAISS / Pinecone)
|
Top-K Chunks
|
LLM (GPT / Mistral)
|
Post-processing & Validation
|
Translate Back
|
Final Answer
|
Feedback Collection
|
Training Dataset
```

---

## 3. Core Capabilities

### 3.1 Retrieval-Augmented Generation (RAG)

- Documents are chunked and embedded  
- Queries retrieve top-k relevant chunks  
- LLM answers grounded on retrieved context  

---

### 3.2 Multilingual Chat

- Automatic language detection  
- Translate queries to English  
- Translate answers back to user language  

---

### 3.3 Cost & Latency Awareness

Tracked per request:

- Input tokens  
- Output tokens  
- Embedding calls  
- LLM calls  
- Latency per stage 
- Cache common embeddings and responses to reduce cost 

Optimized via:

- Query embedding caching  
- LLM response caching  
- Model routing (GPT-4 → GPT-3.5)  

---

### 3.4 Continuous Learning

Captured per interaction:
- prompt
- retrieved_documents
- generated_answer
- user_rating

Used for:

- Supervised fine-tuning (SFT)  
- Preference learning (DPO)  

---

### 3.5 Zero-Downtime Embedding Migration

- Multiple embedding versions  
- Dual indexing  
- Gradual traffic shifting  
- Background backfilling  

---

### 3.6 Enterprise Reliability

Fallbacks:

- GPT-4 → GPT-3.5  
- Vector search → keyword search  
- LLM → cached answer  

---

## 4. Technology Stack

### Backend

| Component | Technology |
|--------|-------------|
| API | FastAPI |
| Auth | JWT / API Key |
| Async | Uvicorn |
| Tracing | OpenTelemetry |

---

### LLM & NLP

| Purpose | Tool |
|------|------|
| LLM | GPT-4 / Mistral |
| Embeddings | OpenAI / BGE / E5 |
| Translation | OpenAI / MarianMT |
| Lang Detection | fastText |

---

### Storage

| Data | Technology |
|------|-----------|
| Vectors | FAISS / Pinecone |
| Metadata | PostgreSQL |
| Feedback | PostgreSQL |
| Cache | Redis |

---

### MLOps

| Area | Tool |
|------|------|
| Model Registry | MLflow |
| Training | Weights & Biases |
| Metrics | Prometheus |
| Dashboards | Grafana |
| CI/CD | GitHub Actions |
| Deployment | Docker, AWS ECS / GCP Cloud Run |

---

## 5. Repository Structure
```
rag-chatbot/
│
├── ingestion/
│   ├── loaders.py
│   ├── chunker.py
│   ├── embedder.py
│   └── index_builder.py
│
├── api/
│   ├── main.py
│   ├── routes/
│   │   ├── chat.py
│   │   ├── feedback.py
│   │   └── metrics.py
│   └── middleware.py
│
├── retrieval/
│   ├── vector_store.py
│   ├── reranker.py
│   └── fallback.py
│
├── llm/
│   ├── prompt_templates.py
│   ├── generator.py
│   └── postprocessor.py
│
├── training/
│   ├── dataset_builder.py
│   ├── sft.py
│   └── dpo.py
│
├── monitoring/
│   ├── metrics.py
│   └── dashboards/
│
├── configs/
│   ├── models.yaml
│   ├── embedding_versions.yaml
│   └── cost_limits.yaml
│
└── docker-compose.yml
```

---

## 6. Query Flow

1. User sends query  
2. Language detected  
3. Translated to English  
4. Query embedding generated  
5. Vector search  
6. Prompt built  
7. LLM generates answer  
8. Post-processing  
9. Translate back  
10. Return answer  
11. Store logs & feedback  

---

## 7. Metrics

### System
- Latency per stage  
- Requests/sec  
- Cache hit rate  
- Vector DB query time

### Model
- Cost per query  
- Token usage  
- Win-rate from feedback  
- Hallucination rate  

---

## 8. Feedback Loop

Users rate responses:

- 👍 Useful  
- 👎 Incorrect  

Stored as:

(prompt, retrieved_docs, answer, rating)

Used for :
- Supervised fine-tuning
- Preference learning (DPO)
- Prompt refinement

---

## 9. Embedding Versioning

Each chunk stores:

chunk_id
text
embedding_v1
embedding_v2

Retrieval router supports:
- 100% v1
- Shadow traffic to v2
- Gradual cutover
- Traffic can be routed gradually from v1 → v2.

---