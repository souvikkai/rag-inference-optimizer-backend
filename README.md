# RAG Inference Optimizer — Backend

FastAPI backend for the RAG Inference Optimizer. Benchmarks three inference 
configurations head-to-head on a resume-to-JD matching task, scored by 
LLM-as-judge. Live at: https://rag-inference-optimizer.vercel.app

## What It Does

Takes a job description → retrieves relevant resume chunks via FAISS vector 
search → runs three parallel inference configurations → scores quality with 
LLM-as-judge → returns comparison results to the frontend.

## Three Configurations Benchmarked

| Configuration | Model | Purpose |
|---|---|---|
| Baseline | Claude Sonnet | Expensive, high quality reference |
| Fast | Llama 3.1 8B (Groq LPU) | Cheap, sub-second latency |
| Optimized | Llama 3.1 8B + Cohere Reranker (Groq) | Cheap + reranked retrieval |

**Judge:** Claude Sonnet 4.6 scores all three on faithfulness, relevance, 
and specificity.

## Architecture
PDF/DOCX resume upload
→ chunk_resume() [600 char chunks, 100 char overlap]
→ SentenceTransformer all-MiniLM-L6-v2 [384-dim embeddings]
→ FAISS IndexFlatIP [cosine similarity search]
→ top-5 chunks retrieved per JD query
→ three parallel inference paths
→ LLM-as-judge scoring
→ JSON response to Next.js frontend

## Stack

- **API:** FastAPI on Railway
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2, local, free)
- **Vector store:** FAISS (IndexFlatIP, cosine similarity)
- **Inference:** Anthropic SDK (Claude Sonnet), Groq SDK (Llama 3.1 8B)
- **Reranker:** Cohere rerank-english-v3.0
- **Observability:** Weights & Biases (latency, cost, quality per run)
- **Judge:** Claude Sonnet 4.6

## Benchmark Results (empirical)

| Configuration | Quality | Faithfulness | Cost/Query | Latency |
|---|---|---|---|---|
| Claude Sonnet | 95/100 | 38/40 | $0.012537 | 8,235ms |
| Llama 3.1 8B (Groq) | 78/100 | 34/40 | $0.000130 | 851ms |
| Llama + Reranker (Groq) | 62/100 | 24/40 | $0.000132 | 2,196ms |

**Key finding:** Groq is 99% cheaper than Claude Sonnet at 82% quality 
retention. The Cohere reranker unexpectedly hurt performance — faithfulness 
dropped from 34 to 24 because the reranker optimizes for JD keyword match 
rather than resume evidence strength.

## W&B Observability

Every benchmark run is logged to Weights & Biases tracking:
- Retrieval latency (FAISS vector search)
- Generation latency per model
- Judge latency (Claude Sonnet scoring)
- Total end-to-end latency
- Cost per configuration
- Quality score, faithfulness, relevance, specificity
- Token counts (input/output)

**PM metric to alert on first:** `faithfulness` — if this drops below 30/40, 
the model is hallucinating resume content. Quality score is lagging; 
faithfulness is the leading signal.

## Reranker Finding and Next Steps

Root cause of reranker underperformance:
- Reranks only 5 candidates (insufficient pool)
- Uses raw JD as query (optimizes keyword match, not evidence strength)
- Collapses chunk diversity — all 5 chunks cover same JD theme

Planned fixes:
1. Expand candidate pool to k=20 before reranking
2. Replace raw JD query with evidence-focused reranker query
3. Add MMR diversity scoring to preserve coverage across experience types

## Environment Variables
ANTHROPIC_API_KEY=
GROQ_API_KEY=
COHERE_API_KEY=
WANDB_API_KEY=

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
