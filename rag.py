# ─────────────────────────────────────────────────────────────
# Product 3 — RAG Inference Optimizer
# Souvik Kundu · AI PM Master Curriculum
#
# What this file does:
# 1. Loads your resume and splits into chunks
# 2. Embeds chunks using sentence-transformers (free, local)
# 3. Stores embeddings in FAISS vector store
# 4. At query time: embeds JD, retrieves top 5 resume chunks
# 5. Runs three configurations simultaneously:
#    - Claude Sonnet (expensive baseline)
#    - Llama 3 8B on Groq (cheap, fast)
#    - Llama 3 8B on Groq + Cohere reranker (cheap, accurate)
# 6. Scores quality using Claude as LLM-as-judge
# ─────────────────────────────────────────────────────────────

import os
import time
import json
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import anthropic
from groq import Groq
import cohere
import wandb

load_dotenv()

# ─────────────────────────────────────────────────────────────
# SECTION 1: CHUNKING
#
# PM explanation: We split the resume into chunks of ~150-200
# tokens. Each chunk should be a meaningful unit -- a role,
# a set of bullets, or a section. This is the chunking
# strategy decision we discussed in Day 12.
#
# We use character count as proxy for token count:
# ~600 characters ≈ 150 tokens at typical density
# ─────────────────────────────────────────────────────────────

def chunk_resume(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """
    Split resume text into overlapping chunks.
    
    chunk_size: target characters per chunk (~150 tokens)
    overlap: characters shared between adjacent chunks
             prevents information loss at boundaries
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a natural boundary (newline or period)
        # rather than cutting mid-sentence
        if end < len(text):
            # Look for newline within last 100 chars of chunk
            break_point = text.rfind('\n', start + chunk_size - 100, end)
            if break_point == -1:
                # No newline found, try period
                break_point = text.rfind('.', start + chunk_size - 100, end)
            if break_point != -1:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start forward with overlap
        # Overlap ensures context is not lost at chunk boundaries
        start = end - overlap
    
    return chunks


# ─────────────────────────────────────────────────────────────
# SECTION 2: EMBEDDING AND FAISS INDEX
#
# PM explanation: We use sentence-transformers to convert
# each chunk to a 384-dimensional vector. This runs locally
# -- no API key, no cost. The model (all-MiniLM-L6-v2) is
# small but effective for semantic similarity tasks.
#
# FAISS stores all vectors and enables fast cosine similarity
# search. For a resume with ~20 chunks, this is instant.
# At production scale (millions of docs) you would use
# approximate nearest neighbor (ANN) indexing.
# ─────────────────────────────────────────────────────────────

class ResumeRAG:
    def __init__(self, resume_path: str):
        print("Initializing RAG pipeline...")
        
        # Load embedding model
        # all-MiniLM-L6-v2: 384 dimensions, fast, good quality
        # Downloads ~80MB on first run
        print("Loading embedding model (downloads ~80MB first run)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load API clients
        self.claude = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.cohere = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        
        # Load and chunk resume
        print("Chunking resume...")
        resume_text = Path(resume_path).read_text(encoding='utf-8')
        self.chunks = chunk_resume(resume_text)
        print(f"Created {len(self.chunks)} chunks from resume")
        
        # Build FAISS index
        print("Building FAISS index...")
        self._build_index()
        print("RAG pipeline ready.")
    
    def _build_index(self):
        """
        Embed all resume chunks and store in FAISS.
        
        FAISS IndexFlatIP uses inner product (dot product) for
        similarity. With normalized vectors this equals cosine
        similarity -- the standard for semantic search.
        """
        # Embed all chunks
        embeddings = self.embedder.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Normalize for cosine similarity
        # faiss.normalize_L2 divides each vector by its magnitude
        # After normalization: dot product = cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build index
        dimension = embeddings.shape[1]  # 384 for MiniLM
        self.index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
        self.index.add(embeddings)
        
        print(f"FAISS index built: {self.index.ntotal} vectors, {dimension} dimensions")
    
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve top K resume chunks most similar to the query.
        
        The query (job description) gets embedded into the same
        vector space as the resume chunks. FAISS finds the K
        chunks with highest cosine similarity.
        
        Returns list of {chunk, score} dicts sorted by relevance.
        """
        # Embed query
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 means no result found
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "chunk_index": int(idx)
                })
        
        return results
    
    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Use Cohere reranker to reorder chunks by relevance.
        
        PM explanation: Vector search finds semantically similar
        chunks. Reranking finds chunks that actually ANSWER the
        query. These are different -- a chunk can be semantically
        similar but not useful for answering.
        
        Cohere rerank-english-v3.0 is a cross-encoder model that
        scores each (query, chunk) pair together -- more accurate
        than embedding similarity alone. Adds ~50ms latency.
        """
        documents = [c["chunk"] for c in chunks]
        
        response = self.cohere.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=5
        )
        
        reranked = []
        for result in response.results:
            original_chunk = chunks[result.index]
            reranked.append({
                "chunk": original_chunk["chunk"],
                "score": result.relevance_score,
                "chunk_index": original_chunk["chunk_index"],
                "rerank_score": result.relevance_score
            })
        
        return reranked


# ─────────────────────────────────────────────────────────────
# SECTION 3: THREE GENERATION CONFIGURATIONS
#
# PM explanation: Same retrieved context, three different
# LLM configurations. This is the benchmark that proves
# you understand inference cost-quality tradeoffs.
#
# Config 1: Claude Sonnet -- expensive, high quality baseline
# Config 2: Llama 3 8B on Groq -- cheap, fast
# Config 3: Llama 3 8B on Groq + reranker -- cheap + accurate
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a career coach helping a job applicant 
prepare for interviews. Generate exactly 5 concise talking points 
that highlight the applicant's most relevant experience for the 
given job description.

CRITICAL RULES:
- Use ONLY information from the resume context provided
- Do not invent or assume any experience not explicitly stated
- Each talking point must be specific and quantified where possible
- Format as a numbered list 1-5
- Keep each point to 1-2 sentences maximum"""

def build_prompt(context_chunks: list[dict], job_description: str) -> str:
    """Build the augmented prompt with retrieved context."""
    context = "\n\n---\n\n".join([c["chunk"] for c in context_chunks])
    
    # Truncate JD to first 2000 characters -- captures the key requirements
    # without hitting token limits
    jd_truncated = job_description[:2000] + "..." if len(job_description) > 2000 else job_description
    
    return f"""RESUME CONTEXT (use only this information):
{context}

JOB DESCRIPTION:
{jd_truncated}

Generate 5 tailored talking points from the resume context above."""


def run_claude(rag: ResumeRAG, chunks: list[dict], jd: str) -> dict:
    """
    Config 1: Claude Sonnet
    Expensive but highest quality. Used as baseline AND as judge.
    """
    prompt = build_prompt(chunks, jd)
    
    start = time.perf_counter()
    
    response = rag.claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    
    latency_ms = (time.perf_counter() - start) * 1000
    
    # Calculate cost
    # claude-sonnet-4-5: $3/M input tokens, $15/M output tokens
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)
    
    return {
        "config": "Claude Sonnet",
        "answer": response.content[0].text,
        "latency_ms": round(latency_ms, 1),
        "cost_usd": round(cost, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }


def run_groq(rag: ResumeRAG, chunks: list[dict], jd: str) -> dict:
    """
    Config 2: Llama 3 8B on Groq
    Free tier, sub-200ms latency, much cheaper than Claude.
    """
    prompt = build_prompt(chunks, jd)
    
    start = time.perf_counter()
    
    response = rag.groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.1
    )
    
    latency_ms = (time.perf_counter() - start) * 1000
    
    # Groq pricing: ~$0.05/M input, $0.08/M output for Llama 3 8B
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (input_tokens * 0.05 / 1_000_000) + (output_tokens * 0.08 / 1_000_000)
    
    return {
        "config": "Llama 3 8B (Groq)",
        "answer": response.choices[0].message.content,
        "latency_ms": round(latency_ms, 1),
        "cost_usd": round(cost, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }


def run_groq_reranked(rag: ResumeRAG, base_chunks: list[dict], jd: str) -> dict:
    """
    Config 3: Llama 3 8B on Groq + Cohere Reranker
    Reranker improves chunk quality before sending to Llama.
    Small cost addition (~$0.001) for meaningful quality gain.
    """
    # Rerank the retrieved chunks
    rerank_start = time.perf_counter()
    reranked_chunks = rag.rerank(jd, base_chunks)
    rerank_latency = (time.perf_counter() - rerank_start) * 1000
    
    prompt = build_prompt(reranked_chunks, jd)
    
    start = time.perf_counter()
    
    response = rag.groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.1
    )
    
    llm_latency = (time.perf_counter() - start) * 1000
    total_latency = rerank_latency + llm_latency
    
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    # Cost: Groq LLM + Cohere rerank
    # Cohere rerank: $0.001 per 1000 docs (essentially free at this scale)
    llm_cost = (input_tokens * 0.05 / 1_000_000) + (output_tokens * 0.08 / 1_000_000)
    rerank_cost = 0.001 / 1000  # ~$0.000001 per rerank call
    total_cost = llm_cost + rerank_cost
    
    return {
        "config": "Llama 3 8B + Reranker (Groq)",
        "answer": response.choices[0].message.content,
        "latency_ms": round(total_latency, 1),
        "rerank_latency_ms": round(rerank_latency, 1),
        "llm_latency_ms": round(llm_latency, 1),
        "cost_usd": round(total_cost, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reranked_chunks": reranked_chunks
    }


# ─────────────────────────────────────────────────────────────
# SECTION 4: LLM-AS-JUDGE QUALITY SCORING
#
# PM explanation: We use Claude to score the quality of each
# configuration's output on a 0-100 scale.
#
# This is the eval methodology from Day 8 -- LLM-as-judge.
# Claude evaluates whether the talking points are:
# - Grounded in the resume context (faithfulness)
# - Relevant to the job description (relevance)
# - Specific and quantified (quality)
#
# Using Claude as judge for all three configs ensures
# consistent scoring. This is standard practice.
# ─────────────────────────────────────────────────────────────

def score_quality(rag: ResumeRAG, answer: str, chunks: list[dict], jd: str) -> dict:
    """
    Use Claude as LLM-as-judge to score answer quality.
    Returns score 0-100 with reasoning.
    """
    context = "\n\n---\n\n".join([c["chunk"] for c in chunks])
    
    judge_prompt = f"""You are evaluating the quality of AI-generated job interview talking points.

RESUME CONTEXT PROVIDED TO THE AI:
{context}

JOB DESCRIPTION:
{jd[:500]}...

AI-GENERATED TALKING POINTS:
{answer}

Score this response on a scale of 0-100 based on:
1. Faithfulness (0-40 points): Are all claims grounded in the resume context? 
   Deduct points for any invented or unsupported claims.
2. Relevance (0-30 points): Do the points address what the JD is looking for?
3. Specificity (0-30 points): Are the points specific with numbers and outcomes?

Respond with ONLY a JSON object in this exact format:
{{"score": 85, "faithfulness": 38, "relevance": 25, "specificity": 22, "reasoning": "Brief explanation"}}"""

    
    judge_start = time.perf_counter()
    response = rag.claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    judge_latency_ms = (time.perf_counter() - judge_start) * 1000
    
    try:
        raw = response.content[0].text.strip()
        # Remove markdown code blocks
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        
        # Extract just the numeric fields we need
        # More robust than full JSON parse
        import re
        score = int(re.search(r'"score":\s*(\d+)', raw).group(1))
        faithfulness = int(re.search(r'"faithfulness":\s*(\d+)', raw).group(1))
        relevance = int(re.search(r'"relevance":\s*(\d+)', raw).group(1))
        specificity = int(re.search(r'"specificity":\s*(\d+)', raw).group(1))
        
        return {
            "score": score,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "specificity": specificity,
            "reasoning": "See raw output"
            "judge_latency_ms": round(judge_latency_ms, 1)
        }
    except Exception as e:
        print(f"  Scoring parse error: {e}")
        print(f"  Raw response: {response.content[0].text[:300]}")
        return {"score": 0, "reasoning": f"Scoring failed: {e}", "judge_latency_ms": 0}


# ─────────────────────────────────────────────────────────────
# SECTION 5: MAIN BENCHMARK FUNCTION
#
# Orchestrates the full pipeline:
# 1. Retrieve top 5 chunks from resume
# 2. Run all three configs in sequence
# 3. Score each config with LLM-as-judge
# 4. Return comparison results
# ─────────────────────────────────────────────────────────────

def run_benchmark(rag: ResumeRAG, job_description: str) -> dict:
    """
    Run the full three-config benchmark for a job description.
    Returns structured results for the frontend to display.
    """
    print(f"\nRunning benchmark for JD ({len(job_description)} chars)...")
    
    # Step 1: Retrieve top 5 resume chunks
    print("Retrieving relevant resume chunks...")
    retrieve_start = time.perf_counter()
    chunks = rag.retrieve(job_description, k=5)
    retrieve_latency = (time.perf_counter() - retrieve_start) * 1000
    
    print(f"Retrieved {len(chunks)} chunks in {retrieve_latency:.1f}ms")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1} (score: {c['score']:.3f}): {c['chunk'][:60]}...")
    
    # Step 2: Run three configs
    print("\nRunning Config 1: Claude Sonnet...")
    claude_result = run_claude(rag, chunks, job_description)
    print(f"  Done: {claude_result['latency_ms']}ms, ${claude_result['cost_usd']:.6f}")
    
    print("Running Config 2: Llama 3 8B on Groq...")
    groq_result = run_groq(rag, chunks, job_description)
    print(f"  Done: {groq_result['latency_ms']}ms, ${groq_result['cost_usd']:.6f}")
    
    print("Running Config 3: Llama 3 8B + Reranker...")
    groq_reranked_result = run_groq_reranked(rag, chunks, job_description)
    print(f"  Done: {groq_reranked_result['latency_ms']}ms, ${groq_reranked_result['cost_usd']:.6f}")
    
    # Step 3: Score quality with LLM-as-judge
    print("\nScoring quality with LLM-as-judge...")
    claude_score = score_quality(rag, claude_result["answer"], chunks, job_description)
    groq_score = score_quality(rag, groq_result["answer"], chunks, job_description)
    groq_reranked_score = score_quality(rag, groq_reranked_result["answer"], chunks, job_description)
    
    # Step 4: Calculate cost savings vs Claude baseline
    claude_cost = claude_result["cost_usd"]
    groq_savings = round((1 - groq_result["cost_usd"] / claude_cost) * 100, 1) if claude_cost > 0 else 0
    groq_reranked_savings = round((1 - groq_reranked_result["cost_usd"] / claude_cost) * 100, 1) if claude_cost > 0 else 0
    
# ─────────────────────────────────────────────────────────────
# W&B LOGGING
# Log every benchmark run for production observability
# Non-blocking — W&B failure never breaks the API response
# ─────────────────────────────────────────────────────────────
try:
    wandb.init(
        project="rag-inference-optimizer",
        job_type="benchmark",
        reinit=True
    )

    for cfg in [
        ("claude_sonnet", claude_result, claude_score),
        ("llama_groq", groq_result, groq_score),
        ("llama_groq_reranker", groq_reranked_result, groq_reranked_score)
    ]:
        config_name, result, score = cfg
        wandb.log({
            f"{config_name}/retrieve_latency_ms": round(retrieve_latency, 1),
            f"{config_name}/generation_latency_ms": result["latency_ms"],
            f"{config_name}/judge_latency_ms": score.get("judge_latency_ms", 0),
            f"{config_name}/total_latency_ms": round(
                retrieve_latency + result["latency_ms"] + score.get("judge_latency_ms", 0), 1
            ),
            f"{config_name}/generation_cost_usd": result["cost_usd"],
            f"{config_name}/quality_score": score.get("score", 0),
            f"{config_name}/faithfulness": score.get("faithfulness", 0),
            f"{config_name}/relevance": score.get("relevance", 0),
            f"{config_name}/specificity": score.get("specificity", 0),
            f"{config_name}/input_tokens": result.get("input_tokens", 0),
            f"{config_name}/output_tokens": result.get("output_tokens", 0),
        })

    wandb.log({
        "retrieval/chunks_retrieved": len(chunks),
        "retrieval/latency_ms": round(retrieve_latency, 1),
        "retrieval/top_chunk_score": chunks[0]["score"] if chunks else 0,
    })

    wandb.finish()
    print("W&B run logged successfully")

except Exception as e:
    print(f"W&B logging failed (non-blocking): {e}")

    return {
        "retrieved_chunks": chunks,
        "retrieve_latency_ms": round(retrieve_latency, 1),
        "configs": [
            {
                **claude_result,
                "quality_score": claude_score.get("score", 0),
                "quality_breakdown": claude_score,
                "cost_savings_vs_baseline": "baseline"
            },
            {
                **groq_result,
                "quality_score": groq_score.get("score", 0),
                "quality_breakdown": groq_score,
                "cost_savings_vs_baseline": f"{groq_savings}% cheaper"
            },
            {
                **groq_reranked_result,
                "quality_score": groq_reranked_score.get("score", 0),
                "quality_breakdown": groq_reranked_score,
                "cost_savings_vs_baseline": f"{groq_reranked_savings}% cheaper"
            }
        ]
    }


# ─────────────────────────────────────────────────────────────
# QUICK TEST -- run this file directly to test the pipeline
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialize RAG with your resume
    rag = ResumeRAG("resume.txt")
    
    # Test with SambaNova JD excerpt
    test_jd = """
    Senior Product Manager - AI Cloud Infrastructure
    
    We are looking for a PM with deep hardware-software fluency
    to own our AI cloud product roadmap. Requirements:
    - 5+ years PM experience in AI infrastructure or silicon
    - Understanding of inference optimization and unit economics
    - Experience with enterprise GTM and OEM partnerships
    - Strong analytical skills with P&L ownership experience
    - Ability to work cross-functionally with engineering teams
    """
    
    results = run_benchmark(rag, test_jd)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    for config in results["configs"]:
        print(f"\n{config['config']}")
        print(f"  Latency:  {config['latency_ms']}ms")
        print(f"  Cost:     ${config['cost_usd']:.6f}")
        print(f"  Quality:  {config['quality_score']}/100")
        print(f"  Savings:  {config['cost_savings_vs_baseline']}")
        print(f"  Answer preview: {config['answer'][:150]}...")