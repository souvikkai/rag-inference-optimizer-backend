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
#    - Claude Haiku (generation baseline)
#    - Llama 3 8B on Groq (cheap, fast)
#    - Llama 3 8B on Groq + evidence-aware rerank & diverse context selection
# 6. Scores quality using Claude Sonnet as LLM-as-judge only
# ─────────────────────────────────────────────────────────────

import os
import time
import json
import re
import unicodedata
from typing import Literal
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
# INPUT CLEANING (JD + resume before retrieval / chunking)
# ─────────────────────────────────────────────────────────────

_URL_RE = re.compile(
    r"https?://[^\s<>\[\](){}\"']+|www\.[^\s<>\[\](){}\"']+",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_JD_HEADER_KEYS = frozenset(
    {
        "responsibilities",
        "requirements",
        "qualifications",
        "skills",
        "nice to have",
        "what you'll do",
        "about the role",
    }
)
# Longest-first so multi-word headers match before single-word suffixes.
_JD_HEADER_ORDERED = (
    "nice to have",
    "what you'll do",
    "about the role",
    "responsibilities",
    "requirements",
    "qualifications",
    "skills",
)


def _normalize_apostrophe(s: str) -> str:
    return (
        s.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("`", "'")
        .strip()
    )


def _strip_urls_emails(text: str) -> str:
    s = _URL_RE.sub(" ", text)
    s = _EMAIL_RE.sub(" ", s)
    return s


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_repeated_symbols(text: str) -> str:
    """Collapse PDF-ish noise: repeated bullets, underscores, dashes, etc."""
    s = re.sub(r"([*•·\-_=□■])\1{2,}", " ", text)
    s = re.sub(r"([^\w\s])\1{2,}", " ", s)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    return s


def _jd_header_key_from_line(line: str) -> str | None:
    s = line.strip()
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"\s*:?\s*$", "", s)
    key = _normalize_apostrophe(s).lower()
    if key in _JD_HEADER_KEYS:
        return key
    for h in _JD_HEADER_ORDERED:
        if key == h or key.endswith(" " + h):
            return h
    return None


def _jd_foreign_section_heading(line: str) -> bool:
    """Non-target section title (e.g. About Us:) — stop capturing until the next known header."""
    if _jd_header_key_from_line(line) is not None:
        return False
    s = line.strip()
    if not s:
        return False
    if re.match(r"^[\-\*•]\s", s):
        return False
    if re.match(r"^#+\s*\S", s):
        return True
    if len(s) <= 90 and s.endswith(":") and len(s.split()) <= 10:
        return True
    return False


def _jd_keep_named_sections(text: str) -> str:
    """
    If known section headers appear as lines, keep only those sections.
    Otherwise return text unchanged (caller already normalized newlines).
    """
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[str] = []
    current: list[str] = []
    active = False
    saw_header = False

    for line in lines:
        hdr = _jd_header_key_from_line(line)
        if hdr is not None:
            saw_header = True
            if active and current:
                blocks.append("\n".join(current).strip())
            current = []
            active = True
        elif active:
            if _jd_foreign_section_heading(line):
                if current:
                    blocks.append("\n".join(current).strip())
                current = []
                active = False
            else:
                current.append(line)

    if active and current:
        blocks.append("\n".join(current).strip())

    if not saw_header:
        return text

    merged = "\n\n".join(b for b in blocks if b.strip())
    return merged if merged.strip() else text


def _clean_job_description(text: str) -> str:
    s = _strip_urls_emails(text)
    s = _remove_repeated_symbols(s)
    s = _jd_keep_named_sections(s)
    s = _collapse_whitespace(s)
    if len(s) > 5000:
        s = s[:5000]
    return s


_RESUME_CONTACT_LINE = re.compile(
    r"(?i)\b(?:linkedin\.com|twitter\.com|x\.com|medium\.com|calendly)\b"
)


def _resume_keep_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _RESUME_CONTACT_LINE.search(s):
        return False
    if re.match(
        r"(?i)^[\s\-•]{0,4}(repository|repo|demo|live\s+demo|source\s+code|staging\s+url|production\s+url)\s*:",
        s,
    ):
        return False
    if re.match(r"(?i)^\s*(?:https?://|www\.)", s):
        return False
    if re.match(r"(?i)^\s*(github\.com|gitlab\.com|bitbucket\.org)/[\w./-]+\s*$", s):
        return False
    if re.match(r"^[\d\s\-+()./]{10,}$", s):
        return False
    if re.search(r"(?i)^(phone|tel|mobile|e-?mail)\s*:", s):
        return False
    if "www." in s.lower():
        return False
    return True


def _clean_resume_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    s = _strip_urls_emails(s)
    s = _remove_repeated_symbols(s)
    lines = s.replace("\r\n", "\n").split("\n")
    kept = [ln for ln in lines if _resume_keep_line(ln)]
    s = "\n".join(kept)
    s = _collapse_whitespace(s)
    return s


def clean_input_text(text: str, kind: Literal["job_description", "resume"]) -> str:
    """Normalize JD or resume text before retrieval / chunking."""
    if kind == "job_description":
        return _clean_job_description(text)
    if kind == "resume":
        return _clean_resume_text(text)
    raise ValueError(f"Unknown kind: {kind!r}")


STANDARD_RETRIEVAL_K = 5
RERANK_CANDIDATE_K = 25
FINAL_CONTEXT_K = 5

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
    
    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int = FINAL_CONTEXT_K,
    ) -> list[dict]:
        """
        Use Cohere reranker to score and order chunks for the given query.
        
        Returns one dict per Cohere result: rerank_score, faiss_score (original
        retrieval score), chunk_index, and chunk text.
        """
        if not chunks:
            return []

        documents = [c["chunk"] for c in chunks]
        n = min(top_n, len(documents))

        response = self.cohere.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=n,
        )

        reranked = []
        for result in response.results:
            original_chunk = chunks[result.index]
            reranked.append({
                "chunk": original_chunk["chunk"],
                "faiss_score": float(original_chunk["score"]),
                "chunk_index": int(original_chunk["chunk_index"]),
                "rerank_score": float(result.relevance_score),
            })

        return reranked


# ─────────────────────────────────────────────────────────────
# SECTION 3: THREE GENERATION CONFIGURATIONS
#
# PM explanation: Haiku + Groq share standard top-K retrieval; the
# third config expands candidates, evidence-aware reranks, and picks
# a diverse final context set before generation.
#
# Config 1: Claude Haiku -- generation baseline
# Config 2: Llama 3 8B on Groq -- cheap, fast
# Config 3: Llama 3 8B on Groq + reranker + diversity -- optimized context
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


def build_evidence_aware_rerank_query(job_description: str) -> str:
    """
    Build a rerank instruction that prefers substantive resume evidence
    over shallow keyword overlap with the JD.
    """
    jd_preview = job_description[:1500] + ("..." if len(job_description) > 1500 else "")
    return f"""You are ranking resume excerpts for an interview-prep assistant.

JOB DESCRIPTION (for relevance only):
{jd_preview}

Rank these excerpts for USE AS SOURCE MATERIAL for five strong talking points. Prioritize:
1) Direct relevance to the responsibilities and themes in the job description
2) Quantified achievements (metrics, scale, %, revenue, latency, etc.)
3) Concrete resume evidence (specific projects, tools, outcomes), not generic claims
4) Technical depth where it grounds credible answers
5) Clear ownership or decision-making (what the candidate led, shipped, or influenced)
6) Product or business impact tied to outcomes

Also prefer a MIX of angles across five eventual talking points — avoid five chunks that all say the same thing.

Do NOT boost excerpts merely because they repeat many job-description keywords or buzzwords without substantive proof. Favor signal over lexical overlap."""


def infer_bucket(text: str) -> str:
    """Assign a coarse topic bucket for diversity filtering."""
    low = text.lower()

    if (
        re.search(r"\b(ai|ml)\b", low)
        or "machine learning" in low
        or "model" in low
        or "vertex" in low
        or "inference" in low
        or re.search(r"\brag\b", low)
        or "llm" in low
    ):
        return "ai_ml"
    if any(
        k in low
        for k in (
            "platform",
            "infrastructure",
            "cloud",
            "gpu",
            "latency",
            "throughput",
            "scaling",
            "api",
        )
    ):
        return "infra_platform"
    if any(
        k in low
        for k in (
            "silicon",
            "semiconductor",
            "power",
            "asic",
            "chip",
            "eda",
            "pdk",
            "nvidia",
        )
    ):
        return "hardware_systems"
    if any(
        k in low
        for k in (
            "revenue",
            "cost",
            "margin",
            "customer",
            "design win",
            "adoption",
            "growth",
        )
    ):
        return "business_impact"
    if any(
        k in low
        for k in (
            "led ",
            "owned ",
            "drove ",
            "cross-functional",
            "cross functional",
            "stakeholder",
            "roadmap",
            "launched",
        )
    ) or re.search(r"\bled\b", low) or re.search(r"\bowned\b", low) or re.search(r"\bdrove\b", low):
        return "leadership_execution"

    return "general"


def select_diverse_chunks(reranked_chunks: list[dict], final_k: int = FINAL_CONTEXT_K) -> list[dict]:
    """
    Greedy selection: prefer one high-ranking chunk per bucket, then fill
    remaining slots by rerank order.
    """
    if not reranked_chunks:
        return []

    if len(reranked_chunks) <= final_k:
        return list(reranked_chunks)

    selected: list[dict] = []
    seen_index: set[int] = set()
    seen_buckets: set[str] = set()

    for c in reranked_chunks:
        if len(selected) >= final_k:
            break
        idx = c["chunk_index"]
        if idx in seen_index:
            continue
        b = infer_bucket(c["chunk"])
        if b not in seen_buckets:
            selected.append(c)
            seen_index.add(idx)
            seen_buckets.add(b)

    for c in reranked_chunks:
        if len(selected) >= final_k:
            break
        idx = c["chunk_index"]
        if idx in seen_index:
            continue
        selected.append(c)
        seen_index.add(idx)

    return selected[:final_k]


def run_haiku(rag: ResumeRAG, chunks: list[dict], jd: str) -> dict:
    """Config 1: Claude Haiku — generation cost/quality baseline."""
    prompt = build_prompt(chunks, jd)

    start = time.perf_counter()

    response = rag.claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    latency_ms = (time.perf_counter() - start) * 1000

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens * 1 / 1_000_000) + (output_tokens * 5 / 1_000_000)

    return {
        "config": "Claude Haiku",
        "answer": response.content[0].text,
        "latency_ms": round(latency_ms, 1),
        "cost_usd": round(cost, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
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


def run_groq_reranked(rag: ResumeRAG, jd: str) -> dict:
    """
    Config 3: Llama 3 8B on Groq + evidence-aware rerank + diverse context.
    Retrieves a wide candidate pool, reranks with an evidence-focused query,
    selects diverse chunks, then generates.
    """
    candidate_retrieve_start = time.perf_counter()
    candidates = rag.retrieve(jd, k=RERANK_CANDIDATE_K)
    candidate_retrieve_latency = (time.perf_counter() - candidate_retrieve_start) * 1000

    rerank_query = build_evidence_aware_rerank_query(jd)
    rerank_start = time.perf_counter()
    reranked = rag.rerank(rerank_query, candidates, top_n=RERANK_CANDIDATE_K)
    rerank_latency = (time.perf_counter() - rerank_start) * 1000
    
    selected = select_diverse_chunks(reranked, final_k=FINAL_CONTEXT_K)
    selected_chunk_buckets = [infer_bucket(c["chunk"]) for c in selected]

    prompt = build_prompt(selected, jd)

    start = time.perf_counter()

    response = rag.groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
        temperature=0.1,
    )

    llm_latency = (time.perf_counter() - start) * 1000
    total_latency = candidate_retrieve_latency + rerank_latency + llm_latency

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    llm_cost = (input_tokens * 0.05 / 1_000_000) + (output_tokens * 0.08 / 1_000_000)
    rerank_cost = 0.001 / 1000
    total_cost = llm_cost + rerank_cost

    return {
        "config": "Llama 3 8B + Reranker (Groq)",
        "answer": response.choices[0].message.content,
        "latency_ms": round(total_latency, 1),
        "candidate_retrieve_latency_ms": round(candidate_retrieve_latency, 1),
        "rerank_latency_ms": round(rerank_latency, 1),
        "llm_latency_ms": round(llm_latency, 1),
        "cost_usd": round(total_cost, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "selected_chunks": selected,
        "candidate_chunks_count": len(candidates),
        "selected_chunk_buckets": selected_chunk_buckets,
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

NON-ANSWER RULE (apply BEFORE the scoring rubric below):
If the AI-generated talking points ask the user to provide a job description, say the JD is missing, refuse to generate talking points, or provide a checklist instead of actual candidate-specific talking points, assign:
score: 0
faithfulness: 0
relevance: 0
specificity: 0

VALID ANSWER REQUIREMENT:
A valid response must consist of exactly five candidate-specific talking points grounded in the resume context above—each point must tie to concrete resume evidence, not generic advice or hypothetical tips. If there are fewer than five substantive points, or points are not grounded in the resume context, assign very low scores across faithfulness, relevance, and specificity.

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
        score = int(re.search(r'"score":\s*(\d+)', raw).group(1))
        faithfulness = int(re.search(r'"faithfulness":\s*(\d+)', raw).group(1))
        relevance = int(re.search(r'"relevance":\s*(\d+)', raw).group(1))
        specificity = int(re.search(r'"specificity":\s*(\d+)', raw).group(1))

        reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', raw)
        reasoning = reasoning_match.group(1) if reasoning_match else "Could not parse"

        return {
            "score": score,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "specificity": specificity,
            "reasoning": reasoning,
            "judge_latency_ms": round(judge_latency_ms, 1),
            "judge_model": "claude-sonnet-4-5",
        }

    except Exception as e:
        print(f"  Scoring parse error: {e}")
        print(f"  Raw response: {response.content[0].text[:300]}")
        return {
            "score": 0,
            "reasoning": f"Scoring failed: {e}",
            "judge_latency_ms": 0,
            "judge_model": "claude-sonnet-4-5",
        }


# ─────────────────────────────────────────────────────────────
# SECTION 5: MAIN BENCHMARK FUNCTION
#
# Orchestrates the full pipeline:
# 1. Retrieve top STANDARD_RETRIEVAL_K chunks for Haiku and Groq
# 2. Run Haiku, Groq, and Groq + rerank/diversity (wide retrieve inside)
# 3. Score each config with Sonnet as LLM-as-judge
# 4. Return comparison results (cost vs Haiku baseline)
# ─────────────────────────────────────────────────────────────

def _wandb_try_init() -> bool:
    try:
        wandb.init(
            project="rag-inference-optimizer",
            job_type="benchmark",
            reinit=True,
        )
        return True
    except Exception as e:
        print(f"W&B init failed (non-blocking): {e}")
        return False


def _wandb_safe_log(payload: dict, active: bool) -> None:
    if not active:
        return
    try:
        wandb.log(payload)
    except Exception as e:
        print(f"W&B log failed (non-blocking): {e}")


def _run_benchmark_impl(
    rag: ResumeRAG,
    job_description: str,
    wandb_active: bool,
    original_job_description: str | None,
) -> dict:
    print(f"\nRunning benchmark for JD ({len(job_description)} chars)...")

    print("Retrieving relevant resume chunks (standard retrieval)...")
    retrieve_start = time.perf_counter()
    chunks = rag.retrieve(job_description, k=STANDARD_RETRIEVAL_K)
    retrieve_latency = (time.perf_counter() - retrieve_start) * 1000

    print(f"Retrieved {len(chunks)} chunks in {retrieve_latency:.1f}ms")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1} (score: {c['score']:.3f}): {c['chunk'][:60]}...")

    jd_first_1000 = job_description[:1000]
    jd_last_1000 = (
        job_description[-1000:] if len(job_description) > 1000 else job_description
    )
    top_chunks_preview = [c["chunk"][:200] for c in chunks]
    top_chunk_scores = [float(c["score"]) for c in chunks]

    preflight: dict = {
        "jd/cleaned_char_count": len(job_description),
        "jd/preview_first_1000_chars": jd_first_1000,
        "jd/preview_last_1000_chars": jd_last_1000,
        "retrieval/top_chunks_preview": top_chunks_preview,
        "retrieval/top_chunk_scores": top_chunk_scores,
    }
    if original_job_description is not None:
        preflight["jd/original_char_count"] = len(original_job_description)
    _wandb_safe_log(preflight, wandb_active)

    print("\nRunning Config 1: Claude Haiku...")
    haiku_prompt = build_prompt(chunks, job_description)
    haiku_result = run_haiku(rag, chunks, job_description)
    print(f"  Done: {haiku_result['latency_ms']}ms, ${haiku_result['cost_usd']:.6f}")
    _wandb_safe_log(
        {
            "claude_haiku/prompt_preview_first_1500_chars": haiku_prompt[:1500],
            "claude_haiku/answer_preview_first_1000_chars": haiku_result["answer"][:1000],
            "claude_haiku/input_prompt_preview": haiku_prompt,
            "claude_haiku/output_text": haiku_result["answer"],
        },
        wandb_active,
    )

    print("Running Config 2: Llama 3 8B on Groq...")
    groq_prompt = build_prompt(chunks, job_description)
    groq_result = run_groq(rag, chunks, job_description)
    print(f"  Done: {groq_result['latency_ms']}ms, ${groq_result['cost_usd']:.6f}")
    _wandb_safe_log(
        {
            "llama_groq/prompt_preview_first_1500_chars": groq_prompt[:1500],
            "llama_groq/answer_preview_first_1000_chars": groq_result["answer"][:1000],
        },
        wandb_active,
    )

    print("Running Config 3: Llama 3 8B + evidence-aware rerank + diversity...")
    groq_reranked_result = run_groq_reranked(rag, job_description)
    print(f"  Done: {groq_reranked_result['latency_ms']}ms, ${groq_reranked_result['cost_usd']:.6f}")
    rerank_gen_prompt = build_prompt(
        groq_reranked_result["selected_chunks"], job_description
    )
    _wandb_safe_log(
        {
            "llama_groq_reranker/prompt_preview_first_1500_chars": rerank_gen_prompt[:1500],
            "llama_groq_reranker/answer_preview_first_1000_chars": groq_reranked_result[
                "answer"
            ][:1000],
        },
        wandb_active,
    )

    print("\nScoring quality with LLM-as-judge (Claude Sonnet)...")
    haiku_score = score_quality(rag, haiku_result["answer"], chunks, job_description)
    groq_score = score_quality(rag, groq_result["answer"], chunks, job_description)
    rerank_chunks_for_judge = groq_reranked_result["selected_chunks"]
    groq_reranked_score = score_quality(
        rag, groq_reranked_result["answer"], rerank_chunks_for_judge, job_description
    )

    baseline_cost = haiku_result["cost_usd"]
    groq_savings = (
        round((1 - groq_result["cost_usd"] / baseline_cost) * 100, 1)
        if baseline_cost > 0
        else 0
    )
    groq_reranked_savings = (
        round((1 - groq_reranked_result["cost_usd"] / baseline_cost) * 100, 1)
        if baseline_cost > 0
        else 0
    )

    try:
        _wandb_safe_log(
            {
                "judge/model": "claude-sonnet-4-5",
                "retrieval/standard_k": STANDARD_RETRIEVAL_K,
                "retrieval/reranker_candidate_k": RERANK_CANDIDATE_K,
                "retrieval/latency_ms": round(retrieve_latency, 1),
                "retrieval/chunks_retrieved": len(chunks),
                "retrieval/top_chunk_score": chunks[0]["score"] if chunks else 0,
                "llama_groq_reranker/used_reranker": True,
                "llama_groq_reranker/selected_chunk_buckets": groq_reranked_result[
                    "selected_chunk_buckets"
                ],
                "llama_groq_reranker/candidate_chunks_count": groq_reranked_result[
                    "candidate_chunks_count"
                ],
                "llama_groq_reranker/final_context_k": FINAL_CONTEXT_K,
            },
            wandb_active,
        )

        for cfg in [
            ("claude_haiku", haiku_result, haiku_score),
            ("llama_groq", groq_result, groq_score),
            ("llama_groq_reranker", groq_reranked_result, groq_reranked_score),
        ]:
            config_name, result, score = cfg
            _wandb_safe_log(
                {
                    f"{config_name}/retrieve_latency_ms": round(retrieve_latency, 1),
                    f"{config_name}/generation_latency_ms": result["latency_ms"],
                    f"{config_name}/judge_latency_ms": score.get("judge_latency_ms", 0),
                    f"{config_name}/total_latency_ms": round(
                        retrieve_latency
                        + result["latency_ms"]
                        + score.get("judge_latency_ms", 0),
                        1,
                    ),
                    f"{config_name}/generation_cost_usd": result["cost_usd"],
                    f"{config_name}/quality_score": score.get("score", 0),
                    f"{config_name}/faithfulness": score.get("faithfulness", 0),
                    f"{config_name}/relevance": score.get("relevance", 0),
                    f"{config_name}/specificity": score.get("specificity", 0),
                    f"{config_name}/input_tokens": result.get("input_tokens", 0),
                    f"{config_name}/output_tokens": result.get("output_tokens", 0),
                },
                wandb_active,
            )
    except Exception as e:
        print(f"W&B summary logging failed (non-blocking): {e}")

    return {
        "retrieved_chunks": chunks,
        "retrieve_latency_ms": round(retrieve_latency, 1),
        "configs": [
            {
                **haiku_result,
                "quality_score": haiku_score.get("score", 0),
                "quality_breakdown": haiku_score,
                "cost_savings_vs_baseline": "baseline",
            },
            {
                **groq_result,
                "quality_score": groq_score.get("score", 0),
                "quality_breakdown": groq_score,
                "cost_savings_vs_baseline": f"{groq_savings}% cheaper",
            },
            {
                **groq_reranked_result,
                "quality_score": groq_reranked_score.get("score", 0),
                "quality_breakdown": groq_reranked_score,
                "cost_savings_vs_baseline": f"{groq_reranked_savings}% cheaper",
            },
        ],
    }


def run_benchmark(
    rag: ResumeRAG,
    job_description: str,
    *,
    original_job_description: str | None = None,
) -> dict:
    """
    Run the full three-config benchmark for a job description.
    Returns structured results for the frontend to display.
    """
    wandb_active = _wandb_try_init()
    try:
        return _run_benchmark_impl(
            rag, job_description, wandb_active, original_job_description
        )
    finally:
        if wandb_active:
            try:
                wandb.finish()
            except Exception as e:
                print(f"W&B finish failed (non-blocking): {e}")
            else:
                print("W&B run logged successfully")


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