# ─────────────────────────────────────────────────────────────
# Product 3 — RAG Inference Optimizer
# FastAPI Backend
#
# Exposes one main endpoint:
# POST /benchmark — takes a job description, returns three
#                   config results with cost/latency/quality
#
# The frontend calls this endpoint when user pastes a JD
# ─────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from rag import ResumeRAG, run_benchmark, clean_input_text

# ─────────────────────────────────────────────────────────────
# STARTUP -- initialize RAG pipeline once when server starts
# Loading the embedding model and building FAISS index
# takes ~5 seconds. We do this once at startup, not per request.
# ─────────────────────────────────────────────────────────────

rag_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_instance
    print("Starting up -- initializing RAG pipeline...")
    resume_path = os.path.join(os.path.dirname(__file__), "resume.txt")
    rag_instance = ResumeRAG(resume_path)
    print("RAG pipeline ready. Server accepting requests.")
    yield
    print("Shutting down.")

app = FastAPI(
    title="RAG Inference Optimizer API",
    description="Benchmarks Claude vs Llama 3.1 8B on Groq for resume-to-JD matching",
    version="1.0.0",
    lifespan=lifespan
)

# Allow frontend to call this API
# In production you would restrict this to your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# REQUEST AND RESPONSE MODELS
# Pydantic models define the shape of API requests and responses
# FastAPI uses these for automatic validation and documentation
# ─────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    job_description: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_description": "Senior PM at AI infrastructure company. Requirements: hardware-software fluency, inference optimization experience, enterprise GTM..."
            }
        }

class ConfigResult(BaseModel):
    config: str
    answer: str
    latency_ms: float
    cost_usd: float
    quality_score: int
    cost_savings_vs_baseline: str

class BenchmarkResponse(BaseModel):
    configs: list[ConfigResult]
    retrieve_latency_ms: float
    retrieved_chunks: list[dict]

# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint. Returns status of RAG pipeline."""
    return {
        "status": "healthy",
        "rag_ready": rag_instance is not None,
        "chunks_indexed": rag_instance.index.ntotal if rag_instance else 0
    }

@app.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    """
    Run three-config RAG benchmark for a job description.
    
    Takes a job description, retrieves relevant resume chunks,
    runs Claude Sonnet + Llama 3.1 8B + Llama 3.1 8B with reranker,
    scores quality with LLM-as-judge, returns comparison results.
    
    Typical response time: 15-20 seconds (three LLM calls + scoring)
    """
    if not rag_instance:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    jd = clean_input_text(request.job_description, kind="job_description")

    if len(jd.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Job description too short. Please paste the full JD."
        )

    try:
        results = run_benchmark(
            rag_instance,
            jd,
            original_job_description=request.job_description,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chunks")
@app.post("/update-resume")
async def update_resume(file: UploadFile = File(...)):
    """
    Upload a new resume file and rebuild the FAISS index.
    Accepts .txt, .pdf, or .docx files.
    """
    if not rag_instance:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    content = await file.read()
    filename = file.filename or ""
    text = ""

    try:
        if filename.endswith('.pdf'):
            # Extract text from PDF using pdfminer
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            import io
            output = io.StringIO()
            extract_text_to_fp(
                io.BytesIO(content),
                output,
                laparams=LAParams(),
                output_type='text',
                codec='utf-8'
            )
            text = output.getvalue()

        elif filename.endswith('.docx'):
            # Extract text from DOCX
            import io
            from docx import Document
            doc = Document(io.BytesIO(content))
            text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])

        else:
            # Plain text
            text = content.decode('utf-8', errors='ignore')

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    text = clean_input_text(text, kind="resume")

    if len(text.strip()) < 100:
        raise HTTPException(
            status_code=400,
            detail="Resume too short or could not extract text. Try a .txt file."
        )

    # Re-chunk and rebuild FAISS index
    from rag import chunk_resume
    import faiss

    print(f"Rebuilding index with new resume ({len(text)} chars)...")
    new_chunks = chunk_resume(text)

    if len(new_chunks) < 3:
        raise HTTPException(
            status_code=400,
            detail="Could not extract enough content. Try a .txt file."
        )

    # Re-embed
    embeddings = rag_instance.embedder.encode(
        new_chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    faiss.normalize_L2(embeddings)

    # Rebuild index
    dimension = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(embeddings)

    # Update in place
    rag_instance.chunks = new_chunks
    rag_instance.index = new_index

    print(f"Index rebuilt: {len(new_chunks)} chunks")

    return {
        "status": "success",
        "message": "Resume updated successfully",
        "chunks_created": len(new_chunks),
        "file_type": filename.split('.')[-1],
        "preview": new_chunks[0][:150] + "..." if new_chunks else ""
    }