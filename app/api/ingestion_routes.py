import os, shutil, uuid
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.services.ingestion import IngestionService

router = APIRouter()
ingestion_service = IngestionService()
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory dictionary to track async jobs
JOBS = {}

def process_in_background(job_id: str, temp_path: str, filename: str):
    """Adds ingestion process as a background job"""
    try:
        JOBS[job_id]["status"] = "processing"
        result = ingestion_service.ingest_file(temp_path, filename)
        JOBS[job_id].update({"status": "completed", "result": result})
    except Exception as e:
        JOBS[job_id].update({"status": "failed", "error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/api/ingest", status_code=202)
async def ingest_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Takes a file and processes it for ingestion"""
    # Check if file was provided and that its a pdf
    if not file.filename or file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDFs allowed.")
    # Generate uuid for job id
    job_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    JOBS[job_id] = {"filename": file.filename, "status": "queued"}
    background_tasks.add_task(process_in_background, job_id, temp_path, file.filename)

    return {"job_id": job_id, "status": "queued", "message": "Processing in background."}

@router.get("/api/ingest/{job_id}")
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JOBS[job_id]