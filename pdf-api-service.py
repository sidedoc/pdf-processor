```python
import fitz
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import uuid
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import uvicorn
from enum import Enum

# Import the PDFProcessor from previous implementation
from pdf_processor import PDFProcessor  # This is the class we created earlier

# Initialize FastAPI app
app = FastAPI(
    title="PDF Processing API",
    description="API for processing PDFs with text extraction and visual element analysis",
    version="1.0.0"
)

# Configuration
UPLOAD_DIR = "temp_uploads"
RESULTS_DIR = "processing_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize PDF Processor
pdf_processor = PDFProcessor(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    site_url=os.getenv("SITE_URL"),
    app_name=os.getenv("APP_NAME"),
    max_workers=int(os.getenv("MAX_WORKERS", "10"))
)

# Enums for status tracking
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Pydantic models for request/response
class ProcessingTask(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the processing task")
    status: ProcessingStatus = Field(..., description="Current status of the processing task")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class PageContent(BaseModel):
    page_number: int
    content_type: str  # "text" or "visual"
    content: str
    metadata: Dict[str, Any] = {}

class ProcessingResult(BaseModel):
    task_id: str
    document_metadata: Dict[str, Any]
    pages: List[PageContent]
    visual_elements: Dict[str, List[int]]
    processing_stats: Dict[str, Any]

# In-memory storage for task tracking
tasks: Dict[str, ProcessingTask] = {}
results: Dict[str, ProcessingResult] = {}

# Helper function for background processing
async def process_pdf_background(task_id: str, file_path: str):
    try:
        # Update task status
        tasks[task_id].status = ProcessingStatus.PROCESSING
        
        # Process the PDF
        raw_results = pdf_processor.process_pdf(file_path)
        
        # Structure the results
        pages = []
        
        # Add text content pages
        for page_num, content in raw_results["text_content"].items():
            pages.append(PageContent(
                page_number=page_num,
                content_type="text",
                content=content,
                metadata={"type": "text"}
            ))
        
        # Add visual content pages
        for page_num, content in raw_results["visual_content"].items():
            pages.append(PageContent(
                page_number=page_num,
                content_type="visual",
                content=content["choices"][0]["message"]["content"] if "choices" in content else str(content),
                metadata={
                    "type": "visual",
                    "elements": [k for k, v in raw_results["metadata"]["visual_elements"].items() if page_num in v]
                }
            ))
        
        # Sort pages by page number
        pages.sort(key=lambda x: x.page_number)
        
        # Create final result
        result = ProcessingResult(
            task_id=task_id,
            document_metadata={
                "total_pages": raw_results["metadata"]["total_pages"],
                "processing_time": raw_results["metadata"]["processing_stats"]["total_duration"]
            },
            pages=pages,
            visual_elements=raw_results["metadata"]["visual_elements"],
            processing_stats=raw_results["metadata"]["processing_stats"]
        )
        
        # Store result
        results[task_id] = result
        
        # Update task status
        tasks[task_id].status = ProcessingStatus.COMPLETED
        tasks[task_id].completed_at = datetime.now()
        
    except Exception as e:
        tasks[task_id].status = ProcessingStatus.FAILED
        tasks[task_id].error = str(e)
    finally:
        # Cleanup temporary file
        try:
            os.remove(file_path)
        except:
            pass

# API Endpoints
@app.post("/process-pdf/", response_model=ProcessingTask)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded file
    temp_path = os.path.join(UPLOAD_DIR, f"{task_id}.pdf")
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Create task
    task = ProcessingTask(
        task_id=task_id,
        status=ProcessingStatus.PENDING
    )
    tasks[task_id] = task
    
    # Start background processing
    background_tasks.add_task(process_pdf_background, task_id, temp_path)
    
    return task

@app.get("/task/{task_id}", response_model=ProcessingTask)
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/result/{task_id}", response_model=ProcessingResult)
async def get_task_result(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status == ProcessingStatus.FAILED:
        raise HTTPException(status_code=400, detail=f"Task failed: {task.error}")
    
    if task.status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=202, detail="Task still processing")
    
    result = results.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return result

@app.get("/task/{task_id}/text-content")
async def get_text_content(task_id: str, split_length: int = None):
    """
    Get text content suitable for text splitting.
    Optionally specify split_length to get text pre-split into chunks.
    """
    result = results.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Combine all text content in order
    text_content = []
    for page in result.pages:
        if page.content_type == "text":
            text_content.append(page.content)
    
    combined_text = "\n\n".join(text_content)
    
    # Split text if requested
    if split_length:
        # Simple splitting by character count
        # You might want to implement more sophisticated splitting logic
        splits = [combined_text[i:i+split_length] 
                 for i in range(0, len(combined_text), split_length)]
        return {"splits": splits}
    
    return {"text": combined_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
