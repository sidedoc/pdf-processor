import os
import fitz
import requests
import json
import base64
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import tempfile
from PIL import Image
import io
import threading
import time
import uuid

app = FastAPI(title="PDF Processing API")

# Storage for tasks and results
class TaskStatus:
    def __init__(self):
        self.tasks = {}
        self.results = {}
        self.lock = threading.Lock()

task_manager = TaskStatus()

class PDFProcessor:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.site_url = os.environ.get("SITE_URL")
        self.app_name = os.environ.get("APP_NAME")
        self.session = requests.Session()
        self.rate_limit_delay = 0.1
        self.last_api_call = 0
        self.api_lock = threading.Lock()

    def process_page(self, page: fitz.Page) -> Dict[str, Any]:
        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes()
        
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffer.getvalue()).decode()

        # Process with vision API
        with self.api_lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call
            if time_since_last_call < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last_call)
            self.last_api_call = time.time()

            response = self.session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.app_name,
                },
                json={
                    "model": "mistralai/pixtral-12b",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Return the text, data, tables or charts verbatim. Describe images in natural language."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                }
            )
            return response.json()

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        doc = fitz.open(file_path)
        results = {
            "pages": [],
            "metadata": {
                "total_pages": len(doc),
                "processing_time": None
            }
        }
        
        start_time = time.time()
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Check if page has images or is likely to have tables/charts
                if len(page.get_images()) > 0 or len(page.get_drawings()) > 5:
                    result = self.process_page(page)
                    results["pages"].append({
                        "page_number": page_num + 1,
                        "type": "visual",
                        "content": result
                    })
                else:
                    text = page.get_text()
                    results["pages"].append({
                        "page_number": page_num + 1,
                        "type": "text",
                        "content": text
                    })
        finally:
            doc.close()
        
        results["metadata"]["processing_time"] = time.time() - start_time
        return results

# Initialize processor
pdf_processor = PDFProcessor()

# API Models
class ProcessingResponse(BaseModel):
    task_id: str
    status: str
    message: str

@app.get("/")
async def root():
    return {
        "greeting": "PDF Processing API",
        "endpoints": {
            "POST /process-pdf": "Upload PDF for processing",
            "GET /status/{task_id}": "Check processing status",
            "GET /result/{task_id}": "Get processing results"
        }
    }

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    task_id = str(uuid.uuid4())
    
    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"{task_id}.pdf")
    
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Initialize task status
    with task_manager.lock:
        task_manager.tasks[task_id] = {"status": "processing", "started_at": datetime.now()}
    
    # Process PDF in background
    async def process_task():
        try:
            result = pdf_processor.process_pdf(temp_path)
            with task_manager.lock:
                task_manager.results[task_id] = result
                task_manager.tasks[task_id]["status"] = "completed"
                task_manager.tasks[task_id]["completed_at"] = datetime.now()
        except Exception as e:
            with task_manager.lock:
                task_manager.tasks[task_id]["status"] = "failed"
                task_manager.tasks[task_id]["error"] = str(e)
        finally:
            try:
                os.remove(temp_path)
                os.rmdir(temp_dir)
            except:
                pass

    background_tasks.add_task(process_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="processing",
        message="PDF processing started"
    )

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    with task_manager.lock:
        task = task_manager.tasks.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task_id,
        "status": task["status"],
        **task
    }

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    with task_manager.lock:
        task = task_manager.tasks.get(task_id)
        result = task_manager.results.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] == "failed":
        raise HTTPException(status_code=400, detail=task.get("error", "Processing failed"))
    
    if task["status"] != "completed":
        raise HTTPException(status_code=202, detail="Processing not completed")
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return result
