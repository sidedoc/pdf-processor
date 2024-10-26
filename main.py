import os
import fitz
import requests
import json
import base64
from typing import Dict, List, Set, Any, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
from PIL import Image
import io
import uvicorn
from enum import Enum
import threading
import time
import uuid

# Initialize FastAPI app
app = FastAPI(
    title="PDF Processing API",
    description="API for processing PDFs with text extraction and visual element analysis",
    version="1.0.0"
)

class PDFProcessor:
    def __init__(self, api_key: str, site_url: str, app_name: str, max_workers: int = None):
        self.api_key = api_key
        self.site_url = site_url
        self.app_name = app_name
        self.max_workers = max_workers
        self.session = requests.Session()
        self.rate_limit_delay = 0.1
        self.last_api_call = 0
        self.api_lock = threading.Lock()

    def detect_visual_elements(self, doc: fitz.Document) -> Dict[str, List[int]]:
        results = {
            "tables": set(),
            "images": set(),
            "charts": set()
        }
        
        def process_page(args: Tuple[int, fitz.Page]) -> Tuple[int, Dict[str, bool]]:
            page_num, page = args
            page_elements = {"tables": False, "images": False, "charts": False}
            
            page_dict = page.get_text("dict")
            page_content = str(page_dict)
            
            # Image detection
            if len(page.get_images()) > 0:
                page_elements["images"] = True
            
            # Table detection
            table_patterns = [r'/Table\b', r'/TableContent\b', r'<table', r'</table>', r'/TD\b', r'/TH\b', r'/TR\b']
            if any(pattern in page_content for pattern in table_patterns):
                page_elements["tables"] = True
            
            # Chart detection
            chart_patterns = [
                r'/Plot\b', r'/Chart\b', r'/Graph\b',
                r'/(Bar|Line|Pie|Scatter|Area)Chart\b',
                r'/Histogram\b', r'/Diagram\b'
            ]
            
            paths = page.get_drawings()
            if len(paths) > 5:
                has_lines = any(p.get("type") == "l" for p in paths)
                has_shapes = any(p.get("type") in ["re", "qu", "ci"] for p in paths)
                if has_lines and has_shapes or any(pattern in page_content for pattern in chart_patterns):
                    page_elements["charts"] = True
            
            return page_num, page_elements

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                futures.append(executor.submit(process_page, (page_num + 1, page)))
            
            for future in as_completed(futures):
                page_num, page_elements = future.result()
                for element_type, has_element in page_elements.items():
                    if has_element:
                        results[element_type].add(page_num)

        return {k: sorted(list(v)) for k, v in results.items()}

    def get_all_visual_pages(self, detection_results: Dict[str, List[int]]) -> Set[int]:
        return set().union(*detection_results.values())

    def process_text_page(self, args: Tuple[fitz.Document, int]) -> Tuple[int, str]:
        doc, page_num = args
        page = doc[page_num - 1]
        return page_num, page.get_text()

    def process_visual_page(self, args: Tuple[fitz.Document, int]) -> Tuple[int, Dict[str, Any]]:
        doc, page_num = args
        
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes()
        
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffer.getvalue()).decode()

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

        return page_num, response.json()

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        doc = fitz.open(pdf_path)
        
        detection_results = self.detect_visual_elements(doc)
        visual_pages = self.get_all_visual_pages(detection_results)
        
        results = {
            "metadata": {
                "total_pages": len(doc),
                "visual_elements": detection_results,
                "processing_stats": {
                    "start_time": time.time(),
                    "end_time": None,
                    "total_duration": None
                }
            },
            "text_content": {},
            "visual_content": {}
        }
        
        text_pages = [(doc, p) for p in range(1, len(doc) + 1) if p not in visual_pages]
        visual_pages = [(doc, p) for p in visual_pages]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for text_page in text_pages:
                futures.append(('text', executor.submit(self.process_text_page, text_page)))
            
            for visual_page in visual_pages:
                futures.append(('visual', executor.submit(self.process_visual_page, visual_page)))
            
            for future_type, future in futures:
                try:
                    page_num, content = future.result()
                    if future_type == 'text':
                        results["text_content"][page_num] = content
                    else:
                        results["visual_content"][page_num] = content
                except Exception as e:
                    print(f"Error processing page: {str(e)}")
        
        results["metadata"]["processing_stats"]["end_time"] = time.time()
        results["metadata"]["processing_stats"]["total_duration"] = (
            results["metadata"]["processing_stats"]["end_time"] - 
            results["metadata"]["processing_stats"]["start_time"]
        )
        
        doc.close()
        return results

# Configuration
UPLOAD_DIR = "temp_uploads"
RESULTS_DIR = "processing_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize PDF Processor with Railway environment variables
pdf_processor = PDFProcessor(
    api_key=os.environ["OPENROUTER_API_KEY"],
    site_url=os.environ["SITE_URL"],
    app_name=os.environ["APP_NAME"],
    max_workers=int(os.environ.get("MAX_WORKERS", "10"))
)

# Enums and Models
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingTask(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the processing task")
    status: ProcessingStatus = Field(..., description="Current status of the processing task")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class PageContent(BaseModel):
    page_number: int
    content_type: str
    content: str
    metadata: Dict[str, Any] = {}

class ProcessingResult(BaseModel):
    task_id: str
    document_metadata: Dict[str, Any]
    pages: List[PageContent]
    visual_elements: Dict[str, List[int]]
    processing_stats: Dict[str, Any]

# Storage
tasks: Dict[str, ProcessingTask] = {}
results: Dict[str, ProcessingResult] = {}

# Helper function for background processing
async def process_pdf_background(task_id: str, file_path: str):
    try:
        tasks[task_id].status = ProcessingStatus.PROCESSING
        
        raw_results = pdf_processor.process_pdf(file_path)
        
        pages = []
        
        for page_num, content in raw_results["text_content"].items():
            pages.append(PageContent(
                page_number=page_num,
                content_type="text",
                content=content,
                metadata={"type": "text"}
            ))
        
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
        
        pages.sort(key=lambda x: x.page_number)
        
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
        
        results[task_id] = result
        tasks[task_id].status = ProcessingStatus.COMPLETED
        tasks[task_id].completed_at = datetime.now()
        
    except Exception as e:
        tasks[task_id].status = ProcessingStatus.FAILED
        tasks[task_id].error = str(e)
    finally:
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
    
    task_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{task_id}.pdf")
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    task = ProcessingTask(
        task_id=task_id,
        status=ProcessingStatus.PENDING
    )
    tasks[task_id] = task
    
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
    result = results.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    text_content = []
    for page in result.pages:
        if page.content_type == "text":
            text_content.append(page.content)
    
    combined_text = "\n\n".join(text_content)
    
    if split_length:
        splits = [combined_text[i:i+split_length] 
                 for i in range(0, len(combined_text), split_length)]
        return {"splits": splits}
    
    return {"text": combined_text}

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
