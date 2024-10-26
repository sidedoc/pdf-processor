import fitz
import requests
import json
import base64
from typing import Dict, List, Set, Any, Tuple
import tempfile
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time

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
