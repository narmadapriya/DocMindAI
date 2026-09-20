from pathlib import Path
from app.rag.ocr.easyocr_engine import OCREngineError,TesseractOCREngine
from app.rag.ocr.image_preprocessing import preprocess_for_ocr
class OCRService:
    def __init__(self,engine=None): self.engine=engine or TesseractOCREngine()
    def extract(self,image_path): return self.engine.extract_text(preprocess_for_ocr(image_path))
    def safe_extract(self,image_path):
        try:
            r=self.extract(image_path); return {'text':r.text,'confidence':r.confidence,'engine':r.engine,'status':'success','metadata':r.metadata or {}}
        except OCREngineError as e: return {'text':'','confidence':None,'engine':'tesseract','status':'unavailable','error':str(e),'metadata':{}}
