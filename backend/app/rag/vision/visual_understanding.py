from pathlib import Path
from app.rag.llm.vision_processor import OllamaVisionClient,VisionModelError
from app.rag.ocr.ocr_service import OCRService
class VisualUnderstandingService:
    def __init__(self,ocr_service=None,vision_client=None): self.ocr=ocr_service or OCRService(); self.vision=vision_client or OllamaVisionClient()
    def process_image(self,image):
        ocr=self.ocr.safe_extract(image.path); r={'asset_id':image.image_id,'asset_type':'image','source_path':image.path,'ocr':ocr,'vision':None,'status':'ocr_only'}
        try: r['vision']=self.vision.analyze_image(image.path).to_dict(); r['status']='success'
        except VisionModelError as e: r['vision_error']=str(e); r['status']='vision_unavailable'
        image.metadata['ocr']=ocr; image.metadata['ocr_text']=ocr.get('text',''); image.metadata['visual_evidence']=r.get('vision'); return r
    def process_chart(self,chart):
        r={'asset_id':chart.chart_id,'asset_type':'chart','source_path':chart.image_path,'chart_type':chart.chart_type,'title':chart.title,'vision':None}
        if not chart.image_path or not Path(chart.image_path).exists(): r['status']='rendering_required'; return r
        try: r['vision']=self.vision.analyze_image(chart.image_path).to_dict(); r['status']='success'
        except VisionModelError as e: r['status']='vision_unavailable'; r['vision_error']=str(e)
        chart.metadata['visual_evidence']=r.get('vision'); chart.metadata['processing_status']=r['status']; return r
    def process_document(self,document):
        ev=[self.process_image(x) for x in document.images]+[self.process_chart(x) for x in document.charts]
        document.visual_evidence=ev; return document
