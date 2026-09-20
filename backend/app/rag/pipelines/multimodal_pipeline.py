from app.rag.parser import parse_document
from app.rag.vision import VisualUnderstandingService
def process_multimodal_document(file_path,*,output_dir=None,visual_service=None): return (visual_service or VisualUnderstandingService()).process_document(parse_document(file_path,output_dir=output_dir))
