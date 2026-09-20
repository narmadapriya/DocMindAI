import shutil
from dataclasses import dataclass
from pathlib import Path
class OCREngineError(RuntimeError): pass
@dataclass
class OCRResult:
    text:str; confidence:float|None=None; engine:str='tesseract'; metadata:dict|None=None
class TesseractOCREngine:
    def __init__(self,language='eng',tesseract_cmd=None): self.language=language; self.tesseract_cmd=tesseract_cmd or shutil.which('tesseract')
    @property
    def available(self): return bool(self.tesseract_cmd)
    def extract_text(self,image):
        if not self.available: raise OCREngineError('Tesseract OCR is not installed or is not on PATH.')
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd=self.tesseract_cmd
        d=pytesseract.image_to_data(image,lang=self.language,output_type=pytesseract.Output.DICT)
        words=[]; conf=[]
        for t,c in zip(d.get('text',[]),d.get('conf',[])):
            t=(t or '').strip()
            try: c=float(c)
            except: c=-1
            if t: words.append(t); conf.append(c) if c>=0 else None
        return OCRResult(' '.join(words).strip(),sum(conf)/len(conf) if conf else None,metadata={'language':self.language})
EasyOCREngine=TesseractOCREngine
