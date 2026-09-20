import base64,json,urllib.request
from dataclasses import dataclass
from pathlib import Path
class VisionModelError(RuntimeError): pass
@dataclass
class VisualEvidence:
    description:str; visual_type:str; text:list[str]; labels:list[str]; axes:list[str]; trends:list[str]; numerical_relationships:list[str]; comparisons:list[str]; confidence:float|None; model:str; source_path:str; raw_response:str=''
    def to_dict(self): return self.__dict__.copy()
DEFAULT_VISION_PROMPT='Return ONLY JSON with keys description, visual_type, text, labels, axes, trends, numerical_relationships, comparisons, confidence. Analyze visible text, labels, axes, trends, numerical relationships and comparisons. Do not invent unreadable values.'
class OllamaVisionClient:
    def __init__(self,base_url='http://localhost:11434',model='qwen2.5vl:3b',timeout=120): self.base_url=base_url.rstrip('/'); self.model=model; self.timeout=timeout
    def is_available(self):
        try:
            with urllib.request.urlopen(self.base_url+'/api/tags',timeout=5) as r: return r.status==200
        except: return False
    def analyze_image(self,image_path,prompt=DEFAULT_VISION_PROMPT):
        p=Path(image_path); encoded=base64.b64encode(p.read_bytes()).decode()
        payload={'model':self.model,'stream':False,'messages':[{'role':'user','content':prompt,'images':[encoded]}],'options':{'temperature':0}}
        req=urllib.request.Request(self.base_url+'/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(req,timeout=self.timeout) as r: data=json.loads(r.read().decode())
        except Exception as e: raise VisionModelError(f'Ollama vision request failed: {e}') from e
        raw=((data.get('message') or {}).get('content') or '').strip()
        x=self._parse_json(raw)
        return VisualEvidence(str(x.get('description','')),str(x.get('visual_type','other')),self._list(x.get('text')),self._list(x.get('labels')),self._list(x.get('axes')),self._list(x.get('trends')),self._list(x.get('numerical_relationships')),self._list(x.get('comparisons')),self._float(x.get('confidence')),self.model,str(p),raw)
    @staticmethod
    def _list(v): return [] if v is None else [str(x) for x in v] if isinstance(v,list) else [str(v)]
    @staticmethod
    def _float(v):
        try: return float(v) if v is not None else None
        except: return None
    @staticmethod
    def _parse_json(raw):
        s=raw.strip(); s=s[7:-3].strip() if s.startswith('```json') and s.endswith('```') else s
        try: return json.loads(s)
        except: pass
        a,b=s.find('{'),s.rfind('}')
        if a>=0 and b>a:
            try: return json.loads(s[a:b+1])
            except: pass
        raise VisionModelError('Qwen2.5-VL response was not valid JSON')
