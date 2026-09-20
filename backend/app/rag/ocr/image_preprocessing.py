from pathlib import Path
from PIL import Image,ImageOps,ImageFilter

def load_image(source):
    if isinstance(source,(str,Path)): im=Image.open(source)
    elif isinstance(source,bytes):
        from io import BytesIO; im=Image.open(BytesIO(source))
    else: im=Image.open(source)
    return im.convert('RGB')

def preprocess_for_ocr(source,scale=1.5,grayscale=True,autocontrast=True,sharpen=True):
    im=load_image(source)
    if scale!=1: im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.Resampling.LANCZOS)
    if grayscale: im=ImageOps.grayscale(im)
    if autocontrast: im=ImageOps.autocontrast(im)
    if sharpen: im=im.filter(ImageFilter.SHARPEN)
    return im
