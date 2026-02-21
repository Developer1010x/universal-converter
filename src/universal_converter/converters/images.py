"""Image conversion module"""

from pathlib import Path
from typing import Optional, List
from PIL import Image

from . import BaseConverter, ConversionTask, ConversionResult, DependencyError


class ImageConverter(BaseConverter):
    """Converter for image formats"""
    
    SUPPORTED_CONVERSIONS = {
        "png": ["jpg", "gif", "bmp", "tiff", "webp"],
        "jpg": ["png", "gif", "bmp", "tiff", "webp"],
        "jpeg": ["png", "gif", "bmp", "tiff", "webp"],
        "gif": ["png", "jpg", "bmp", "tiff", "webp"],
        "bmp": ["png", "jpg", "gif", "tiff", "webp"],
        "tiff": ["png", "jpg", "gif", "bmp", "webp"],
        "webp": ["png", "jpg", "gif", "bmp", "tiff"],
    }
    PRIORITY = 10
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        try:
            img = Image.open(task.source_path)
            
            if task.target_format.lower() in ["jpg", "jpeg"]:
                img = img.convert("RGB")
            
            img.save(task.target_path)
            return ConversionResult(
                success=True,
                output_path=str(task.target_path),
                metadata={"size": img.size, "mode": img.mode}
            )
        except ImportError:
            return ConversionResult(
                success=False,
                error="PIL/Pillow required. Install: pip install pillow"
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def resize(
        self,
        path: str,
        output: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: Optional[float] = None
    ) -> ConversionResult:
        try:
            img = Image.open(path)
            original_width, original_height = img.size
            
            if scale:
                width = int(original_width * scale)
                height = int(original_height * scale)
            elif width and not height:
                height = int(original_height * (width / original_width))
            elif height and not width:
                width = int(original_width * (height / original_height))
            
            resized = img.resize((width, height), Image.Resampling.LANCZOS)
            
            out_path = output or str(Path(path).with_suffix(".png"))
            resized.save(out_path)
            
            return ConversionResult(
                success=True,
                output_path=out_path,
                metadata={"size": (width, height)}
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def thumbnail(self, path: str, output: Optional[str] = None, max_size: tuple = (128, 128)) -> ConversionResult:
        try:
            img = Image.open(path)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            out_path = output or str(Path(path).with_suffix(".png"))
            img.save(out_path)
            
            return ConversionResult(success=True, output_path=out_path)
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def convert_format(self, path: str, to_format: str, output: Optional[str] = None) -> ConversionResult:
        """Convert image to a specific format"""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format
        )
        return self.convert(task)
