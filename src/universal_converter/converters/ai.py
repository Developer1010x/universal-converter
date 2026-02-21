"""AI and ML model conversion module"""

from pathlib import Path
from typing import Optional, Dict, Any

from . import BaseConverter, ConversionTask, ConversionResult, DependencyError


class AIConverter(BaseConverter):
    """Converter for AI/ML models and formats"""
    
    SUPPORTED_CONVERSIONS = {
        "h5": ["onnx", "tflite", "pt"],
        "pt": ["onnx", "tflite", "h5"],
        "onnx": ["pt", "tflite", "h5"],
        "tflite": ["onnx", "pt"],
        "pkl": ["onnx", "json"],
        "joblib": ["onnx", "json"],
    }
    PRIORITY = 25
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()
        
        if source == "h5" and target == "onnx":
            return self._h5_to_onnx(task)
        elif source == "pt" and target == "onnx":
            return self._pt_to_onnx(task)
        elif source == "onnx" and target == "pt":
            return self._onnx_to_pt(task)
        elif source == "onnx" and target == "tflite":
            return self._onnx_to_tflite(task)
        elif source in ["pkl", "joblib"] and target == "onnx":
            return self._sklearn_to_onnx(task)
        
        return ConversionResult(success=False, error=f"Unsupported: {source} → {target}")
    
    def _h5_to_onnx(self, task: ConversionTask) -> ConversionResult:
        try:
            import onnx
            from onnxmltools import convert_keras
            from onnxmltools.utils import save_model
            
            return ConversionResult(
                success=False,
                error="H5 to ONNX requires keras2onnx. Install: pip install universal-converter[ai]"
            )
        except ImportError:
            return ConversionResult(
                success=False,
                error="keras2onnx required. Install: pip install universal-converter[ai]"
            )
    
    def _pt_to_onnx(self, task: ConversionTask) -> ConversionResult:
        try:
            import torch
            import onnx
            
            model = torch.load(task.source_path, weights_only=False)
            model.eval()
            
            dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                model, dummy_input, str(task.target_path),
                export_params=True, opset_version=11,
                do_constant_folding=True,
                input_names=['input'], output_names=['output']
            )
            return ConversionResult(success=True, output_path=str(task.target_path))
        except ImportError:
            return ConversionResult(
                success=False,
                error="PyTorch and ONNX required. Install: pip install universal-converter[ai]"
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _onnx_to_pt(self, task: ConversionResult) -> ConversionResult:
        try:
            import torch
            import onnx
            
            onnx_model = onnx.load(str(task.source_path))
            torch.onnx.utils._export_model(onnx_model, (), str(task.target_path))
            
            return ConversionResult(success=True, output_path=str(task.target_path))
        except ImportError:
            return ConversionResult(
                success=False,
                error="PyTorch and ONNX required. Install: pip install universal-converter[ai]"
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _onnx_to_tflite(self, task: ConversionTask) -> ConversionResult:
        return ConversionResult(
            success=False,
            error="ONNX to TFLite requires onnx-tf. Install: pip install universal-converter[ai]"
        )
    
    def _sklearn_to_onnx(self, task: ConversionTask) -> ConversionResult:
        return ConversionResult(
            success=False,
            error="sklearn to ONNX requires skl2onnx. Install: pip install universal-converter[ai]"
        )
    
    def convert_format(self, path: str, to_format: str, output: Optional[str] = None) -> ConversionResult:
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format
        )
        return self.convert(task)


class TokenizerConverter(BaseConverter):
    """Converter for tokenizer formats (HuggingFace)"""
    
    SUPPORTED_CONVERSIONS = {
        "json": ["txt", "vocab"],
        "txt": ["json", "vocab"],
    }
    PRIORITY = 30
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(task.source_path)
            tokenizer.save_pretrained(str(task.target_path.parent))
            
            return ConversionResult(
                success=True,
                output_path=str(task.target_path)
            )
        except ImportError:
            return ConversionResult(
                success=False,
                error="transformers required. Install: pip install universal-converter[ai]"
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
