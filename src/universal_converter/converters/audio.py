"""Audio conversion module"""

from pathlib import Path
from typing import Optional

from . import BaseConverter, ConversionTask, ConversionResult, DependencyError


class AudioConverter(BaseConverter):
    """Converter for audio formats"""
    
    SUPPORTED_CONVERSIONS = {
        "mp3": ["wav", "flac", "ogg", "aac", "m4a"],
        "wav": ["mp3", "flac", "ogg", "aac", "m4a"],
        "flac": ["mp3", "wav", "ogg", "aac", "m4a"],
        "ogg": ["mp3", "wav", "flac", "aac", "m4a"],
        "aac": ["mp3", "wav", "flac", "ogg", "m4a"],
        "m4a": ["mp3", "wav", "flac", "ogg", "aac"],
    }
    PRIORITY = 20
    REQUIRES_EXTERNAL = ["ffmpeg"]
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        from ..utils.platform import find_executable, run_command
        
        if not find_executable("ffmpeg"):
            return ConversionResult(
                success=False,
                error="FFmpeg required. Install: pip install universal-converter[audio]"
            )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(task.source_path),
            str(task.target_path)
        ]
        
        result = run_command(cmd)
        
        if result["returncode"] == 0:
            return ConversionResult(
                success=True,
                output_path=str(task.target_path)
            )
        return ConversionResult(success=False, error=result.get("stderr", "Conversion failed"))
    
    def convert_format(self, path: str, to_format: str, output: Optional[str] = None) -> ConversionResult:
        """Convert audio to a specific format"""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format
        )
        return self.convert(task)
    
    def extract_audio(self, video_path: str, output: Optional[str] = None, format: str = "mp3") -> ConversionResult:
        """Extract audio from video file"""
        from ..utils.platform import find_executable
        
        if not find_executable("ffmpeg"):
            return ConversionResult(
                success=False,
                error="FFmpeg required. Install: pip install universal-converter[video]"
            )
        
        out_path = output or str(Path(video_path).with_suffix(f".{format}"))
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "libmp3lame", out_path]
        
        from ..utils.platform import run_command
        result = run_command(cmd)
        
        if result["returncode"] == 0:
            return ConversionResult(success=True, output_path=out_path)
        return ConversionResult(success=False, error=result.get("stderr", "Extraction failed"))
