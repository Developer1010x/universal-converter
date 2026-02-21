"""Video conversion module"""

from pathlib import Path
from typing import Optional

from . import BaseConverter, ConversionTask, ConversionResult


class VideoConverter(BaseConverter):
    """Converter for video formats"""
    
    SUPPORTED_CONVERSIONS = {
        "mp4": ["avi", "mkv", "mov", "webm", "flv", "wmv"],
        "avi": ["mp4", "mkv", "mov", "webm", "flv", "wmv"],
        "mkv": ["mp4", "avi", "mov", "webm", "flv", "wmv"],
        "mov": ["mp4", "avi", "mkv", "webm", "flv", "wmv"],
        "webm": ["mp4", "avi", "mkv", "mov", "flv", "wmv"],
        "flv": ["mp4", "avi", "mkv", "mov", "webm", "wmv"],
    }
    PRIORITY = 20
    REQUIRES_EXTERNAL = ["ffmpeg"]
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        from ..utils.platform import find_executable, run_command
        
        if not find_executable("ffmpeg"):
            return ConversionResult(
                success=False,
                error="FFmpeg required. Install: pip install universal-converter[video]"
            )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(task.source_path),
            "-c:v", "libx264",
            "-c:a", "aac",
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
        """Convert video to a specific format"""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format
        )
        return self.convert(task)
    
    def extract_thumbnail(self, video_path: str, output: Optional[str] = None, timestamp: str = "00:00:01") -> ConversionResult:
        """Extract thumbnail from video"""
        from ..utils.platform import find_executable
        
        if not find_executable("ffmpeg"):
            return ConversionResult(
                success=False,
                error="FFmpeg required. Install: pip install universal-converter[video]"
            )
        
        out_path = output or str(Path(video_path).with_suffix(".jpg"))
        cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", timestamp, "-vframes", "1", out_path]
        
        from ..utils.platform import run_command
        result = run_command(cmd)
        
        if result["returncode"] == 0:
            return ConversionResult(success=True, output_path=out_path)
        return ConversionResult(success=False, error=result.get("stderr", "Extraction failed"))
    
    def get_info(self, video_path: str) -> ConversionResult:
        """Get video metadata using ffprobe"""
        from ..utils.platform import find_executable, run_command
        
        if not find_executable("ffprobe"):
            return ConversionResult(
                success=False,
                error="FFprobe required. Install FFmpeg."
            )
        
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
        result = run_command(cmd)
        
        if result["returncode"] == 0:
            import json
            try:
                data = json.loads(result.get("stdout", "{}"))
                return ConversionResult(success=True, data=data)
            except json.JSONDecodeError:
                return ConversionResult(success=False, error="Failed to parse video info")
        return ConversionResult(success=False, error=result.get("stderr", "Failed to get video info"))
