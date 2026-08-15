"""Video conversion module (ffmpeg-backed)."""

from pathlib import Path
from typing import Optional

from . import BaseConverter, ConversionTask, ConversionResult
from .audio import _FFMPEG_HINT, _tail

# Codecs that are actually valid for each container. Forcing libx264/aac into a
# .webm produces "Only VP8 or VP9 or AV1 video and Vorbis or Opus audio and
# WebVTT subtitles are supported for WebM", which is why the container has to
# pick its own codecs rather than using one hardcoded pair for everything.
_CODECS = {
    "webm": ["-c:v", "libvpx-vp9", "-c:a", "libopus"],
    "mp4": ["-c:v", "libx264", "-c:a", "aac"],
    "mov": ["-c:v", "libx264", "-c:a", "aac"],
    "mkv": ["-c:v", "libx264", "-c:a", "aac"],
    "flv": ["-c:v", "libx264", "-c:a", "aac"],
    "avi": ["-c:v", "mpeg4", "-c:a", "libmp3lame"],
    "wmv": ["-c:v", "msmpeg4v3", "-c:a", "wmav2"],
}


class VideoConverter(BaseConverter):
    """Converter for video containers. Requires the ``ffmpeg`` binary on PATH."""

    SUPPORTED_CONVERSIONS = {
        "mp4": ["avi", "mkv", "mov", "webm", "flv", "wmv"],
        "avi": ["mp4", "mkv", "mov", "webm", "flv", "wmv"],
        "mkv": ["mp4", "avi", "mov", "webm", "flv", "wmv"],
        "mov": ["mp4", "avi", "mkv", "webm", "flv", "wmv"],
        "webm": ["mp4", "avi", "mkv", "mov", "flv", "wmv"],
        "flv": ["mp4", "avi", "mkv", "mov", "webm", "wmv"],
        "wmv": ["mp4", "avi", "mkv", "mov", "webm", "flv"],
    }
    PRIORITY = 20
    REQUIRES_EXTERNAL = ["ffmpeg"]

    def convert(self, task: ConversionTask) -> ConversionResult:
        from ..utils.platform import find_executable, run_command

        if not find_executable("ffmpeg"):
            return ConversionResult(success=False, error=_FFMPEG_HINT)

        target = task.target_format.lower()
        cmd = (
            ["ffmpeg", "-y", "-i", str(task.source_path)]
            + _CODECS.get(target, ["-c:v", "libx264", "-c:a", "aac"])
            + [str(task.target_path)]
        )

        try:
            result = run_command(cmd)
        except Exception as exc:
            return ConversionResult(success=False, error=f"ffmpeg failed: {exc}")

        if result.returncode == 0:
            return ConversionResult(
                success=True,
                output_path=str(task.target_path),
                metadata={"tool": "ffmpeg"},
            )
        return ConversionResult(success=False, error=_tail(result.stderr))

    def convert_format(
        self, path: str, to_format: str, output: Optional[str] = None
    ) -> ConversionResult:
        """Convert a video file to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)

    def extract_thumbnail(
        self,
        video_path: str,
        output: Optional[str] = None,
        timestamp: str = "00:00:01",
    ) -> ConversionResult:
        """Grab a single frame at ``timestamp`` as a still image."""
        from ..utils.platform import find_executable, run_command

        if not find_executable("ffmpeg"):
            return ConversionResult(success=False, error=_FFMPEG_HINT)

        out_path = output or str(Path(video_path).with_suffix(".jpg"))
        # -ss before -i seeks, which is both faster and actually honoured.
        cmd = [
            "ffmpeg", "-y", "-ss", timestamp, "-i", str(video_path),
            "-frames:v", "1", out_path,
        ]

        try:
            result = run_command(cmd)
        except Exception as exc:
            return ConversionResult(success=False, error=f"ffmpeg failed: {exc}")

        if result.returncode == 0:
            return ConversionResult(success=True, output_path=out_path)
        return ConversionResult(success=False, error=_tail(result.stderr))

    def get_info(self, video_path: str) -> ConversionResult:
        """Return the ffprobe metadata for ``video_path`` as a dict."""
        import json

        from ..utils.platform import find_executable, run_command

        if not find_executable("ffprobe"):
            return ConversionResult(
                success=False,
                error="The ffprobe binary was not found on PATH (ships with ffmpeg).",
            )

        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(video_path),
        ]
        try:
            result = run_command(cmd)
        except Exception as exc:
            return ConversionResult(success=False, error=f"ffprobe failed: {exc}")

        if result.returncode != 0:
            return ConversionResult(success=False, error=_tail(result.stderr))

        try:
            return ConversionResult(success=True, data=json.loads(result.stdout or "{}"))
        except json.JSONDecodeError as exc:
            return ConversionResult(
                success=False, error=f"Failed to parse ffprobe output: {exc}"
            )
