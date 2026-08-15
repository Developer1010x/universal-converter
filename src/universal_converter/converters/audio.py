"""Audio conversion module (ffmpeg-backed)."""

from pathlib import Path
from typing import Optional

from . import BaseConverter, ConversionTask, ConversionResult

_FFMPEG_HINT = (
    "The ffmpeg binary was not found on PATH. It is a system package, not a "
    "Python one: apt install ffmpeg / brew install ffmpeg / winget install ffmpeg"
)


def _tail(text: Optional[str], lines: int = 6) -> str:
    """Return the last few lines of ffmpeg's stderr, which hold the real error."""
    if not text:
        return "conversion failed"
    return "\n".join(text.strip().splitlines()[-lines:])


class AudioConverter(BaseConverter):
    """Converter for audio formats. Requires the ``ffmpeg`` binary on PATH."""

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
            return ConversionResult(success=False, error=_FFMPEG_HINT)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(task.source_path),
            str(task.target_path),
        ]

        try:
            result = run_command(cmd)
        except Exception as exc:  # timeout, OSError, ...
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
        """Convert an audio file to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)

    def extract_audio(
        self, video_path: str, output: Optional[str] = None, format: str = "mp3"
    ) -> ConversionResult:
        """Strip the video stream and keep the audio track."""
        from ..utils.platform import find_executable, run_command

        if not find_executable("ffmpeg"):
            return ConversionResult(success=False, error=_FFMPEG_HINT)

        out_path = output or str(Path(video_path).with_suffix(f".{format}"))
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", out_path]

        try:
            result = run_command(cmd)
        except Exception as exc:
            return ConversionResult(success=False, error=f"ffmpeg failed: {exc}")

        if result.returncode == 0:
            return ConversionResult(success=True, output_path=out_path)
        return ConversionResult(success=False, error=_tail(result.stderr))
