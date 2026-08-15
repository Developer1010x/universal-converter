"""Image conversion module (Pillow-backed)."""

from pathlib import Path
from typing import Optional, Tuple

from . import BaseConverter, ConversionTask, ConversionResult

_PILLOW_HINT = "Pillow required. Install: pip install universal-converter[images]"

# Formats that cannot carry an alpha channel; anything with one is flattened.
_NO_ALPHA = {"jpg", "jpeg", "bmp"}


def _load_pillow():
    """Import Pillow lazily so the package imports without it installed."""
    from PIL import Image  # noqa: WPS433 - deliberate lazy import

    return Image


class ImageConverter(BaseConverter):
    """Converter for raster image formats."""

    SUPPORTED_CONVERSIONS = {
        "png": ["jpg", "gif", "bmp", "tiff", "webp"],
        "jpg": ["png", "gif", "bmp", "tiff", "webp"],
        "gif": ["png", "jpg", "bmp", "tiff", "webp"],
        "bmp": ["png", "jpg", "gif", "tiff", "webp"],
        "tiff": ["png", "jpg", "gif", "bmp", "webp"],
        "webp": ["png", "jpg", "gif", "bmp", "tiff"],
    }
    PRIORITY = 10
    REQUIRES_PYTHON = ["PIL"]

    def convert(self, task: ConversionTask) -> ConversionResult:
        try:
            Image = _load_pillow()
        except ImportError:
            return ConversionResult(success=False, error=_PILLOW_HINT)

        try:
            with Image.open(task.source_path) as img:
                out = self._flatten(img, task.target_format)
                out.save(task.target_path)
                size, mode = img.size, out.mode
            return ConversionResult(
                success=True,
                output_path=str(task.target_path),
                metadata={"size": size, "mode": mode},
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    @staticmethod
    def _flatten(img, target_format: str):
        """Drop the alpha channel when the target format cannot store one."""
        if target_format.lower().lstrip(".") in _NO_ALPHA and img.mode in (
            "RGBA", "LA", "P",
        ):
            return img.convert("RGB")
        return img

    @staticmethod
    def _derive_output(path: str, suffix: str, extension: Optional[str] = None) -> str:
        """Build ``photo.png`` -> ``photo_resized.png``.

        Never returns ``path`` itself: defaulting the output to the input is how
        the previous implementation silently destroyed the source image.
        """
        source = Path(path)
        ext = f".{extension.lstrip('.')}" if extension else source.suffix or ".png"
        return str(source.with_name(f"{source.stem}{suffix}{ext}"))

    def resize(
        self,
        path: str,
        output: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> ConversionResult:
        """Resize an image. Writes to ``<name>_resized.<ext>`` unless ``output``
        is given; the source file is never overwritten implicitly."""
        if not any((width, height, scale)):
            return ConversionResult(
                success=False, error="resize needs one of width, height or scale"
            )
        try:
            Image = _load_pillow()
        except ImportError:
            return ConversionResult(success=False, error=_PILLOW_HINT)

        out_path = output or self._derive_output(path, "_resized")
        if Path(out_path).resolve() == Path(path).resolve():
            return ConversionResult(
                success=False,
                error="output path is the same as the input; refusing to overwrite",
            )

        try:
            with Image.open(path) as img:
                original_width, original_height = img.size

                if scale:
                    width = max(1, int(original_width * scale))
                    height = max(1, int(original_height * scale))
                elif width and not height:
                    height = max(1, round(original_height * (width / original_width)))
                elif height and not width:
                    width = max(1, round(original_width * (height / original_height)))

                resized = img.resize((width, height), Image.Resampling.LANCZOS)
                self._flatten(resized, Path(out_path).suffix[1:]).save(out_path)

            return ConversionResult(
                success=True, output_path=out_path, metadata={"size": (width, height)}
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    def thumbnail(
        self,
        path: str,
        output: Optional[str] = None,
        max_size: Tuple[int, int] = (128, 128),
    ) -> ConversionResult:
        """Write an aspect-preserving thumbnail to ``<name>_thumb.<ext>``."""
        try:
            Image = _load_pillow()
        except ImportError:
            return ConversionResult(success=False, error=_PILLOW_HINT)

        out_path = output or self._derive_output(path, "_thumb")
        if Path(out_path).resolve() == Path(path).resolve():
            return ConversionResult(
                success=False,
                error="output path is the same as the input; refusing to overwrite",
            )

        try:
            with Image.open(path) as img:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                self._flatten(img, Path(out_path).suffix[1:]).save(out_path)
                size = img.size
            return ConversionResult(
                success=True, output_path=out_path, metadata={"size": size}
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    def convert_format(
        self, path: str, to_format: str, output: Optional[str] = None
    ) -> ConversionResult:
        """Convert an image to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)
