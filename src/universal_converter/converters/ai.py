"""Model conversion module.

Only one pair is implemented: PyTorch -> ONNX, via ``torch.onnx.export``.
The h5/tflite/sklearn pairs this module used to advertise were never
implemented -- each imported a library and then returned ``success=False``
unconditionally -- so they have been removed rather than left in the registry.
"""

from pathlib import Path
from typing import List, Optional

from . import BaseConverter, ConversionResult, ConversionTask

_TORCH_HINT = "PyTorch and ONNX required. Install: pip install universal-converter[torch]"


class AIConverter(BaseConverter):
    """Export a serialized PyTorch model to ONNX."""

    SUPPORTED_CONVERSIONS = {
        "pt": ["onnx"],
        "pth": ["onnx"],
    }
    PRIORITY = 35
    REQUIRES_PYTHON = ["torch", "onnx"]

    @classmethod
    def requirements_for(cls, source_format: str, target_format: str) -> List[str]:
        return ["torch", "onnx"]

    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()

        if source in ("pt", "pth") and target == "onnx":
            return self._pt_to_onnx(task)
        return ConversionResult(success=False, error=f"Unsupported: {source} -> {target}")

    def _pt_to_onnx(self, task: ConversionTask) -> ConversionResult:
        try:
            import torch
        except ImportError:
            return ConversionResult(success=False, error=_TORCH_HINT)

        model = self._load_model(task, torch)
        if isinstance(model, ConversionResult):
            return model

        # A serialized model carries no input signature, so the caller has to
        # say what shape to trace with. Default matches a standard image model.
        shape = tuple(task.options.get("input_shape", (1, 3, 224, 224)))
        opset = int(task.options.get("opset", 17))

        try:
            model.eval()
            torch.onnx.export(
                model,
                torch.randn(*shape),
                str(task.target_path),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"input_shape": shape, "opset": opset},
        )

    @staticmethod
    def _load_model(task: ConversionTask, torch):
        """Load a model without silently unpickling untrusted input.

        ``torch.load(..., weights_only=False)`` executes arbitrary code from the
        file, so it is only reachable when the caller explicitly opts in with
        ``options={"trust_pickle": True}``. TorchScript archives are tried first
        because they need no such opt-in.
        """
        path = str(task.source_path)
        try:
            return torch.jit.load(path)
        except Exception:
            pass

        if not task.options.get("trust_pickle"):
            return ConversionResult(
                success=False,
                error=(
                    f"{path} is not a TorchScript archive. Loading a pickled "
                    "checkpoint executes code from the file; re-run with "
                    "options={'trust_pickle': True} only if you trust its origin, "
                    "or export it with torch.jit.script/torch.jit.trace first."
                ),
            )
        try:
            return torch.load(path, weights_only=False)
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    def convert_format(
        self, path: str, to_format: str, output: Optional[str] = None
    ) -> ConversionResult:
        """Convert a model file to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)
