"""Route planning and execution on top of the converter registry.

A conversion request is resolved to a *route*: one step when a converter
declares the pair directly, or several when it has to be chained
(``json -> html -> md``). Intermediate files live in a temporary directory and
only the final step writes to the destination the caller asked for.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .converters import ConversionResult, ConversionTask
from .registry import RouteStep, format_route, get_registry


@dataclass
class PipelineResult:
    """Outcome of a (possibly multi-step) conversion."""

    success: bool
    output_path: Optional[str] = None
    route: List[RouteStep] = field(default_factory=list)
    results: List[ConversionResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def hops(self) -> int:
        """Number of conversion steps that were executed."""
        return len(self.route)

    @property
    def route_str(self) -> str:
        """The route rendered as ``json -> html -> md``."""
        return format_route(self.route)


def plan(
    source_format: str,
    target_format: str,
    max_hops: int = 3,
    allow_multi_hop: bool = True,
    require_available: bool = True,
) -> Optional[List[RouteStep]]:
    """Return the route for a format pair, or ``None`` if there is none."""
    registry = get_registry()
    if not allow_multi_hop:
        converter = registry.find_converter(source_format, target_format)
        if converter is None:
            return None
        return [
            RouteStep(
                registry.normalize(source_format),
                registry.normalize(target_format),
                converter,
            )
        ]

    route = registry.find_route(
        source_format, target_format, max_hops=max_hops, require_available=True
    )
    if route is None and not require_available:
        # Fall back to a dependency-blind route so the caller can still be told
        # what *would* work, and which package is missing.
        route = registry.find_route(source_format, target_format, max_hops=max_hops)
    return route


def convert_path(
    source: str,
    output: Optional[str] = None,
    source_format: Optional[str] = None,
    target_format: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    max_hops: int = 3,
    allow_multi_hop: bool = True,
) -> PipelineResult:
    """Convert ``source`` to ``output``, chaining converters when needed."""
    registry = get_registry()
    source_path = Path(source)

    if not source_path.exists():
        return PipelineResult(success=False, error=f"no such file: {source}")
    if output and Path(output).resolve() == source_path.resolve():
        # Checked before format resolution so the message names the real
        # problem rather than "source and target format are both json".
        return PipelineResult(
            success=False, error="output path is the same as the input file"
        )

    src_fmt = registry.normalize(source_format or registry.detect_format(source) or "")
    if not src_fmt:
        return PipelineResult(
            success=False,
            error=f"cannot infer the source format of {source}; pass --from",
        )

    if target_format:
        tgt_fmt = registry.normalize(target_format)
    elif output:
        tgt_fmt = registry.normalize(registry.detect_format(output) or "")
    else:
        tgt_fmt = ""
    if not tgt_fmt:
        return PipelineResult(
            success=False, error="no target format; pass --to or an output path"
        )

    if src_fmt == tgt_fmt:
        return PipelineResult(
            success=False, error=f"source and target format are both '{src_fmt}'"
        )

    route = plan(
        src_fmt, tgt_fmt, max_hops=max_hops, allow_multi_hop=allow_multi_hop,
        require_available=False,
    )
    if not route:
        return PipelineResult(
            success=False,
            error=(
                f"no conversion path from {src_fmt} to {tgt_fmt} "
                f"(searched up to {max_hops} hops)"
            ),
        )

    target_path = Path(output) if output else source_path.with_suffix(f".{tgt_fmt}")
    if target_path.resolve() == source_path.resolve():
        return PipelineResult(
            success=False, error="output path is the same as the input file"
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    return _execute(route, source_path, target_path, options or {})


def _execute(
    route: List[RouteStep],
    source_path: Path,
    target_path: Path,
    options: Dict[str, Any],
) -> PipelineResult:
    results: List[ConversionResult] = []

    with tempfile.TemporaryDirectory(prefix="universal-converter-") as tmpdir:
        current = source_path
        for index, step in enumerate(route):
            is_last = index == len(route) - 1
            destination = (
                target_path
                if is_last
                else Path(tmpdir) / f"step{index}_{current.stem}.{step.target}"
            )

            task = ConversionTask(
                source_path=current,
                target_path=destination,
                source_format=step.source,
                target_format=step.target,
                options=dict(options),
            )
            result = step.converter().convert(task)
            results.append(result)

            if not result.success:
                return PipelineResult(
                    success=False,
                    route=route,
                    results=results,
                    error=(
                        f"step {index + 1}/{len(route)} "
                        f"({step.source} -> {step.target}, "
                        f"{step.converter.__name__}): {result.error}"
                    ),
                )
            # A converter may write somewhere other than target_path (the
            # SQLite -> CSV fan-out does); follow whatever it reports.
            current = Path(result.output_path or destination)
            if not is_last and not current.exists():
                return PipelineResult(
                    success=False,
                    route=route,
                    results=results,
                    error=f"step {index + 1} reported success but wrote no file",
                )

    return PipelineResult(
        success=True, output_path=str(current), route=route, results=results
    )
