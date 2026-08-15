"""Infrastructure and configuration converters (Terraform, JSON/YAML/TOML/env)."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import BaseConverter, ConversionResult, ConversionTask

_YAML_HINT = "PyYAML required. Install: pip install universal-converter[data]"
_TOML_WRITE_HINT = "tomli-w required. Install: pip install universal-converter[toml]"


def _load_toml(content: str) -> Dict[str, Any]:
    """Parse TOML with the stdlib reader on 3.11+, falling back to tomli."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 and older
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(content)


def _load_yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(_YAML_HINT) from exc
    return yaml


class CloudConverter(BaseConverter):
    """Converter for infrastructure configuration formats.

    Handles the config-file square (JSON / YAML / TOML / .env) plus a
    regex-level reader for Terraform ``.tf`` and ``.tfvars`` files. The
    Terraform parsing is intentionally shallow: it lifts top-level resource
    blocks and variable assignments, not the full HCL grammar.
    """

    SUPPORTED_CONVERSIONS = {
        "tf": ["json"],
        "tfvars": ["json", "env"],
        "json": ["yaml", "toml", "env"],
        "yaml": ["json", "toml", "env"],
        "toml": ["json", "yaml", "env"],
        "env": ["json", "yaml"],
    }
    PRIORITY = 30
    REQUIRES_PYTHON = ["yaml"]

    @classmethod
    def requirements_for(cls, source_format: str, target_format: str) -> List[str]:
        needed: List[str] = []
        if "yaml" in (source_format.lower(), target_format.lower()):
            needed.append("yaml")
        if target_format.lower() == "toml":
            needed.append("tomli_w")
        return needed

    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()

        try:
            if source == "tf" and target == "json":
                return self._tf_convert(task)
            if source == "tfvars" and target in ("json", "env"):
                return self._tfvars_convert(task)
            if source in ("json", "yaml", "toml", "env"):
                return self._config_convert(task)

            return ConversionResult(
                success=False, error=f"Unsupported: {source} -> {target}"
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    # --- terraform -------------------------------------------------------

    def _tf_convert(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        parsed = self._parse_terraform(content)
        Path(task.target_path).write_text(
            json.dumps(parsed, indent=2), encoding="utf-8"
        )
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"resource_types": list(parsed)},
        )

    def _parse_terraform(self, content: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for res_type, res_name, body in re.findall(
            r'resource\s+"([^"]+)"\s+"([^"]+)"\s*\{([^}]*)\}', content
        ):
            result.setdefault(res_type, {})[res_name] = self._parse_block(body)
        return result

    @staticmethod
    def _parse_block(block: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for line in block.splitlines():
            match = re.match(r'\s*(\w+)\s*=\s*(.+?)\s*$', line)
            if match:
                result[match.group(1)] = match.group(2).strip().strip('"')
        return result

    def _tfvars_convert(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        parsed = self._parse_tfvars(content)

        if task.target_format.lower() == "json":
            Path(task.target_path).write_text(
                json.dumps(parsed, indent=2), encoding="utf-8"
            )
        else:
            Path(task.target_path).write_text(
                "\n".join(f"{k}={v}" for k, v in parsed.items()) + "\n",
                encoding="utf-8",
            )
        return ConversionResult(success=True, output_path=str(task.target_path))

    @staticmethod
    def _parse_tfvars(content: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for line in content.splitlines():
            match = re.match(r'\s*(\w+)\s*=\s*"([^"]*)"', line)
            if match:
                result[match.group(1)] = match.group(2)
        return result

    # --- config square ---------------------------------------------------

    def _config_convert(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        source = task.source_format.lower()
        target = task.target_format.lower()

        if source == "json":
            data = json.loads(content)
        elif source == "yaml":
            # safe_load_all so multi-document manifests (Kubernetes, Helm
            # output) survive; a single document unwraps back to a mapping.
            documents = list(_load_yaml_module().safe_load_all(content))
            data = documents[0] if len(documents) == 1 else documents
        elif source == "env":
            data = self._parse_env(content)
        else:
            data = _load_toml(content)

        if target == "json":
            rendered = json.dumps(data, indent=2, default=str)
        elif target == "yaml":
            yaml = _load_yaml_module()
            if isinstance(data, list):
                rendered = yaml.safe_dump_all(data, sort_keys=False, default_flow_style=False)
            else:
                rendered = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
        elif target == "toml":
            try:
                import tomli_w
            except ImportError:
                return ConversionResult(success=False, error=_TOML_WRITE_HINT)
            if not isinstance(data, dict):
                return ConversionResult(
                    success=False, error="TOML output requires a mapping at the top level"
                )
            rendered = tomli_w.dumps(data)
        else:
            flat = self._flatten_dict(data if isinstance(data, dict) else {"root": data})
            rendered = "\n".join(f"{k}={v}" for k, v in flat.items()) + "\n"

        Path(task.target_path).write_text(rendered, encoding="utf-8")
        return ConversionResult(success=True, output_path=str(task.target_path))

    @staticmethod
    def _parse_env(content: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key.strip().removeprefix("export ").strip()] = (
                value.strip().strip('"').strip("'")
            )
        return result

    def _flatten_dict(
        self, data: Dict, parent_key: str = "", sep: str = "_"
    ) -> Dict[str, Any]:
        items = []
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep=sep).items())
            elif isinstance(value, (list, tuple)):
                items.append((new_key, json.dumps(list(value), default=str)))
            else:
                items.append((new_key, value))
        return dict(items)

    def convert_format(
        self, path: str, to_format: str, output: Optional[str] = None
    ) -> ConversionResult:
        """Convert a configuration file to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)
