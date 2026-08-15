"""Structured data converters (JSON, CSV, TSV, XML, YAML, text, HTML)."""

import csv
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Union

from . import BaseConverter, ConversionResult, ConversionTask

_YAML_HINT = "PyYAML required. Install: pip install universal-converter[data]"

_DELIMITERS = {"csv": ",", "tsv": "\t"}


class DataConverter(BaseConverter):
    """Generic structured-data converter.

    This is the fallback converter (``PRIORITY`` 50): every specialised
    converter outranks it, and it picks up whatever is left.
    """

    SUPPORTED_CONVERSIONS = {
        'json': ['csv', 'tsv', 'xml', 'yaml', 'txt', 'html', 'md'],
        'csv': ['json', 'tsv', 'xml', 'yaml', 'html', 'md'],
        'tsv': ['json', 'csv', 'xml', 'yaml', 'html', 'md'],
        'xml': ['json', 'csv', 'yaml', 'html', 'md'],
        'yaml': ['json', 'csv', 'xml', 'html', 'md'],
        'txt': ['json', 'csv'],
    }
    PRIORITY = 50
    REQUIRES_PYTHON = ["yaml"]

    @classmethod
    def requirements_for(cls, source_format: str, target_format: str) -> List[str]:
        # Only the YAML pairs need PyYAML; everything else is stdlib.
        return (
            ["yaml"]
            if "yaml" in (source_format.lower(), target_format.lower())
            else []
        )

    def convert(self, task: ConversionTask) -> ConversionResult:
        try:
            data = self._read_input(task.source_path, task.source_format.lower())
            rendered = self._convert_data(data, task.target_format.lower())

            Path(task.target_path).write_text(rendered, encoding='utf-8')
            return ConversionResult(success=True, output_path=str(task.target_path))
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    # --- reading ---------------------------------------------------------

    def _read_input(self, path: Path, fmt: str) -> Union[Dict, List]:
        content = Path(path).read_text(encoding='utf-8')

        if fmt == 'json':
            return json.loads(content)
        if fmt in _DELIMITERS:
            # StringIO (not splitlines) so quoted fields may contain newlines.
            reader = csv.DictReader(io.StringIO(content), delimiter=_DELIMITERS[fmt])
            return [dict(row) for row in reader]
        if fmt == 'xml':
            return self._parse_xml(content)
        if fmt == 'yaml':
            return self._parse_yaml(content)
        return {'text': content}

    def _parse_xml(self, content: str) -> Dict:
        root = ET.fromstring(content)
        return {root.tag: self._xml_to_obj(root)}

    def _xml_to_obj(self, element) -> Any:
        """Recursively turn an element into dicts/lists/strings.

        Repeated child tags collapse into a list, so ``<r><item>a</item>
        <item>b</item></r>`` round-trips back to ``["a", "b"]``.
        """
        children = list(element)
        if not children:
            return (element.text or '').strip()

        if all(child.tag == children[0].tag for child in children) and len(children) > 1:
            return [self._xml_to_obj(child) for child in children]

        result: Dict[str, Any] = {}
        for child in children:
            value = self._xml_to_obj(child)
            if child.tag in result:
                existing = result[child.tag]
                if not isinstance(existing, list):
                    result[child.tag] = [existing]
                result[child.tag].append(value)
            else:
                result[child.tag] = value
        return result

    def _parse_yaml(self, content: str) -> Any:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(_YAML_HINT) from exc
        return yaml.safe_load(content)

    # --- writing ---------------------------------------------------------

    def _convert_data(self, data: Any, to_fmt: str) -> str:
        if to_fmt == 'json':
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        if to_fmt in _DELIMITERS:
            return self._to_csv(data, delimiter=_DELIMITERS[to_fmt])
        if to_fmt == 'xml':
            return self._to_xml(data)
        if to_fmt == 'yaml':
            return self._to_yaml(data)
        if to_fmt == 'txt':
            return self._to_text(data)
        if to_fmt == 'html':
            return self._to_html(data)
        if to_fmt == 'md':
            return self._to_markdown(data)
        raise ValueError(f"unsupported target format: {to_fmt}")

    @staticmethod
    def _record_keys(data: Any) -> List[str]:
        """Return the union of keys if ``data`` is a list of mappings, else []."""
        if not isinstance(data, list) or not data:
            return []
        if not all(isinstance(row, dict) for row in data):
            return []
        keys: List[str] = []
        for row in data:
            for key in row:
                if key not in keys:
                    keys.append(str(key))
        return keys

    def _to_markdown(self, data: Any) -> str:
        """Render data as GitHub-flavoured Markdown.

        A list of records becomes one table; a mapping becomes a two-column
        key/value table; anything else becomes a bullet list.
        """
        def cell(value: Any) -> str:
            return self._scalar(value).replace("|", r"\|").replace("\n", " ")

        keys = self._record_keys(data)
        if keys:
            lines = [
                "| " + " | ".join(keys) + " |",
                "|" + "|".join([" --- "] * len(keys)) + "|",
            ]
            lines += [
                "| " + " | ".join(cell(row.get(k)) for k in keys) + " |" for row in data
            ]
            return "\n".join(lines) + "\n"

        if isinstance(data, dict):
            lines = ["| key | value |", "| --- | --- |"]
            lines += [f"| {cell(k)} | {cell(v)} |" for k, v in data.items()]
            return "\n".join(lines) + "\n"

        if isinstance(data, list):
            return "\n".join(f"- {cell(item)}" for item in data) + "\n"
        return self._scalar(data) + "\n"

    @staticmethod
    def _scalar(value: Any) -> str:
        """Render a cell. Nested values become compact JSON, not a Python repr."""
        if value is None:
            return ''
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return str(value)

    def _to_csv(self, data: Union[Dict, List], delimiter: str = ',') -> str:
        if isinstance(data, dict):
            data = [data]
        if not data:
            return ""

        buffer = io.StringIO()
        if all(isinstance(row, dict) for row in data):
            # Union of every key, in first-seen order: a key that only appears
            # on a later row must not be silently dropped.
            keys: List[str] = []
            for row in data:
                for key in row:
                    if key not in keys:
                        keys.append(key)
            writer = csv.DictWriter(
                buffer, fieldnames=keys, delimiter=delimiter, lineterminator='\n'
            )
            writer.writeheader()
            for row in data:
                writer.writerow({k: self._scalar(row.get(k)) for k in keys})
        else:
            writer = csv.writer(buffer, delimiter=delimiter, lineterminator='\n')
            writer.writerow(['value'])
            for item in data:
                writer.writerow([self._scalar(item)])
        return buffer.getvalue()

    def _to_html(self, data: Any) -> str:
        """Render data as a self-contained, styled HTML document.

        A list of records is rendered as one table with a shared header row,
        which is what makes converted JSON/CSV actually readable in a browser;
        everything else falls back to nested key/value tables and lists.
        """
        from html import escape

        def render(value: Any) -> str:
            if isinstance(value, dict):
                cells = ''.join(
                    f"<tr><th>{escape(str(k))}</th><td>{render(v)}</td></tr>"
                    for k, v in value.items()
                )
                return f"<table>{cells}</table>"
            if isinstance(value, (list, tuple)):
                items = ''.join(f"<li>{render(v)}</li>" for v in value)
                return f"<ol>{items}</ol>"
            if value is None:
                return '<span class="null">null</span>'
            if isinstance(value, bool):
                return f'<span class="bool">{str(value).lower()}</span>'
            if isinstance(value, (int, float)):
                return f'<span class="num">{value}</span>'
            return escape(str(value))

        keys = self._record_keys(data)
        if keys:
            head = ''.join(f"<th>{escape(k)}</th>" for k in keys)
            body = ''.join(
                "<tr>"
                # An absent key renders empty; only an explicit JSON null
                # renders as null. Conflating the two would be a lie.
                + ''.join(
                    f"<td>{render(row[k]) if k in row else ''}</td>" for k in keys
                )
                + "</tr>"
                for row in data
            )
            table = (
                '<table class="records"><thead><tr>'
                f"{head}</tr></thead><tbody>{body}</tbody></table>"
            )
            caption = f"<p class=\"meta\">{len(data)} records &middot; {len(keys)} fields</p>"
            return _HTML_TEMPLATE.format(body=caption + table)

        return _HTML_TEMPLATE.format(body=render(data))

    def _to_xml(self, data: Any) -> str:
        root = ET.Element('root')
        self._obj_to_xml(data, root)
        ET.indent(root, space="  ")
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
            root, encoding='unicode'
        )

    def _obj_to_xml(self, data: Any, parent: ET.Element) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                child = ET.SubElement(parent, self._safe_tag(str(key)))
                self._obj_to_xml(value, child)
        elif isinstance(data, (list, tuple)):
            for item in data:
                child = ET.SubElement(parent, 'item')
                self._obj_to_xml(item, child)
        else:
            parent.text = '' if data is None else str(data)

    @staticmethod
    def _safe_tag(name: str) -> str:
        """Coerce an arbitrary key into a legal XML element name."""
        cleaned = ''.join(c if (c.isalnum() or c in '._-') else '_' for c in name)
        if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == '_'):
            cleaned = f"_{cleaned}"
        return cleaned

    def _to_yaml(self, data: Any) -> str:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(_YAML_HINT) from exc
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)

    def _to_text(self, data: Any) -> str:
        if isinstance(data, dict):
            return '\n'.join(f"{k}: {self._scalar(v)}" for k, v in data.items())
        if isinstance(data, list):
            return '\n'.join(self._scalar(item) for item in data)
        return self._scalar(data)

    def convert_format(
        self, path: str, to_format: str, output: str = None
    ) -> ConversionResult:
        """Convert a data file to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>universal-converter output</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif;
         margin: 2rem auto; max-width: 60rem; padding: 0 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: .25rem 0; }}
  th, td {{ border: 1px solid #8884; padding: .35rem .6rem; text-align: left;
           vertical-align: top; }}
  th {{ width: 1%; white-space: nowrap; font-weight: 600; background: #8881; }}
  table.records th {{ width: auto; }}
  table.records tbody tr:nth-child(even) {{ background: #8880; }}
  .meta {{ color: #888; font-size: 12px; margin: 0 0 .5rem; }}
  ol {{ margin: .25rem 0; padding-left: 1.5rem; }}
  .num {{ color: #0a7; font-variant-numeric: tabular-nums; }}
  .bool {{ color: #a70; }}
  .null {{ color: #888; font-style: italic; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def convert_json_to_csv(json_path: str, csv_path: str) -> str:
    """Convert a JSON file to CSV, returning the output path."""
    return _run(json_path, csv_path, 'json', 'csv')


def convert_csv_to_json(csv_path: str, json_path: str) -> str:
    """Convert a CSV file to JSON, returning the output path."""
    return _run(csv_path, json_path, 'csv', 'json')


def convert_json_to_xml(json_path: str, xml_path: str) -> str:
    """Convert a JSON file to XML, returning the output path."""
    return _run(json_path, xml_path, 'json', 'xml')


def _run(source: str, target: str, source_fmt: str, target_fmt: str) -> str:
    from . import ConversionError

    task = ConversionTask(
        source_path=Path(source),
        target_path=Path(target),
        source_format=source_fmt,
        target_format=target_fmt,
    )
    result = DataConverter().convert(task)
    if result.success:
        return result.output_path
    raise ConversionError(result.error, source_fmt, target_fmt)
