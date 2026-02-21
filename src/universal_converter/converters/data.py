"""Data format converters (JSON, CSV, XML, YAML, etc.)"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Union
from ..converters import BaseConverter, ConversionTask, ConversionResult, DependencyError


class DataConverter(BaseConverter):
    """Handles data format conversions"""
    
    SUPPORTED_CONVERSIONS = {
        'json': ['csv', 'xml', 'yaml', 'txt', 'html'],
        'csv': ['json', 'xml', 'yaml', 'tsv'],
        'xml': ['json', 'csv', 'yaml'],
        'yaml': ['json', 'csv', 'xml'],
        'txt': ['json', 'csv'],
    }
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        try:
            data = self._read_input(task.source_path, task.source_format)
            result = self._convert_data(data, task.source_format, task.target_format)
            
            with open(task.target_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            return ConversionResult(success=True, output_path=str(task.target_path))
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _read_input(self, path: Path, fmt: str) -> Union[Dict, List]:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if fmt == 'json':
            return json.loads(content)
        elif fmt == 'csv':
            return list(csv.DictReader(content.splitlines()))
        elif fmt == 'xml':
            return self._parse_xml(content)
        elif fmt == 'yaml':
            return self._parse_yaml(content)
        return {'text': content}
    
    def _parse_xml(self, content: str) -> Dict:
        root = ET.fromstring(content)
        return {root.tag: self._xml_to_dict(root)}
    
    def _xml_to_dict(self, element) -> Dict:
        result = {}
        for child in element:
            result[child.tag] = child.text
        return result
    
    def _parse_yaml(self, content: str) -> Dict:
        try:
            import yaml
            return yaml.safe_load(content)
        except ImportError:
            return {'content': content}
    
    def _convert_data(self, data: Any, from_fmt: str, to_fmt: str) -> str:
        if to_fmt == 'json':
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        elif to_fmt == 'csv':
            return self._to_csv(data)
        elif to_fmt == 'xml':
            return self._to_xml(data)
        elif to_fmt == 'yaml':
            return self._to_yaml(data)
        elif to_fmt == 'txt':
            return self._to_text(data)
        return str(data)
    
    def _to_csv(self, data: Union[Dict, List]) -> str:
        if isinstance(data, dict):
            data = [data] if data else []
        if not data:
            return ""
        
        keys = list(data[0].keys()) if isinstance(data[0], dict) else ['value']
        rows = [','.join(keys)]
        for item in data:
            if isinstance(item, dict):
                rows.append(','.join(str(item.get(k, '')) for k in keys))
            else:
                rows.append(str(item))
        return '\n'.join(rows)
    
    def _to_xml(self, data: Any) -> str:
        root = ET.Element('root')
        self._dict_to_xml(data, root)
        return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding='unicode')
    
    def _dict_to_xml(self, data: Any, parent: ET.Element):
        if isinstance(data, dict):
            for k, v in data.items():
                child = ET.SubElement(parent, str(k))
                self._dict_to_xml(v, child)
        elif isinstance(data, list):
            for item in data:
                child = ET.SubElement(parent, 'item')
                self._dict_to_xml(item, child)
        else:
            parent.text = str(data) if data is not None else ''
    
    def _to_yaml(self, data: Any, indent: int = 0) -> str:
        lines = []
        prefix = '  ' * indent
        
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{k}:")
                    lines.append(self._to_yaml(v, indent + 1))
                else:
                    lines.append(f"{prefix}{k}: {v}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(self._to_yaml(item, indent))
                else:
                    lines.append(f"{prefix}- {item}")
        
        return '\n'.join(lines)
    
    def _to_text(self, data: Any) -> str:
        if isinstance(data, dict):
            return '\n'.join(f"{k}: {v}" for k, v in data.items())
        elif isinstance(data, list):
            return '\n'.join(str(item) for item in data)
        return str(data)


def convert_json_to_csv(json_path: str, csv_path: str) -> str:
    """Convert JSON to CSV"""
    task = ConversionTask(
        source_path=Path(json_path),
        target_path=Path(csv_path),
        source_format='json',
        target_format='csv'
    )
    result = DataConverter().convert(task)
    if result.success:
        return result.output_path
    raise Exception(result.error)


def convert_csv_to_json(csv_path: str, json_path: str) -> str:
    """Convert CSV to JSON"""
    task = ConversionTask(
        source_path=Path(csv_path),
        target_path=Path(json_path),
        source_format='csv',
        target_format='json'
    )
    result = DataConverter().convert(task)
    if result.success:
        return result.output_path
    raise Exception(result.error)


def convert_json_to_xml(json_path: str, xml_path: str) -> str:
    """Convert JSON to XML"""
    task = ConversionTask(
        source_path=Path(json_path),
        target_path=Path(xml_path),
        source_format='json',
        target_format='xml'
    )
    result = DataConverter().convert(task)
    if result.success:
        return result.output_path
    raise Exception(result.error)
