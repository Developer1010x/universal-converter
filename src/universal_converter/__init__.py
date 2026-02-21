#!/usr/bin/env python3
"""
Universal Converter - Convert Anything to Anything
OS-neutral, minimal dependencies, comprehensive conversion tool
"""

import os
import sys
import json
import csv
import xml.etree.ElementTree as ET
import base64
import hashlib
import shutil
import tempfile
import subprocess
import logging
import struct
import zipfile
import tarfile
import configparser
import re
import mimetypes
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import traceback
import io
import stat


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversionError(Exception):
    def __init__(self, message: str, source_format: str = None, target_format: str = None, details: str = None):
        self.source_format = source_format
        self.target_format = target_format
        self.details = details
        super().__init__(message)
    
    def __str__(self):
        msg = super().__str__()
        if self.source_format and self.target_format:
            msg = f"{msg} ({self.source_format} → {self.target_format})"
        return msg


@dataclass
class ConversionResult:
    success: bool
    data: Any = None
    output_path: str = None
    error: str = None
    metadata: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class UniversalConverter:
    """Main universal converter - OS neutral, minimal dependencies"""
    
    TEXT_FORMATS = {'json', 'csv', 'xml', 'yaml', 'yml', 'txt', 'html', 'htm', 'markdown', 'md', 'rst', 'sql', 'ini', 'toml', 'url', 'log'}
    DATA_FORMATS = {'json', 'csv', 'tsv', 'xml', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf'}
    DOCUMENT_FORMATS = {'pdf', 'docx', 'doc', 'odt', 'rtf', 'tex', 'epub', 'mobi'}
    IMAGE_FORMATS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'ico', 'ppm', 'pgm', 'pbm', 'jp2', 'svg'}
    ARCHIVE_FORMATS = {'zip', 'tar', 'gz', 'bz2', 'xz', '7z', 'rar'}
    CODE_FORMATS = {'py', 'js', 'ts', 'java', 'c', 'cpp', 'h', 'hpp', 'cs', 'go', 'rs', 'rb', 'php', 'swift', 'kt', 'scala', 'r', 'sql', 'sh', 'bash', 'ps1'}
    
    ALL_FORMATS = TEXT_FORMATS | DATA_FORMATS | DOCUMENT_FORMATS | IMAGE_FORMATS | ARCHIVE_FORMATS | CODE_FORMATS
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def detect_format(self, data: Any) -> Optional[str]:
        if isinstance(data, str):
            path = Path(data)
            if path.exists() and path.is_file():
                ext = path.suffix[1:].lower()
                if ext:
                    return ext
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read(1000)
                    return self._detect_text_format(content)
                except:
                    return None
        if isinstance(data, dict): return 'json'
        if isinstance(data, list): return 'json'
        return None
    
    def _detect_text_format(self, content: str) -> str:
        content = content.strip()
        if content.startswith('{') or content.startswith('['): return 'json'
        if content.startswith('<?xml'): return 'xml'
        if '<!DOCTYPE html>' in content or '<html' in content.lower(): return 'html'
        if content.startswith('#'): return 'markdown'
        if '---' in content and ':' in content: return 'yaml'
        if content.startswith('<?'): return 'xml'
        if ',' in content and '\n' in content: return 'csv'
        return 'txt'
    
    def convert(self, data: Any, from_fmt: str = None, to_fmt: str = None, output_path: str = None, **options) -> ConversionResult:
        if from_fmt is None:
            from_fmt = self.detect_format(data)
        
        if from_fmt is None or to_fmt is None:
            return ConversionResult(success=False, error="Could not detect format")
        
        from_fmt, to_fmt = from_fmt.lower(), to_fmt.lower()
        
        try:
            if isinstance(data, str) and Path(data).exists() and Path(data).is_file():
                return self.convert_file(data, output_path or f"output.{to_fmt}", from_fmt, to_fmt)
            
            result = self._convert_data(data, from_fmt, to_fmt, options)
            
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                mode = 'wb' if isinstance(result, bytes) else 'w'
                encoding = None if isinstance(result, bytes) else 'utf-8'
                with open(output_path, mode, encoding=encoding) as f:
                    f.write(result if isinstance(result, (bytes, str)) else str(result))
                return ConversionResult(success=True, output_path=output_path, metadata={'format': to_fmt})
            
            return ConversionResult(success=True, data=result, metadata={'format': to_fmt})
            
        except Exception as e:
            return ConversionResult(success=False, error=str(e), details=traceback.format_exc())
    
    def _convert_data(self, data: Any, from_fmt: str, to_fmt: str, options: Dict) -> Any:
        data = self._parse_input(data, from_fmt)
        
        converters = {
            'json': self._to_json,
            'csv': self._to_csv,
            'tsv': self._to_tsv,
            'xml': self._to_xml,
            'yaml': self._to_yaml,
            'yml': self._to_yaml,
            'txt': self._to_text,
            'html': self._to_html,
            'markdown': self._to_markdown,
            'md': self._to_markdown,
            'sql': self._to_sql,
            'ini': self._to_ini,
            'toml': self._to_toml,
            'url': self._to_urlencoded,
            'base64': self._to_base64,
            'query': self._to_sql_query,
        }
        
        if to_fmt in converters:
            return converters[to_fmt](data, options)
        
        raise ConversionError(f"Unsupported: {from_fmt} → {to_fmt}", from_fmt, to_fmt)
    
    def _parse_input(self, data: Any, fmt: str) -> Dict:
        if isinstance(data, dict): return data
        if isinstance(data, list): return {'data': data}
        if isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {'text': data}
        return {'value': str(data)}
    
    def _to_json(self, data: Dict, opts: Dict) -> str:
        return json.dumps(data, indent=opts.get('indent', 2), ensure_ascii=False, default=str)
    
    def _to_csv(self, data: Dict, opts: Dict) -> str:
        if isinstance(data, dict):
            if all(isinstance(v, list) for v in data.values()):
                return self._dict_of_lists_to_csv(data)
            data = [data]
        if not isinstance(data, list): data = [data]
        if not data: return ""
        
        keys = set()
        for item in data:
            if isinstance(item, dict): keys.update(item.keys())
        if not keys: return ""
        
        keys = list(keys)
        rows = [','.join(keys)]
        for item in data:
            if isinstance(item, dict):
                rows.append(','.join(self._escape_csv(str(item.get(k, ''))) for k in keys))
        return '\n'.join(rows)
    
    def _escape_csv(self, val: str) -> str:
        if ',' in val or '"' in val or '\n' in val:
            return f'"{val.replace("\"", "\"\"")}"'
        return val
    
    def _dict_of_lists_to_csv(self, data: Dict) -> str:
        max_len = max(len(v) for v in data.values()) if data else 0
        keys = list(data.keys())
        rows = [','.join(keys)]
        for i in range(max_len):
            rows.append(','.join(str(data[k][i] if i < len(data[k]) else '') for k in keys))
        return '\n'.join(rows)
    
    def _to_tsv(self, data: Dict, opts: Dict) -> str:
        csv_str = self._to_csv(data, opts)
        return csv_str.replace(',', '\t')
    
    def _to_xml(self, data: Any, opts: Dict) -> str:
        root = ET.Element('root')
        self._dict_to_xml_element(data, root)
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding='unicode')
    
    def _dict_to_xml_element(self, data: Any, parent: ET.Element):
        if isinstance(data, dict):
            for k, v in data.items():
                child = ET.SubElement(parent, str(k))
                self._dict_to_xml_element(v, child)
        elif isinstance(data, list):
            for item in data:
                child = ET.SubElement(parent, 'item')
                self._dict_to_xml_element(item, child)
        else:
            parent.text = str(data) if data is not None else ''
    
    def _to_yaml(self, data: Any, opts: Dict) -> str:
        return self._yaml_convert(data, 0)
    
    def _yaml_convert(self, data: Any, indent: int) -> str:
        lines = []
        prefix = '  ' * indent
        
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{k}:")
                    lines.append(self._yaml_convert(v, indent + 1))
                else:
                    lines.append(f"{prefix}{k}: {self._yaml_escape(v)}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.append(self._yaml_convert(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {self._yaml_escape(item)}")
        else:
            lines.append(f"{prefix}{self._yaml_escape(data)}")
        
        return '\n'.join(lines)
    
    def _yaml_escape(self, v: Any) -> str:
        if v is None: return 'null'
        if isinstance(v, bool): return 'true' if v else 'false'
        v = str(v)
        if any(c in v for c in ':#[]{},&*!|>\'"%@`') or v.startswith('- ') or v.startswith(' '):
            return f'"{v.replace("\\", "\\\\").replace("\"", "\\\"")}"'
        return v
    
    def _to_text(self, data: Any, opts: Dict) -> str:
        lines = []
        if isinstance(data, dict):
            for k, v in data.items():
                lines.append(f"{k}: {v}")
        elif isinstance(data, list):
            for item in data:
                lines.append(str(item))
        else:
            return str(data)
        return '\n'.join(lines)
    
    def _to_html(self, data: Any, opts: Dict) -> str:
        html = ['<!DOCTYPE html>', '<html>', '<head>',
                '<meta charset="UTF-8"><title>Converted</title>',
                '<style>',
                'body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:900px;margin:0 auto;padding:20px;line-height:1.6}',
                'table{border-collapse:collapse;width:100%;margin:20px 0}',
                'th,td{border:1px solid #ddd;padding:12px;text-align:left}',
                'th{background:#f5f5f5}',
                'code{background:#f4f4f4;padding:2px 6px;border-radius:3px}',
                'pre{background:#f4f4f5;padding:15px;border-radius:6px;overflow-x:auto}',
                '</style></head><body>']
        
        if isinstance(data, dict):
            html.append('<table>')
            for k, v in data.items():
                html.append(f'<tr><th>{k}</th><td>{self._escape_html(v)}</td></tr>')
            html.append('</table>')
        elif isinstance(data, list):
            html.append('<ul>')
            for item in data:
                html.append(f'<li>{self._escape_html(item)}</li>')
            html.append('</ul>')
        else:
            html.append(f'<p>{self._escape_html(data)}</p>')
        
        html.extend(['</body>', '</html>'])
        return '\n'.join(html)
    
    def _escape_html(self, s: Any) -> str:
        s = str(s)
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    
    def _to_markdown(self, data: Any, opts: Dict) -> str:
        lines = ['# Converted Document', '']
        
        if isinstance(data, dict):
            for k, v in data.items():
                lines.append(f"## {k}")
                lines.append(self._md_value(v))
                lines.append('')
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for k, v in item.items():
                        lines.append(f"- **{k}**: {v}")
                else:
                    lines.append(f"- {item}")
            lines.append('')
        else:
            lines.append(str(data))
        
        return '\n'.join(lines)
    
    def _md_value(self, data: Any, indent: int = 0) -> str:
        lines = []
        prefix = '  ' * indent
        
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}- **{k}**:")
                    lines.append(self._md_value(v, indent + 1))
                else:
                    lines.append(f"{prefix}- **{k}**: {v}")
        elif isinstance(data, list):
            for item in data:
                lines.append(f"{prefix}- {item}")
        
        return '\n'.join(lines)
    
    def _to_sql(self, data: Any, opts: Dict) -> str:
        lines = []
        table = opts.get('table', 'data')
        
        if isinstance(data, dict) and 'tables' in data:
            for tbl, tbl_data in data['tables'].items():
                cols = tbl_data.get('columns', [])
                rows = tbl_data.get('data', [])
                col_defs = ', '.join([f"{c} TEXT" for c in cols]) if cols else "id INTEGER PRIMARY KEY"
                lines.append(f"CREATE TABLE {tbl} ({col_defs});")
                for row in rows:
                    if isinstance(row, dict):
                        vals = "', '".join(str(v) for v in row.values())
                        lines.append(f"INSERT INTO {tbl} VALUES ('{vals}');")
        else:
            if not isinstance(data, list): data = [data]
            if data and isinstance(data[0], dict):
                cols = list(data[0].keys())
                col_defs = ', '.join([f"{c} TEXT" for c in cols])
                lines.append(f"CREATE TABLE {table} ({col_defs});")
                for row in data:
                    if isinstance(row, dict):
                        vals = "', '".join(str(v) for v in row.values())
                        lines.append(f"INSERT INTO {table} VALUES ('{vals}');")
        
        return '\n'.join(lines)
    
    def _to_ini(self, data: Dict, opts: Dict) -> str:
        config = configparser.ConfigParser()
        
        if isinstance(data, dict):
            if all(isinstance(v, dict) for v in data.values()):
                for section, values in data.items():
                    config[section] = values
            else:
                config['DEFAULT'] = data
        
        out = io.StringIO()
        config.write(out)
        return out.getvalue()
    
    def _to_toml(self, data: Any, opts: Dict) -> str:
        return self._to_yaml(data, opts)
    
    def _to_urlencoded(self, data: Any, opts: Dict) -> str:
        from urllib.parse import urlencode
        flat = self._flatten_dict(data) if isinstance(data, dict) else {'data': str(data)}
        return urlencode(flat)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)
    
    def _to_base64(self, data: Any, opts: Dict) -> str:
        return base64.b64encode(str(data).encode('utf-8')).decode('utf-8')
    
    def _to_sql_query(self, data: Any, opts: Dict) -> str:
        queries = []
        if not isinstance(data, list): data = [data]
        for item in data:
            if isinstance(item, dict):
                conds = ' AND '.join(f"{k} = '{v}'" for k, v in item.items())
                queries.append(f"SELECT * FROM table WHERE {conds};")
        return '\n'.join(queries)
    
    def convert_file(self, input_path: str, output_path: str = None, from_fmt: str = None, to_fmt: str = None) -> ConversionResult:
        path = Path(input_path)
        
        if not path.exists():
            return ConversionResult(success=False, error=f"File not found: {input_path}")
        
        if from_fmt is None:
            from_fmt = path.suffix[1:].lower()
        
        if output_path is None:
            output_path = str(path.parent / f"{path.stem}.converted")
        
        if to_fmt is None:
            to_fmt = Path(output_path).suffix[1:].lower()
        
        try:
            if from_fmt == to_fmt:
                shutil.copy2(input_path, output_path)
                return ConversionResult(success=True, output_path=output_path)
            
            if from_fmt in self.IMAGE_FORMATS and to_fmt in self.IMAGE_FORMATS:
                return self._convert_image(input_path, output_path, from_fmt, to_fmt)
            
            if from_fmt == 'pdf' and to_fmt == 'txt':
                return self._pdf_to_text(input_path, output_path)
            
            if from_fmt == 'txt' and to_fmt == 'pdf':
                return self._text_to_pdf(input_path, output_path)
            
            if from_fmt == 'pdf' and to_fmt in ('docx', 'doc'):
                return self._convert_pdf_word(input_path, output_path, to_fmt)
            
            if from_fmt in ('docx', 'doc') and to_fmt == 'pdf':
                return self._convert_word_pdf(input_path, output_path)
            
            if from_fmt in ('docx', 'doc') and to_fmt in ('txt', 'html'):
                return self._convert_word(input_path, output_path, to_fmt)
            
            if from_fmt in self.ARCHIVE_FORMATS:
                return self._convert_archive(input_path, output_path, from_fmt, to_fmt)
            
            if from_fmt == 'markdown' and to_fmt == 'html':
                return self.convert(input_path, 'markdown', 'html', output_path)
            
            if from_fmt in self.TEXT_FORMATS or from_fmt in self.DATA_FORMATS:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = f.read()
                return self.convert(data, from_fmt, to_fmt, output_path)
            
            return ConversionResult(success=False, error=f"Unsupported: {from_fmt} → {to_fmt}")
            
        except Exception as e:
            return ConversionResult(success=False, error=str(e), details=traceback.format_exc())
    
    def _convert_image(self, input_path: str, output_path: str, from_fmt: str, to_fmt: str) -> ConversionResult:
        try:
            from PIL import Image
        except ImportError:
            return ConversionResult(success=False, error="Pillow required: pip install pillow")
        
        try:
            img = Image.open(input_path)
            
            if to_fmt in ('jpg', 'jpeg') and img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            img.save(output_path, format=to_fmt.upper())
            return ConversionResult(success=True, output_path=output_path, metadata={'format': to_fmt})
        except Exception as e:
            return ConversionResult(success=False, error=f"Image conversion failed: {e}")
    
    def resize_image(self, input_path: str, output_path: str = None, width: int = None, height: int = None, scale: float = None, keep_aspect: bool = True) -> ConversionResult:
        try:
            from PIL import Image
        except ImportError:
            return ConversionResult(success=False, error="Pillow required: pip install pillow")
        
        try:
            img = Image.open(input_path)
            orig_w, orig_h = img.size
            
            if scale:
                width = int(orig_w * scale)
                height = int(orig_h * scale)
            elif width and height:
                if keep_aspect:
                    img.thumbnail((width, height), Image.LANCZOS)
                    width, height = img.size
                else:
                    pass
            elif width:
                height = int(orig_h * width / orig_w) if keep_aspect else width
            elif height:
                width = int(orig_w * height / orig_h) if keep_aspect else height
            
            if keep_aspect and width and height:
                img.thumbnail((width, height), Image.LANCZOS)
            elif width and height:
                img = img.resize((width, height), Image.LANCZOS)
            
            if output_path is None:
                output_path = input_path
            
            img.save(output_path)
            return ConversionResult(success=True, output_path=output_path, 
                                  metadata={'format': Path(output_path).suffix[1:], 'size': f"{img.size[0]}x{img.size[1]}"})
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _pdf_to_text(self, input_path: str, output_path: str) -> ConversionResult:
        try:
            import PyPDF2
            with open(input_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return ConversionResult(success=True, output_path=output_path, metadata={'format': 'txt'})
        except ImportError:
            return ConversionResult(success=False, error="PyPDF2 required: pip install PyPDF2")
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _text_to_pdf(self, input_path: str, output_path: str) -> ConversionResult:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            c = canvas.Canvas(output_path, pagesize=letter)
            width, height = letter
            text_object = c.beginText(50, height - 50)
            text_object.setFont("Helvetica", 10)
            
            for line in text.split('\n'):
                if line:
                    text_object.textLine(line)
            
            c.drawText(text_object)
            c.save()
            return ConversionResult(success=True, output_path=output_path, metadata={'format': 'pdf'})
        except ImportError:
            return ConversionResult(success=False, error="reportlab required: pip install reportlab")
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _convert_pdf_word(self, input_path: str, output_path: str, to_format: str) -> ConversionResult:
        temp_txt = tempfile.mktemp(suffix='.txt')
        result = self._pdf_to_text(input_path, temp_txt)
        if not result.success:
            return result
        
        try:
            from docx import Document
            doc = Document()
            
            with open(temp_txt, 'r', encoding='utf-8') as f:
                for para in f.read().split('\n\n'):
                    if para.strip():
                        doc.add_paragraph(para.strip())
            
            doc.save(output_path)
            os.unlink(temp_txt)
            return ConversionResult(success=True, output_path=output_path, metadata={'format': to_format})
        except ImportError:
            return ConversionResult(success=False, error="python-docx required: pip install python-docx")
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _convert_word_pdf(self, input_path: str, output_path: str) -> ConversionResult:
        return ConversionResult(success=False, error="Word to PDF requires LibreOffice or pandoc. Install: brew install libreoffice")
    
    def _convert_word(self, input_path: str, output_path: str, to_fmt: str) -> ConversionResult:
        try:
            from docx import Document
            doc = Document(input_path)
            
            if to_fmt == 'txt':
                text = '\n\n'.join(para.text for para in doc.paragraphs)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            elif to_fmt == 'html':
                html = ['<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body>']
                for para in doc.paragraphs:
                    if para.text.strip():
                        html.append(f'<p>{para.text}</p>')
                html.append('</body></html>')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(html))
            
            return ConversionResult(success=True, output_path=output_path, metadata={'format': to_fmt})
        except ImportError:
            return ConversionResult(success=False, error="python-docx required: pip install python-docx")
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _convert_archive(self, input_path: str, output_path: str, from_fmt: str, to_fmt: str) -> ConversionResult:
        try:
            if from_fmt == 'zip' and to_fmt == 'tar':
                with zipfile.ZipFile(input_path, 'r') as z:
                    with tarfile.open(output_path, 'w') as t:
                        for name in z.namelist():
                            t.addfile(z.open(name), arcname=name)
                return ConversionResult(success=True, output_path=output_path)
            
            if from_fmt == 'tar' and to_fmt == 'zip':
                with tarfile.open(input_path, 'r') as t:
                    with zipfile.ZipFile(output_path, 'w') as z:
                        for member in t.getmembers():
                            f = t.extractfile(member)
                            if f:
                                z.writestr(member.name, f.read())
                return ConversionResult(success=True, output_path=output_path)
            
            return ConversionResult(success=False, error=f"Archive conversion {from_fmt} → {to_fmt} not supported")
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def batch_convert(self, input_files: List[str], output_dir: str = None, to_format: str = None) -> List[ConversionResult]:
        results = []
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for input_file in input_files:
            ip = Path(input_file)
            output_format = to_format or ip.suffix[1:].lower()
            op = str(Path(output_dir) / f"{ip.stem}.{output_format}") if output_dir else None
            result = self.convert_file(input_file, op)
            results.append(result)
        
        return results
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {
            'text': sorted(self.TEXT_FORMATS),
            'data': sorted(self.DATA_FORMATS),
            'document': sorted(self.DOCUMENT_FORMATS),
            'image': sorted(self.IMAGE_FORMATS),
            'archive': sorted(self.ARCHIVE_FORMATS),
            'code': sorted(self.CODE_FORMATS),
        }
    
    def all_formats(self) -> List[str]:
        return sorted(self.ALL_FORMATS)


def convert(data: Any, from_fmt: str = None, to_fmt: str = None, output: str = None, **kwargs) -> Any:
    """Quick conversion function"""
    converter = UniversalConverter()
    result = converter.convert(data, from_fmt, to_fmt, output, **kwargs)
    if result.success:
        return result.data or result.output_path
    raise ConversionError(result.error or "Conversion failed")


def convert_file(input_path: str, output_path: str = None, from_fmt: str = None, to_fmt: str = None) -> str:
    """Convert file with format detection"""
    converter = UniversalConverter()
    result = converter.convert_file(input_path, output_path, from_fmt, to_fmt)
    if result.success:
        return result.output_path or result.data
    raise ConversionError(result.error)


def resize_image(input_path: str, output_path: str = None, width: int = None, height: int = None, scale: float = None) -> str:
    """Resize image"""
    converter = UniversalConverter()
    result = converter.resize_image(input_path, output_path, width, height, scale)
    if result.success:
        return result.output_path
    raise ConversionError(result.error)


def batch_convert(files: List[str], output_dir: str = None, to_format: str = 'json') -> List[Dict]:
    """Batch convert files"""
    converter = UniversalConverter()
    results = converter.batch_convert(files, output_dir, to_format)
    return [{'file': f, 'success': r.success, 'output': r.output_path, 'error': r.error} 
            for f, r in zip(files, results)]


def get_formats() -> Dict[str, List[str]]:
    return UniversalConverter().get_supported_formats()


def all_supported_formats() -> List[str]:
    return UniversalConverter().all_formats()


def file_info(file_path: str) -> Dict:
    """Get file information"""
    p = Path(file_path)
    if not p.exists():
        return {'valid': False, 'error': 'File not found'}
    
    stat = p.stat()
    return {
        'valid': True,
        'name': p.name,
        'size': stat.st_size,
        'size_human': f"{stat.st_size / 1024:.2f} KB",
        'format': p.suffix[1:].lower() or None,
        'mime': mimetypes.guess_type(str(p))[0],
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'permissions': oct(stat.st_mode)[-3:],
    }


def hash_file(file_path: str, algorithm: str = 'sha256') -> str:
    """Hash a file"""
    h = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_file(file_path: str) -> Dict:
    """Validate file format"""
    p = Path(file_path)
    if not p.exists():
        return {'valid': False, 'error': 'File not found'}
    
    fmt = p.suffix[1:].lower()
    supported = UniversalConverter().all_formats()
    
    return {
        'valid': fmt in supported,
        'format': fmt,
        'supported': fmt in supported,
        'category': _get_format_category(fmt),
    }


def _get_format_category(fmt: str) -> str:
    c = UniversalConverter()
    if fmt in c.TEXT_FORMATS: return 'text'
    if fmt in c.DATA_FORMATS: return 'data'
    if fmt in c.DOCUMENT_FORMATS: return 'document'
    if fmt in c.IMAGE_FORMATS: return 'image'
    if fmt in c.ARCHIVE_FORMATS: return 'archive'
    if fmt in c.CODE_FORMATS: return 'code'
    return 'unknown'


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Converter - Convert Anything to Anything')
    parser.add_argument('input', help='Input file, data, or directory')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('-f', '--from', dest='from_fmt', help='Input format')
    parser.add_argument('-t', '--to', dest='to_fmt', help='Output format')
    parser.add_argument('-l', '--list', action='store_true', help='List supported formats')
    parser.add_argument('-i', '--info', action='store_true', help='Show file info')
    parser.add_argument('--validate', action='store_true', help='Validate file format')
    parser.add_argument('--hash', choices=['md5', 'sha1', 'sha256', 'sha512'], help='File hash')
    parser.add_argument('--resize', action='store_true', help='Resize image')
    parser.add_argument('--width', type=int, help='Target width')
    parser.add_argument('--height', type=int, help='Target height')
    parser.add_argument('--scale', type=float, help='Scale factor (e.g., 0.5)')
    parser.add_argument('--batch', nargs='+', help='Batch convert')
    parser.add_argument('--output-dir', help='Output directory')
    
    args = parser.parse_args()
    
    if args.list:
        formats = get_formats()
        print("Supported Formats:")
        for cat, fmts in formats.items():
            print(f"\n{cat.upper()} ({len(fmts)}):")
            print("  " + ", ".join(fmts))
        return
    
    if args.info:
        print(json.dumps(file_info(args.input), indent=2))
        return
    
    if args.validate:
        print(json.dumps(validate_file(args.input), indent=2))
        return
    
    if args.hash:
        print(f"{args.hash}: {hash_file(args.input, args.hash)}")
        return
    
    if args.resize:
        result = resize_image(args.input, args.output, args.width, args.height, args.scale)
        print(f"Resized: {result}")
        return
    
    if args.batch:
        results = batch_convert(args.batch, args.output_dir, args.to_fmt or 'json')
        for r in results:
            status = "✓" if r['success'] else "✗"
            print(f"{status} {r['file']} → {r.get('output') or r.get('error')}")
        return
    
    try:
        result = convert_file(args.input, args.output, args.from_fmt, args.to_fmt)
        print(f"Converted: {result}")
    except ConversionError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
