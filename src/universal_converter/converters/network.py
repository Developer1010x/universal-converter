"""Network and web format conversion module"""

from pathlib import Path
from typing import Optional, Dict, Any

from . import BaseConverter, ConversionTask, ConversionResult


class NetworkConverter(BaseConverter):
    """Converter for network and web file formats"""
    
    SUPPORTED_CONVERSIONS = {
        "html": ["markdown", "text", "json"],
        "markdown": ["html", "text"],
        "text": ["html", "markdown"],
        "xml": ["json", "yaml"],
        "json": ["yaml", "xml", "toml"],
        "url": ["json"],
    }
    PRIORITY = 10
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()
        
        try:
            if source == "html" and target in ["markdown", "text"]:
                return self._html_convert(task)
            elif source == "markdown" and target == "html":
                return self._markdown_to_html(task)
            elif source in ["xml", "json"] and target in ["json", "yaml", "xml"]:
                return self._data_convert(task)
            elif source == "url":
                return self._url_convert(task)
            
            return ConversionResult(success=False, error=f"Unsupported: {source} → {target}")
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _html_convert(self, task: ConversionTask) -> ConversionResult:
        import re
        
        content = task.source_path.read_text()
        
        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        if task.target_format == "markdown":
            text = self._html_to_md(text)
        
        task.target_path.write_text(text)
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _html_to_md(self, html: str) -> str:
        import re
        
        md = html
        md = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n', md, flags=re.DOTALL)
        md = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n', md, flags=re.DOTALL)
        md = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n', md, flags=re.DOTALL)
        md = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', md, flags=re.DOTALL)
        md = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', md, flags=re.DOTALL)
        md = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', md, flags=re.DOTALL)
        md = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', md, flags=re.DOTALL)
        md = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', md, flags=re.DOTALL)
        md = re.sub(r'<br\s*/?>', '\n', md, flags=re.IGNORECASE)
        md = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', md, flags=re.DOTALL)
        md = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', md, flags=re.DOTALL)
        md = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', md, flags=re.DOTALL)
        md = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```', md, flags=re.DOTALL)
        md = re.sub(r'<[^>]+>', '', md)
        
        return md.strip()
    
    def _markdown_to_html(self, task: ConversionTask) -> ConversionResult:
        content = task.source_path.read_text()
        
        import re
        html = content
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)
        
        full_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Converted</title></head>
<body>
{html}
</body>
</html>"""
        
        task.target_path.write_text(full_html)
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _data_convert(self, task: ConversionTask) -> ConversionResult:
        import json
        import yaml
        
        content = task.source_path.read_text()
        
        if task.source_format == "json":
            data = json.loads(content)
        elif task.source_format == "xml":
            data = self._xml_to_dict(content)
        else:
            return ConversionResult(success=False, error="Unknown source format")
        
        if task.target_format == "json":
            task.target_path.write_text(json.dumps(data, indent=2))
        elif task.target_format == "yaml":
            task.target_path.write_text(yaml.dump(data))
        elif task.target_format == "xml":
            xml = self._dict_to_xml(data)
            task.target_path.write_text(xml)
        
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _xml_to_dict(self, xml_str: str) -> Dict:
        import xml.etree.ElementTree as ET
        
        root = ET.fromstring(xml_str)
        return {root.tag: self._parse_element(root)}
    
    def _parse_element(self, element) -> Any:
        result = {}
        if element.text and element.text.strip():
            return element.text.strip()
        
        for child in element:
            child_data = self._parse_element(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result if result else None
    
    def _dict_to_xml(self, data: Dict, root_tag: str = "root") -> str:
        import xml.etree.ElementTree as ET
        
        root = ET.Element(root_tag)
        self._build_element(root, data)
        
        return ET.tostring(root, encoding='unicode')
    
    def _build_element(self, parent, data):
        import xml.etree.ElementTree as ET
        
        if isinstance(data, dict):
            for key, value in data.items():
                child = ET.SubElement(parent, key)
                self._build_element(child, value)
        elif isinstance(data, list):
            for item in data:
                child = ET.SubElement(parent, "item")
                self._build_element(child, item)
        else:
            parent.text = str(data)
    
    def _url_convert(self, task: ConversionTask) -> ConversionResult:
        import json
        from urllib.parse import urlparse, parse_qs
        
        content = task.source_path.read_text().strip()
        
        if not content.startswith(('http://', 'https://')):
            parsed = urlparse(content)
        else:
            parsed = urlparse(content)
        
        url_data = {
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "path": parsed.path,
            "params": parsed.params,
            "query": parse_qs(parsed.query),
            "fragment": parsed.fragment
        }
        
        task.target_path.write_text(json.dumps(url_data, indent=2))
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def convert_format(self, path: str, to_format: str, output: Optional[str] = None) -> ConversionResult:
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format
        )
        return self.convert(task)


class HTTPConverter(NetworkConverter):
    """Converter for HTTP formats"""
    pass
