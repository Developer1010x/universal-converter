"""Cloud storage and format conversion module"""

from pathlib import Path
from typing import Optional, Dict, Any

from . import BaseConverter, ConversionTask, ConversionResult


class CloudConverter(BaseConverter):
    """Converter for cloud storage formats and configurations"""
    
    SUPPORTED_CONVERSIONS = {
        "tf": ["json", "hcl", "yaml"],
        "tfvars": ["json", "env"],
        "json": ["yaml", "toml", "env"],
        "yaml": ["json", "toml", "toml"],
        "toml": ["json", "yaml"],
        "env": ["json", "yaml"],
    }
    PRIORITY = 20
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()
        
        try:
            if source == "tf" and target in ["json", "hcl"]:
                return self._tf_convert(task)
            elif source == "tfvars":
                return self._tfvars_convert(task)
            elif source in ["json", "yaml", "toml", "env"]:
                return self._config_convert(task)
            
            return ConversionResult(success=False, error=f"Unsupported: {source} → {target}")
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _tf_convert(self, task: ConversionTask) -> ConversionResult:
        content = task.source_path.read_text()
        
        if task.target_format == "json":
            import json
            parsed = self._parse_terraform(content)
            task.target_path.write_text(json.dumps(parsed, indent=2))
        else:
            task.target_path.write_text(content)
        
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _parse_terraform(self, content: str) -> Dict:
        import re
        result = {}
        resources = re.findall(r'resource\s+"([^"]+)"\s+"([^"]+)"\s*\{([^}]+)\}', content)
        
        for res_type, res_name, res_body in resources:
            if res_type not in result:
                result[res_type] = {}
            result[res_type][res_name] = self._parse_block(res_body)
        
        return result
    
    def _parse_block(self, block: str) -> Dict:
        import re
        result = {}
        for line in block.split('\n'):
            match = re.match(r'\s*(\w+)\s*=\s*(.+)', line)
            if match:
                key, value = match.groups()
                result[key] = value.strip('"')
        return result
    
    def _tfvars_convert(self, task: ConversionTask) -> ConversionResult:
        content = task.source_path.read_text()
        
        if task.target_format == "json":
            import json
            parsed = self._parse_tfvars(content)
            task.target_path.write_text(json.dumps(parsed, indent=2))
        elif task.target_format == "env":
            parsed = self._parse_tfvars(content)
            lines = [f"{k}={v}" for k, v in parsed.items()]
            task.target_path.write_text("\n".join(lines))
        else:
            task.target_path.write_text(content)
        
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _parse_tfvars(self, content: str) -> Dict[str, str]:
        import re
        result = {}
        for line in content.split('\n'):
            match = re.match(r'\s*(\w+)\s*=\s*"([^"]*)"', line)
            if match:
                result[match.group(1)] = match.group(2)
        return result
    
    def _config_convert(self, task: ConversionTask) -> ConversionResult:
        import json
        import yaml
        
        content = task.source_path.read_text()
        
        if task.source_format == "json":
            data = json.loads(content)
        elif task.source_format == "yaml":
            data = yaml.safe_load(content)
        elif task.source_format == "env":
            data = self._parse_env(content)
        elif task.source_format == "toml":
            try:
                import tomli
                data = tomli.loads(content)
            except ImportError:
                return ConversionResult(
                    success=False,
                    error="tomli required. Install: pip install universal-converter[cloud]"
                )
        else:
            return ConversionResult(success=False, error="Unknown source format")
        
        if task.target_format == "json":
            task.target_path.write_text(json.dumps(data, indent=2))
        elif task.target_format == "yaml":
            task.target_path.write_text(yaml.dump(data))
        elif task.target_format == "toml":
            try:
                import tomli_w
                task.target_path.write_text(tomli_w.dumps(data))
            except ImportError:
                return ConversionResult(
                    success=False,
                    error="tomli-w required. Install: pip install universal-converter[cloud]"
                )
        elif task.target_format == "env":
            lines = [f"{k}={v}" for k, v in self._flatten_dict(data).items()]
            task.target_path.write_text("\n".join(lines))
        
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _parse_env(self, content: str) -> Dict:
        result = {}
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    result[key.strip()] = value.strip().strip('"')
        return result
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def convert_format(self, path: str, to_format: str, output: Optional[str] = None) -> ConversionResult:
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format
        )
        return self.convert(task)


class TerraformConverter(CloudConverter):
    """Alias for Terraform-specific conversions"""
    pass


class KubernetesConverter(BaseConverter):
    """Converter for Kubernetes manifests"""
    
    SUPPORTED_CONVERSIONS = {
        "yaml": ["json", "helm"],
        "json": ["yaml"],
    }
    PRIORITY = 15
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        import yaml
        import json
        
        try:
            with open(task.source_path) as f:
                data = list(yaml.safe_load_all(f))
            
            if task.target_format == "json":
                output = json.dumps(data, indent=2)
                task.target_path.write_text(output)
            else:
                with open(task.target_path, 'w') as f:
                    yaml.dump_all(data, f, default_flow_style=False)
            
            return ConversionResult(success=True, output_path=str(task.target_path))
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
