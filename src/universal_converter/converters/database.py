"""Database format conversion module"""

from pathlib import Path
from typing import Optional, Dict, Any, List

from . import BaseConverter, ConversionTask, ConversionResult


class DatabaseConverter(BaseConverter):
    """Converter for database and data interchange formats"""
    
    SUPPORTED_CONVERSIONS = {
        "sql": ["json", "csv", "markdown"],
        "sqlite": ["sql", "csv", "json"],
        "db": ["sql", "csv", "json"],
        "xlsx": ["csv", "json", "html"],
        "xls": ["csv", "json"],
        "ods": ["csv", "xlsx"],
    }
    PRIORITY = 20
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()
        
        try:
            if source in ["xlsx", "xls", "ods"] and target in ["csv", "json", "html"]:
                return self._spreadsheet_convert(task)
            elif source in ["sqlite", "db"] and target in ["sql", "csv", "json"]:
                return self._sqlite_convert(task)
            elif source == "sql" and target in ["json", "csv", "markdown"]:
                return self._sql_convert(task)
            
            return ConversionResult(success=False, error=f"Unsupported: {source} → {target}")
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _spreadsheet_convert(self, task: ConversionTask) -> ConversionResult:
        try:
            import openpyxl
            
            wb = openpyxl.load_workbook(task.source_path, data_only=True)
            sheet = wb.active
            
            rows = []
            headers = []
            
            for i, row in enumerate(sheet.iter_rows(values_only=True)):
                if i == 0:
                    headers = [str(cell) if cell is not None else f"col_{j}" for j, cell in enumerate(row)]
                else:
                    rows.append(dict(zip(headers, row)))
            
            if task.target_format == "csv":
                import csv
                with open(task.target_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
            
            elif task.target_format == "json":
                import json
                task.target_path.write_text(json.dumps(rows, indent=2, default=str))
            
            elif task.target_format == "html":
                html = ['<table border="1">', '<thead><tr>']
                html.extend([f'<th>{h}</th>' for h in headers])
                html.extend(['</tr></thead>', '<tbody>'])
                for row in rows:
                    html.append('<tr>')
                    html.extend([f'<td>{v}</td>' for v in row.values()])
                    html.append('</tr>')
                html.extend(['</tbody></table>'])
                task.target_path.write_text(''.join(html))
            
            return ConversionResult(success=True, output_path=str(task.target_path))
        
        except ImportError:
            return ConversionResult(
                success=False,
                error="openpyxl required. Install: pip install universal-converter[database]"
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _sqlite_convert(self, task: ConversionTask) -> ConversionResult:
        try:
            import sqlite3
            
            conn = sqlite3.connect(task.source_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if task.target_format == "sql":
                sql_parts = []
                for table in tables:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 0")
                    columns = [desc[0] for desc in cursor.description]
                    
                    sql_parts.append(f"CREATE TABLE {table} ({', '.join(columns)});")
                    sql_parts.append("")
                    
                    cursor.execute(f"SELECT * FROM {table}")
                    for row in cursor.fetchall():
                        values = [f"'{v}'" if v is not None else "NULL" for v in row]
                        sql_parts.append(f"INSERT INTO {table} VALUES ({', '.join(values)});")
                    sql_parts.append("")
                
                task.target_path.write_text('\n'.join(sql_parts))
            
            elif task.target_format == "json":
                import json
                data = {}
                for table in tables:
                    cursor.execute(f"SELECT * FROM {table}")
                    rows = [dict(row) for row in cursor.fetchall()]
                    data[table] = rows
                task.target_path.write_text(json.dumps(data, indent=2, default=str))
            
            elif task.target_format == "csv":
                import csv
                for table in tables:
                    cursor.execute(f"SELECT * FROM {table}")
                    rows = cursor.fetchall()
                    if not rows:
                        continue
                    
                    with open(task.target_path.parent / f"{table}.csv", 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([desc[0] for desc in cursor.description])
                        writer.writerows(rows)
                
                return ConversionResult(
                    success=True, 
                    output_path=str(task.target_path.parent / f"{tables[0]}.csv") if tables else str(task.target_path)
                )
            
            conn.close()
            return ConversionResult(success=True, output_path=str(task.target_path))
        
        except ImportError:
            return ConversionResult(
                success=False,
                error="sqlite3 (built-in) required"
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _sql_convert(self, task: ConversionTask) -> ConversionResult:
        import re
        
        content = task.source_path.read_text()
        
        create_table_pattern = r'CREATE TABLE (\w+) \((.+?)\);'
        insert_pattern = r'INSERT INTO (\w+) VALUES \((.+?)\);'
        
        tables = {}
        
        for match in re.finditer(create_table_pattern, content, re.DOTALL):
            table_name = match.group(1)
            columns = [c.strip().split()[0] for c in match.group(2).split(',')]
            tables[table_name] = {"columns": columns, "rows": []}
        
        for match in re.finditer(insert_pattern, content):
            table_name = match.group(1)
            values = [v.strip().strip("'") for v in match.group(2).split(',')]
            if table_name in tables:
                tables[table_name]["rows"].append(values)
        
        if task.target_format == "json":
            import json
            data = {table: {"columns": info["columns"], "rows": info["rows"]} 
                   for table, info in tables.items()}
            task.target_path.write_text(json.dumps(data, indent=2))
        
        elif task.target_format == "csv":
            import csv
            first_table = list(tables.keys())[0]
            if first_table in tables and tables[first_table]["rows"]:
                with open(task.target_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(tables[first_table]["columns"])
                    writer.writerows(tables[first_table]["rows"])
        
        elif task.target_format == "markdown":
            md_lines = []
            for table_name, info in tables.items():
                md_lines.append(f"## {table_name}")
                md_lines.append("")
                if info["rows"]:
                    md_lines.append("| " + " | ".join(info["columns"]) + " |")
                    md_lines.append("|" + "|".join(["---"] * len(info["columns"])) + "|")
                    for row in info["rows"]:
                        md_lines.append("| " + " | ".join(row) + " |")
                md_lines.append("")
            task.target_path.write_text('\n'.join(md_lines))
        
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def convert_format(self, path: str, to_format: str, output: Optional[str] = None) -> ConversionResult:
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format
        )
        return self.convert(task)


class SQLConverter(DatabaseConverter):
    """Alias for SQL-specific conversions"""
    pass
