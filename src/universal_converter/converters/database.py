"""Database and spreadsheet converters (SQLite, SQL dumps, XLSX)."""

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import BaseConverter, ConversionResult, ConversionTask

_XLSX_HINT = "openpyxl required. Install: pip install universal-converter[xlsx]"


def _quote_identifier(name: str) -> str:
    """Quote an SQL identifier, doubling any embedded quote."""
    return '"' + str(name).replace('"', '""') + '"'


def _quote_literal(value: Any) -> str:
    """Render a Python value as an SQL literal with correct escaping."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, bytes):
        return "X'" + value.hex() + "'"
    return "'" + str(value).replace("'", "''") + "'"


class DatabaseConverter(BaseConverter):
    """Converter for SQLite databases, SQL dumps and XLSX workbooks."""

    SUPPORTED_CONVERSIONS = {
        "sqlite": ["sql", "json", "csv"],
        "db": ["sql", "json", "csv"],
        "sql": ["json", "csv", "md"],
        "xlsx": ["csv", "json", "html"],
    }
    PRIORITY = 25
    REQUIRES_PYTHON = ["openpyxl"]

    @classmethod
    def requirements_for(cls, source_format: str, target_format: str) -> List[str]:
        # Only the spreadsheet path needs openpyxl; SQLite ships with Python.
        return ["openpyxl"] if source_format.lower() == "xlsx" else []

    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()

        try:
            if source == "xlsx" and target in ("csv", "json", "html"):
                return self._spreadsheet_convert(task)
            if source in ("sqlite", "db") and target in ("sql", "json", "csv"):
                return self._sqlite_convert(task)
            if source == "sql" and target in ("json", "csv", "md"):
                return self._sql_convert(task)

            return ConversionResult(
                success=False, error=f"Unsupported: {source} -> {target}"
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    # --- spreadsheets ----------------------------------------------------

    def _spreadsheet_convert(self, task: ConversionTask) -> ConversionResult:
        try:
            import openpyxl
        except ImportError:
            return ConversionResult(success=False, error=_XLSX_HINT)

        workbook = openpyxl.load_workbook(str(task.source_path), data_only=True)
        sheet = workbook.active

        headers: List[str] = []
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(sheet.iter_rows(values_only=True)):
            if index == 0:
                headers = [
                    str(cell) if cell is not None else f"col_{position}"
                    for position, cell in enumerate(row)
                ]
            else:
                rows.append(dict(zip(headers, row)))

        target = task.target_format.lower()
        if target == "csv":
            with open(task.target_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
        elif target == "json":
            Path(task.target_path).write_text(
                json.dumps(rows, indent=2, default=str), encoding="utf-8"
            )
        else:
            Path(task.target_path).write_text(
                self._html_table(headers, [list(row.values()) for row in rows]),
                encoding="utf-8",
            )

        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"sheet": sheet.title, "rows": len(rows)},
        )

    @staticmethod
    def _html_table(headers: List[str], rows: List[List[Any]]) -> str:
        from html import escape

        head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{escape('' if v is None else str(v))}</td>" for v in row)
            + "</tr>"
            for row in rows
        )
        return (
            '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
            "<style>table{border-collapse:collapse}"
            "th,td{border:1px solid #8884;padding:.3rem .5rem;text-align:left}"
            "</style></head><body>\n"
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>\n"
            "</body></html>"
        )

    # --- sqlite ----------------------------------------------------------

    def _sqlite_convert(self, task: ConversionTask) -> ConversionResult:
        target = task.target_format.lower()
        # Read-only URI: converting a database must never be able to write to it.
        uri = f"file:{Path(task.source_path).resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            if target == "sql":
                output = self._dump_sql(cursor, tables)
                Path(task.target_path).write_text(output, encoding="utf-8")
                written = [str(task.target_path)]
            elif target == "json":
                data = {}
                for table in tables:
                    cursor.execute(f"SELECT * FROM {_quote_identifier(table)}")
                    data[table] = [dict(row) for row in cursor.fetchall()]
                Path(task.target_path).write_text(
                    json.dumps(data, indent=2, default=str), encoding="utf-8"
                )
                written = [str(task.target_path)]
            else:
                written = self._dump_csv(cursor, tables, Path(task.target_path))
        finally:
            # Every early return above still has to release the handle.
            connection.close()

        return ConversionResult(
            success=True,
            output_path=written[0] if written else str(task.target_path),
            metadata={"tables": tables, "files": written},
        )

    def _dump_sql(self, cursor, tables: List[str]) -> str:
        parts: List[str] = []
        for table in tables:
            quoted = _quote_identifier(table)
            # sqlite_master keeps the original CREATE statement, types and all;
            # regenerating it from cursor.description would lose every type.
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            row = cursor.fetchone()
            ddl = row[0] if row and row[0] else self._infer_ddl(cursor, table)
            parts.append(f"{ddl.rstrip(';')};")

            cursor.execute(f"SELECT * FROM {quoted}")
            columns = [description[0] for description in cursor.description]
            column_list = ", ".join(_quote_identifier(c) for c in columns)
            for record in cursor.fetchall():
                values = ", ".join(_quote_literal(v) for v in tuple(record))
                parts.append(f"INSERT INTO {quoted} ({column_list}) VALUES ({values});")
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _infer_ddl(cursor, table: str) -> str:
        cursor.execute(f"PRAGMA table_info({_quote_identifier(table)})")
        columns = [
            f"{_quote_identifier(row[1])} {row[2] or 'TEXT'}" for row in cursor.fetchall()
        ]
        return f"CREATE TABLE {_quote_identifier(table)} ({', '.join(columns)})"

    @staticmethod
    def _dump_csv(cursor, tables: List[str], target: Path) -> List[str]:
        written: List[str] = []
        for table in tables:
            cursor.execute(f"SELECT * FROM {_quote_identifier(table)}")
            rows = cursor.fetchall()
            if not rows:
                continue
            # One CSV per table; the first is reported as the primary output.
            path = target if len(tables) == 1 else target.parent / f"{table}.csv"
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([d[0] for d in cursor.description])
                writer.writerows(tuple(row) for row in rows)
            written.append(str(path))
        return written

    # --- sql dumps -------------------------------------------------------

    def _sql_convert(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        tables = self._parse_sql(content)
        target = task.target_format.lower()

        if target == "json":
            Path(task.target_path).write_text(
                json.dumps(tables, indent=2, default=str), encoding="utf-8"
            )
        elif target == "csv":
            if not tables:
                return ConversionResult(
                    success=False, error="no CREATE TABLE statements found"
                )
            name = next(iter(tables))
            with open(task.target_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(tables[name]["columns"])
                writer.writerows(tables[name]["rows"])
        else:
            Path(task.target_path).write_text(
                self._to_markdown(tables), encoding="utf-8"
            )

        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"tables": list(tables)},
        )

    @staticmethod
    def _split_values(values: str) -> List[Any]:
        """Split an SQL VALUES tuple, honouring quotes and doubled quotes."""
        out: List[Any] = []
        current: List[str] = []
        in_string = False
        index = 0
        while index < len(values):
            char = values[index]
            if in_string:
                if char == "'":
                    if index + 1 < len(values) and values[index + 1] == "'":
                        current.append("'")
                        index += 2
                        continue
                    in_string = False
                else:
                    current.append(char)
            elif char == "'":
                in_string = True
            elif char == ",":
                out.append("".join(current).strip())
                current = []
            else:
                current.append(char)
            index += 1
        out.append("".join(current).strip())
        return [None if v == "NULL" else v for v in out]

    def _parse_sql(self, content: str) -> Dict[str, Dict[str, Any]]:
        tables: Dict[str, Dict[str, Any]] = {}

        for match in re.finditer(
            r'CREATE TABLE (?:IF NOT EXISTS )?["`\[]?(\w+)["`\]]?\s*\((.+?)\);',
            content,
            re.DOTALL | re.IGNORECASE,
        ):
            columns = [
                part.strip().split()[0].strip('"`[]')
                for part in re.split(r",(?![^(]*\))", match.group(2))
                if part.strip()
            ]
            tables[match.group(1)] = {"columns": columns, "rows": []}

        for match in re.finditer(
            r'INSERT INTO ["`\[]?(\w+)["`\]]?[^(]*(?:\(([^)]*)\))?\s*VALUES\s*\((.+?)\);',
            content,
            re.DOTALL | re.IGNORECASE,
        ):
            name = match.group(1)
            if name in tables:
                tables[name]["rows"].append(self._split_values(match.group(3)))
        return tables

    @staticmethod
    def _to_markdown(tables: Dict[str, Dict[str, Any]]) -> str:
        lines: List[str] = []
        for name, info in tables.items():
            lines.append(f"## {name}\n")
            columns = info["columns"]
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("|" + "|".join(["---"] * len(columns)) + "|")
            for row in info["rows"]:
                cells = ["" if v is None else str(v).replace("|", r"\|") for v in row]
                cells += [""] * (len(columns) - len(cells))
                lines.append("| " + " | ".join(cells[: len(columns)]) + " |")
            lines.append("")
        return "\n".join(lines)

    def convert_format(
        self, path: str, to_format: str, output: Optional[str] = None
    ) -> ConversionResult:
        """Convert a database or workbook to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)
