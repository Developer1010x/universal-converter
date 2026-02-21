# Universal Converter

Convert anything to anything - a comprehensive, OS-neutral Python conversion library.

[![PyPI](https://img.shields.io/pypi/v/universal-converter)](https://pypi.org/project/universal-converter/)
[![Python](https://img.shields.io/pypi/pyversions/universal-converter)](https://pypi.org/project/universal-converter/)
[![License](https://img.shields.io/pypi/l/universal-converter)](LICENSE)

## Features

### Data Formats
- JSON, CSV, TSV, XML, YAML, INI, TOML

### Documents
- PDF, DOCX, TXT, HTML, Markdown

### Databases
- SQLite → JSON, CSV, MySQL, PostgreSQL

### Images
- PNG, JPG, JPEG, GIF, BMP, TIFF, WebP, ICO
- Resize, convert between formats

### Archives
- ZIP, TAR, GZ, BZ2

### Encoding
- Base64 encode/decode
- Hex encode/decode
- URL encode/decode
- HTML encode/decode

### Color Formats
- HEX ↔ RGB ↔ HSL

### Date/Time
- Unix timestamp ↔ ISO 8601 ↔ Human readable

### Case Conversions
- camelCase, snake_case, kebab-case, PascalCase, SCREAMING_SNAKE_CASE

### Units
- Length: m, km, cm, mm, mi, ft, in
- Weight: kg, g, mg, lb, oz
- Temperature: °C, °F, K
- Volume: L, mL, gal, qt, pt
- Data: B, KB, MB, GB, TB

### Hash Functions
- MD5, SHA1, SHA256, SHA512

### Serialization
- JSON, Pickle, MessagePack, TOML

### Utilities
- File info & type detection
- Magic byte detection
- MIME type detection

## Installation

```bash
# Core (no dependencies)
pip install universal-converter

# Full features
pip install universal-converter[full]
```

### Optional Dependencies
- `pillow` - Image conversion & resizing
- `PyPDF2` - PDF to text
- `reportlab` - Text to PDF
- `python-docx` - Word documents
- `pandas` - CSV/database enhancements
- `msgpack` - MessagePack serialization

## Usage

### Python API

```python
from universal_converter import (
    convert, convert_file, resize_image,
    encode_base64, decode_hex, convert_color,
    convert_case, convert_unit, timestamp_to_iso,
    hash_string, detect_file_type, file_info
)

# File conversion
convert_file('input.json', 'output.xml')

# Image resize
resize_image('photo.png', 'small.png', width=800)

# Encoding
encode_base64("Hello World")  # SGVsbG8gV29ybGQ=
decode_hex("48656c6c6f")  # Hello

# Color conversion
convert_color("#FF5733", "rgb")  # rgb(255, 87, 51)
convert_color("rgb(255, 87, 51)", "hsl")  # hsl(11, 100%, 60%)

# Case conversion
convert_case("hello_world", "camel")  # helloWorld
convert_case("HelloWorld", "snake")  # hello_world
convert_case("hello_world", "kebab")  # hello-world

# Unit conversion
convert_unit(100, "km", "mi")  # 62.137
convert_unit(32, "celsius", "fahrenheit")  # 89.6

# Date conversion
timestamp_to_iso(1700000000)  # 2023-11-14T21:33:20
human_to_timestamp("2024-01-01", "%Y-%m-%d")

# Hash
hash_string("password", "sha256")
hash_file("file.txt", "md5")

# File info
file_info("document.pdf")
detect_file_type("image.jpg")
```

### Command Line

```bash
# Convert file
universal-convert input.json -t csv -o output.csv

# Image resize
universal-convert image.png --resize --width 800

# SQLite to MySQL
universal-convert database.sqlite -t mysql -o output.sql

# Batch convert
universal-convert --batch file1.json file2.xml --output-dir out/

# File info
universal-convert document.pdf -i

# Hash
universal-convert file.txt --hash sha256

# List formats
universal-convert -l
```

## Supported Formats

| Category | Formats |
|----------|---------|
| Text | txt, md, html, rst |
| Data | json, csv, tsv, xml, yaml, ini, toml |
| Document | pdf, docx, doc, odt, rtf |
| Database | sqlite, db, mysql, postgresql |
| Image | png, jpg, jpeg, gif, bmp, tiff, webp, ico |
| Archive | zip, tar, gz, bz2 |
| Code | py, js, java, c, cpp, go, rs, rb, etc. |

## Requirements

- Python 3.8+
- Standard library (core features)

## License

MIT

## Author

Prajwall Narayana

## Repository

https://github.com/Developer1010x/universal-converter
