# Universal Converter

Convert anything to anything - a comprehensive, dependency-free (core) Python conversion library.

## Features

- **Data Formats**: JSON, CSV, TSV, XML, YAML, INI, TOML
- **Documents**: PDF, DOCX, TXT, HTML, Markdown
- **Images**: PNG, JPG, JPEG, GIF, BMP, TIFF, WebP, ICO (with Pillow)
- **Archives**: ZIP, TAR, GZ, BZ2
- **Code**: 15+ programming languages
- **Utilities**: File info, validation, hashing

## Installation

```bash
# Core (no dependencies)
pip install universal-converter

# Full features
pip install universal-converter[full]
```

## Usage

### Python API

```python
from universal_converter import convert, convert_file, resize_image, get_formats

# Convert data
result = convert('{"name": "John"}', to_fmt='csv')
print(result)

# Convert file
convert_file('input.json', 'output.xml')

# Resize image
resize_image('photo.png', 'small.png', width=800)

# Get supported formats
print(get_formats())
```

### Command Line

```bash
# Convert file
universal-convert input.json -t csv -o output.csv

# Image resize
universal-convert image.png --resize --width 800

# Batch convert
universal-convert --batch file1.json file2.xml --output-dir out/

# List formats
universal-convert -l
```

## Requirements

- Python 3.8+

### Optional Dependencies

- `pillow` - Image conversion & resizing
- `PyPDF2` - PDF to text
- `reportlab` - Text to PDF
- `python-docx` - Word document handling

## License

MIT
