# Universal Converter

Convert anything to anything - a comprehensive, OS-neutral Python conversion library.

[![PyPI](https://img.shields.io/pypi/v/universal-converter)](https://pypi.org/project/universal-converter/)
[![Python](https://img.shields.io/pypi/pyversions/universal-converter)](https://pypi.org/project/universal-converter/)
[![License](https://img.shields.io/pypi/l/universal-converter)](LICENSE)

## Features

### Data Formats
- JSON, CSV, TSV, XML, YAML, INI, TOML

### Documents
- PDF, DOCX, TXT, HTML, Markdown, RTF, ODT

### Spreadsheets
- CSV ↔ Excel (XLSX, XLS)
- Excel ↔ JSON
- CSV ↔ Parquet

### Databases
- SQLite → JSON, CSV, MySQL, PostgreSQL
- MySQL/PostgreSQL dump parsing

### Images
- PNG, JPG, JPEG, GIF, BMP, TIFF, WebP, ICO, SVG
- Resize, convert between formats
- OCR (Image to Text)

### Archives
- ZIP, TAR, GZ, BZ2, RAR
- Encrypted ZIP

### Presentations
- Text → PPTX
- PPTX → PDF, TXT

### Audio
- MP3 ↔ WAV
- Audio → Text (Speech-to-Text)

### Video
- MP4 ↔ AVI
- Video → GIF
- Video → Audio

### Web & API
- HTML → Text, Markdown
- XML ↔ Dict
- URL → File Download

### Scientific Data
- NumPy arrays
- HDF5 files
- MATLAB (.mat) files
- Parquet files

### Encryption & Encoding
- QR Code generation & reading
- Password hashing (bcrypt, argon2)
- JWT encode/decode

### eBooks
- EPUB ↔ Text

### Advanced Units
- Unit conversion (pint)
- Currency conversion

## Installation

```bash
# Core (no dependencies)
pip install universal-converter

# Full features
pip install universal-converter[all]

# Specific features
pip install universal-converter[spreadsheet]  # Excel/CSV
pip install universal-converter[video]       # Video conversions
pip install universal-converter[audio]       # Audio conversions
pip install universal-converter[web]          # HTML/XML conversions
pip install universal-converter[encryption]   # QR/JWT/hashing
pip install universal-converter[ocr]          # Image to text
```

### Optional Dependencies by Feature

| Feature | Packages |
|---------|----------|
| Images | pillow, cairosvg |
| PDF | PyPDF2, reportlab, pdf2image |
| Word | python-docx |
| Excel | pandas, openpyxl, xlrd |
| Presentations | python-pptx |
| Markdown/HTML | markdown, weasyprint |
| ODT | odfpy |
| Audio | pydub, SpeechRecognition |
| Video | moviepy |
| Archives | rarfile, pyzipper |
| Web | beautifulsoup4, html2text, requests, xmltodict |
| Scientific | numpy, scipy, h5py, pyarrow |
| Encryption | qrcode, pyzbar, bcrypt, argon2, PyJWT |
| eBooks | ebooklib |
| Units | pint, forex-python |
| OCR | pytesseract |

## Usage

### Python API

```python
from universal_converter import (
    convert, convert_file, resize_image,
    encode_base64, decode_hex, convert_color,
    convert_case, convert_unit, timestamp_to_iso,
    hash_string, detect_file_type, file_info,
    csv_to_excel, excel_to_csv,
    text_to_qr, qr_to_text,
    image_to_text, html_to_text,
    epub_to_text, text_to_epub,
    mp3_to_wav, video_to_gif
)

# File conversion
convert_file('input.json', 'output.xml')

# Image resize
resize_image('photo.png', 'small.png', width=800)

# Spreadsheet
csv_to_excel('data.csv', 'data.xlsx')
excel_to_csv('data.xlsx', 'data.csv')

# Color conversion
convert_color("#FF5733", "rgb")  # rgb(255, 87, 51)

# Case conversion
convert_case("hello_world", "camel")  # helloWorld

# Unit conversion
convert_unit(100, "km", "mi")  # 62.137

# QR Code
text_to_qr("Hello World", "qr.png")
qr_to_text("qr.png")  # "Hello World"

# OCR
image_to_text("screenshot.png")  # Extract text from image

# Audio
mp3_to_wav("audio.mp3", "audio.wav")

# Video
video_to_gif("video.mp4", "animation.gif")

# eBook
epub_to_text("book.epub", "book.txt")

# Web
html_to_text("page.html", "page.txt")
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
| Text | txt, md, html, rst, rtf |
| Data | json, csv, tsv, xml, yaml, ini, toml, parquet |
| Document | pdf, docx, doc, odt |
| Database | sqlite, db, mysql, postgresql |
| Image | png, jpg, jpeg, gif, bmp, tiff, webp, ico, svg |
| Archive | zip, tar, gz, bz2, rar |
| Code | py, js, java, c, cpp, go, rs, rb, etc. |
| Audio | mp3, wav, ogg, flac |
| Video | mp4, avi, mov, gif |
| eBook | epub, mobi, azw |

## Requirements

- Python 3.8+
- Standard library (core features)

## License

MIT

## Author

Prajwall Narayana

## Repository

https://github.com/Developer1010x/universal-converter
