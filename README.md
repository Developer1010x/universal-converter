# Universal Converter

Convert anything to anything - the most comprehensive Python conversion library.

[![PyPI](https://img.shields.io/pypi/v/universal-converter)](https://pypi.org/project/universal-converter/)
[![Python](https://img.shields.io/pypi/pyversions/universal-converter)](https://pypi.org/project/universal-converter/)
[![License](https://img.shields.io/pypi/l/universal-converter)](LICENSE)

## Features (100+ Conversions)

### 📄 Data Formats
- JSON, CSV, TSV, XML, YAML, INI, TOML, Parquet

### 📝 Documents
- PDF, DOCX, TXT, HTML, Markdown, RTF, ODT

### 📊 Spreadsheets
- CSV ↔ Excel (XLSX, XLS)
- Excel ↔ JSON
- CSV ↔ Parquet

### 🗄️ Databases
- SQLite → JSON, CSV, MySQL, PostgreSQL
- MySQL/PostgreSQL dump parsing

### 🖼️ Images
- PNG, JPG, JPEG, GIF, BMP, TIFF, WebP, ICO, SVG
- Resize, convert between formats
- OCR (Image to Text)

### 📦 Archives
- ZIP, TAR, GZ, BZ2, RAR
- Encrypted ZIP

### 📽️ Presentations
- Text → PPTX
- PPTX → PDF, TXT

### 🎵 Audio
- MP3 ↔ WAV
- Audio → Text (Speech-to-Text)
- Text → Speech (TTS)

### 🎥 Video
- MP4 ↔ AVI
- Video → GIF
- Video → Audio

### 🌐 Web & API
- HTML ↔ Text, Markdown
- XML ↔ Dict
- URL → File Download

### 🔬 Scientific Data
- NumPy arrays
- HDF5 files
- MATLAB (.mat) files
- FASTA/GenBank (Bioinformatics)

### 🔐 Encryption & Encoding
- QR Code generation & reading
- Password hashing (bcrypt, argon2)
- JWT encode/decode

### 📚 eBooks
- EPUB ↔ Text

### 💱 Advanced Units
- Unit conversion (pint)
- Currency conversion

### 🤖 AI / NLP
- Text → Embeddings
- Text → Tokens
- Audio → Transcription (Whisper)
- Text → Speech (TTS)

### ☁️ Cloud Storage
- AWS S3 upload/download
- Google Cloud Storage upload/download
- Azure Blob upload/download

### ⛓️ Blockchain
- Ethereum key pair generation
- Private key → Address
- Keccak-256 hash

### 🧬 Bioinformatics
- FASTA ↔ Dict
- GenBank → FASTA

### 🐳 Containers
- Dockerfile → Image
- Image ↔ TAR

### 📡 Streaming
- Kafka produce/consume
- Redis set/get
- MQTT publish

### 🎨 CAD/Graphics
- DXF → SVG

### 📜 Data Exchange
- EDI → JSON
- HL7 → Dict

## Installation

```bash
# Install with all dependencies (recommended)
pip install universal-converter
```

All 40+ dependencies are installed by default!

## Usage

### Python API

```python
from universal_converter import (
    # Basic conversions
    convert, convert_file, resize_image,
    encode_base64, decode_hex, convert_color,
    convert_case, convert_unit, timestamp_to_iso,
    
    # Spreadsheet
    csv_to_excel, excel_to_csv, csv_to_parquet,
    
    # Media
    text_to_qr, qr_to_text,
    image_to_text, html_to_text,
    epub_to_text, text_to_epub,
    mp3_to_wav, video_to_gif,
    
    # AI/Cloud
    text_to_embeddings, audio_to_transcription,
    upload_to_s3, download_from_gcs,
    
    # Blockchain
    generate_key_pair, private_key_to_address,
    
    # Bioinformatics
    fasta_to_dict, genbank_to_fasta,
    
    # Streaming
    send_to_kafka, set_redis, publish_mqtt,
)

# File conversion
convert_file('input.json', 'output.xml')

# Image resize
resize_image('photo.png', 'small.png', width=800)

# QR Code
text_to_qr("Hello World", "qr.png")

# AI Embeddings
embeddings = text_to_embeddings("Hello world")

# Cloud Storage
upload_to_s3('file.txt', 'my-bucket')

# Blockchain
keys = generate_key_pair()

# Bioinformatics
sequences = fasta_to_dict('dna.fasta')
```

### Command Line

```bash
# Convert file
universal-convert input.json -t csv -o output.csv

# Image resize
universal-convert image.png --resize --width 800

# File hash
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
| Audio | mp3, wav, ogg, flac |
| Video | mp4, avi, mov, gif |
| eBook | epub, mobi, azw |
| Scientific | fasta, genbank, hdf5, mat, npy |

## Requirements

- Python 3.8+
- All dependencies auto-installed

## License

MIT

## Author

Prajwall Narayana

## Repository

https://github.com/Developer1010x/universal-converter
