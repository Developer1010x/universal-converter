# Universal Converter

A comprehensive Python library for converting between file formats, data types, and domain-specific formats.

## Overview

Universal Converter provides a unified interface for hundreds of conversion operations across multiple domains:

- **Data formats**: JSON, CSV, XML, YAML, TOML, Parquet
- **Documents**: PDF, DOCX, Markdown, HTML, RTF
- **Media**: Images (PNG, JPG, WebP, TIFF), Audio (MP3, WAV, FLAC), Video (MP4, AVI, MKV)
- **Scientific**: Bioinformatics (FASTA, FASTQ, GenBank, VCF), GIS (GeoJSON, Shapefile, KML)
- **ML/AI**: PyTorch, ONNX, TensorFlow, Keras model formats
- **Infrastructure**: Terraform, Kubernetes manifests, cloud configs

## Installation

```bash
pip install universal-converter
```

Install with specific extras for additional features:

```bash
pip install universal-converter[images]    # Image processing
pip install universal-converter[audio]     # Audio conversion
pip install universal-converter[video]     # Video conversion
pip install universal-converter[ai]         # ML model conversion
pip install universal-converter[database]  # Database formats
pip install universal-converter[all]       # All features
```

## Architecture

The library follows a modular, plugin-based architecture:

```
universal_converter/
├── converters/          # Format-specific converters
│   ├── data.py         # JSON, CSV, XML, YAML
│   ├── images.py       # Image formats
│   ├── audio.py        # Audio formats
│   ├── video.py        # Video formats
│   ├── documents.py    # Document formats
│   ├── ai.py          # ML model formats
│   ├── cloud.py        # Cloud configs
│   ├── bioinformatics.py
│   ├── gis.py
│   ├── network.py
│   └── database.py
└── utils/
    └── platform.py     # System utilities
```

All converters inherit from `BaseConverter` and implement a consistent interface.

## Usage

### Python API

```python
from universal_converter import convert_file, resize_image

# Convert between data formats
convert_file('data.json', 'output.xml')

# Resize an image
resize_image('photo.png', 'thumb.png', width=200)

# Access specific converters directly
from universal_converter.converters import DataConverter, ImageConverter

converter = DataConverter()
result = converter.convert(task)
```

### Command Line

```bash
# Convert a file
python -m universal_converter input.json -t csv -o output.csv

# List supported formats
python -m universal_converter -l
```

## Supported Formats

| Category | Formats |
|----------|---------|
| Data | json, csv, xml, yaml, toml, parquet |
| Images | png, jpg, gif, bmp, tiff, webp |
| Audio | mp3, wav, flac, ogg, aac, m4a |
| Video | mp4, avi, mkv, mov, webm |
| Documents | pdf, docx, txt, html, md |
| ML Models | pt, h5, onnx, tflite, pkl |
| Bioinformatics | fasta, fastq, genbank, vcf, bed |
| GIS | shp, geojson, kml, gpx, tif |
| Database | sqlite, sql, xlsx |

## Design Principles

1. **Lazy loading**: Converters are loaded on-demand to minimize startup time
2. **Optional dependencies**: Heavy dependencies are optional; the core works without them
3. **Clear error messages**: Missing dependencies return actionable installation instructions
4. **Type hints**: Full type annotations for IDE support

## Requirements

- Python 3.9+
- Core dependencies installed automatically

## License

MIT

## Repository

https://github.com/Developer1010x/universal-converter
