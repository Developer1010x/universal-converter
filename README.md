# Universal Converter

A file-conversion library whose converters are a **graph**, not a lookup table.
198 format pairs are implemented directly; another 234 are reached by chaining
converters automatically, so `csv → pdf` works even though nothing implements
it, the registry routes it through `html`.

```console
$ universal-convert cities.csv -t pdf
converted cities.pdf  via csv -> html -> pdf
```

Every number in this README is printed by a command you can run:
`universal-convert --list` for the pairs, `universal-convert --doctor` for what
works in *your* environment.

---

## Install

```bash
pip install universal-converter          # core: 65 pairs, zero dependencies
pip install "universal-converter[all]"   # 126 pairs, everything except torch
                                         # + 72 more once ffmpeg is on PATH
```

The core has **no dependencies**. Extras are per-capability, and each one is
imported by code in `src/`, nothing is declared that is not used:

| extra | packages | unlocks |
|---|---|---|
| `data` | PyYAML | YAML reading and writing |
| `toml` | tomli-w (+ tomli on <3.11) | TOML output |
| `images` | Pillow | PNG/JPG/GIF/BMP/TIFF/WebP, resize, thumbnails |
| `docx` | python-docx | Word documents |
| `pdf` | pypdf, reportlab | PDF reading and writing |
| `xlsx` | openpyxl | Excel workbooks |
| `torch` | torch, onnx | PyTorch → ONNX export (large; not in `all`) |

**Audio and video need the `ffmpeg` binary on `PATH`**, a system package, not a
pip one. `apt install ffmpeg`, `brew install ffmpeg`, or `winget install ffmpeg`.
`--doctor` tells you if it is missing.

## Command line

Both `universal-convert` and `python -m universal_converter` work.

```bash
# Target format from -t, or inferred from the output path
universal-convert data.json -t csv
universal-convert data.json -o report.html

# Multi-hop happens automatically and is reported
universal-convert database.sqlite -t md
# converted database.md  via sqlite -> csv -> md

# Plan a route without running it
universal-convert --route csv pdf
# csv -> html -> pdf  (2 hops)
#   1.      csv -> html     DataConverter        ok
#   2.     html -> pdf      DocumentConverter    ok

# Require a single converter
universal-convert data.json -t pdf --direct

# What works here, right now
universal-convert --doctor
universal-convert --list
```

### Try it on the bundled samples

`examples/` holds one small real input per converter family:

```bash
universal-convert examples/people.json   -o /tmp/people.html   # records -> styled table
universal-convert examples/people.json   -o /tmp/people.md     # records -> markdown table
universal-convert examples/cities.csv    -o /tmp/cities.pdf    # routed via html
universal-convert examples/notes.md      -o /tmp/notes.docx
universal-convert examples/places.geojson -o /tmp/places.kml
universal-convert examples/sequences.fasta -o /tmp/seqs.fastq
```

`--doctor` walks every registered pair, checks whether its Python imports
resolve and its external binaries are on `PATH`, and prints how many pairs each
missing package would unlock:

```
  converter                prio  pairs  ready  status
  ImageConverter             10     30     30  all dependencies present
  DocumentConverter          20     12      8  missing reportlab (x4)
  ...
  184/198 conversion pairs are runnable in this environment

  install to unlock:
    reportlab                  4 pairs   pip install universal-converter[pdf]
```

## Python API

```python
from universal_converter import convert_file, can_convert, plan_route

convert_file('data.json', 'report.html')          # direct
convert_file('notes.md', 'notes.pdf')             # direct
convert_file('cities.csv', 'cities.pdf')          # routed through html

can_convert('json', 'csv')                        # -> True  (direct only)
plan_route('csv', 'pdf')                          # -> [csv->html, html->pdf]
```

Concrete converters are importable and lazy, importing one does not import the
others, or their dependencies:

```python
from universal_converter.converters import DataConverter, ImageConverter

ImageConverter().convert_format('photo.png', 'webp')
ImageConverter().resize('photo.png', width=200)    # -> photo_resized.png
```

`resize`/`thumbnail` write to `<name>_resized.<ext>` / `<name>_thumb.<ext>`.
They never default to the input path.

### The registry

```python
from universal_converter import get_registry

registry = get_registry()
registry.find_converter('tiff', 'png')        # -> <class 'ImageConverter'>
registry.targets_for('json')                  # one hop
registry.reachable_from('json', max_hops=2)   # {'csv': 1..., 'pdf': 2}
registry.find_route('csv', 'pdf')             # [RouteStep(...), RouteStep(...)]
```

Discovery walks `converters/`, collects every concrete `BaseConverter`, and
orders them by `PRIORITY`, **lower wins**, so `ImageConverter` (10) beats the
generic `DataConverter` (50) for a pair both declare. Importing the registry
pulls in no optional dependency: converter modules keep heavy imports inside
methods.

## How routing works

`supported_conversions()` is a `{source: [targets]}` adjacency map. `find_route`
runs breadth-first search over it, so the first path found has the fewest hops.
Intermediate files are written to a temporary directory and cleaned up; only the
final step touches the destination.

```
csv ──DataConverter──▶ html ──DocumentConverter──▶ pdf
```

Routing prefers paths whose converters have their dependencies satisfied, so it
will not propose a chain that is going to die on a missing import.

## Supported conversions

Run `universal-convert --list` for the authoritative map. Summary:

| Converter | Pairs | Formats | Needs |
|---|---:|---|---|
| `ImageConverter` | 30 | png, jpg, gif, bmp, tiff, webp | Pillow |
| `VideoConverter` | 42 | mp4, avi, mkv, mov, webm, flv, wmv | ffmpeg binary |
| `AudioConverter` | 30 | mp3, wav, flac, ogg, aac, m4a | ffmpeg binary |
| `DataConverter` | 29 | json, csv, tsv, xml, yaml, txt, html, md | PyYAML for yaml |
| `CloudConverter` | 14 | tf, tfvars, json, yaml, toml, env | PyYAML, tomli-w |
| `DocumentConverter` | 12 | pdf, docx, txt, md, html | pypdf, reportlab, python-docx |
| `DatabaseConverter` | 12 | sqlite, db, sql, xlsx | openpyxl for xlsx |
| `BioinformaticsConverter` | 11 | fasta, fastq, genbank, bed, vcf |, |
| `GISConverter` | 9 | geojson, kml, gpx, csv |, |
| `NetworkConverter` | 7 | html, md, txt, url |, |
| `AIConverter` | 2 | pt/pth → onnx | torch, onnx |

### What this does not do

Fidelity is text-level, and the README would rather say so than let you find
out:

- **Documents** carry paragraphs and heading levels across `docx`/`md`/`html`/
  `pdf`/`txt`. Styling, images, tables and layout do not survive. For real
  layout fidelity use LibreOffice or pandoc.
- **Terraform** parsing is regex-level: top-level `resource` blocks and
  `.tfvars` assignments, not the HCL grammar.
- **Shapefiles** are not supported. They need GDAL/fiona, which pip cannot
  reliably install; GeoJSON/KML/GPX are handled with the standard library.
- **`pt → onnx`** refuses to unpickle a checkpoint unless you pass
  `options={"trust_pickle": True}`. `torch.load(weights_only=False)` executes
  code from the file. TorchScript archives load without the opt-in.
- Removed in 1.1.0 because they were advertised without an implementation:
  bam/cram/sam/bcf/bigbed, h5/tflite model formats, shapefiles, and the
  `PDBConverter` stub.

## Design

1. **The registry decides, not the caller.** `convert_file` and the CLI resolve
   the pair through the registry; no converter class is hardcoded anywhere.
2. **Lazy everything.** Optional imports live inside methods, so the core works
   without them and `--list` never silently shrinks because a module failed to
   import, `--doctor` reports the failure instead.
3. **Advertise only what runs.** A test walks every pair in the registry and
   fails if any converter answers "unsupported".
4. **Actionable errors.** Every dependency message names an extra that exists.

## Development

```bash
git clone https://github.com/Developer1010x/universal-converter
cd universal-converter
pip install -e ".[all,dev]"
pytest                       # 78 tests
```

The suite is round-trip based: `json → csv → json`, `fasta → fastq → fasta`,
`geojson → kml → geojson`, `md → html → md`. Generated SQL is verified by
replaying it into an in-memory SQLite. Media tests skip themselves when `ffmpeg`
is absent.

## Requirements

Python 3.9+. Core has no dependencies.

## License

MIT

## Repository

https://github.com/Developer1010x/universal-converter
