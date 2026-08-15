"""Shared fixtures: small, hand-written samples of every supported input."""

import json
from pathlib import Path

import pytest

RECORDS = [
    # The comma, the embedded quotes and the nested list are the exact shapes
    # that the old naive CSV writer corrupted.
    {"name": "Ada, Lovelace", "lang": "analytical engine", "tags": ["math"], "year": 1843},
    {
        "name": 'Grace "Amazing" Hopper',
        "lang": "COBOL",
        "tags": ["compiler"],
        "year": 1959,
        "note": "key only on the second record",
    },
]


@pytest.fixture
def records():
    return [dict(row) for row in RECORDS]


@pytest.fixture
def json_file(tmp_path: Path, records) -> Path:
    path = tmp_path / "people.json"
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "table.csv"
    path.write_text(
        'name,role\n"Hopper, Grace",compiler\n"She said ""go""",note\n', encoding="utf-8"
    )
    return path


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    path = tmp_path / "notes.md"
    path.write_text(
        "# Title\n\nSome **bold** and `code` text.\n\n- one\n- two\n\n"
        "[link](https://example.com)\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def html_file(tmp_path: Path) -> Path:
    path = tmp_path / "page.html"
    path.write_text(
        "<!DOCTYPE html><html><head><style>p{color:red}</style></head><body>"
        "<h1>Heading</h1><p>Hello <strong>world</strong> &amp; friends.</p>"
        "<script>alert(1)</script>"
        "<table><thead><tr><th>a</th><th>b</th></tr></thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
        "</body></html>",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fasta_file(tmp_path: Path) -> Path:
    path = tmp_path / "seq.fasta"
    path.write_text(
        ">seq1 example sequence\nACGTACGTAC\nGTACGTACGT\n>seq2 second\nTTTTGGGGCC\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def vcf_file(tmp_path: Path) -> Path:
    path = tmp_path / "variants.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t100\trs1\tA\tG\t50.0\tPASS\tDP=10;AF=0.5\n"
        "chr2\t250\t.\tAT\tA\t.\tPASS\tDP=8\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def geojson_file(tmp_path: Path) -> Path:
    path = tmp_path / "places.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Bengaluru"},
                        "geometry": {"type": "Point", "coordinates": [77.5946, 12.9716]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Route"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[77.5, 12.9], [77.6, 13.0]],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sqlite_file(tmp_path: Path) -> Path:
    import sqlite3

    path = tmp_path / "shop.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL)"
    )
    connection.executemany(
        "INSERT INTO items (id, name, price) VALUES (?, ?, ?)",
        [(1, "widget", 9.99), (2, "O'Brien's bolt", 4.5), (3, None, None)],
    )
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "settings.env"
    path.write_text(
        '# comment\nAPP_NAME="demo"\nPORT=8080\nDEBUG=true\n', encoding="utf-8"
    )
    return path


@pytest.fixture
def png_file(tmp_path: Path) -> Path:
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "photo.png"
    Image.new("RGBA", (400, 300), (30, 120, 200, 255)).save(path)
    return path
