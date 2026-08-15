"""Round-trip and golden-property tests, one per converter family."""

import csv
import io
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from universal_converter import convert_file, get_registry
from universal_converter.converters.data import DataConverter
from universal_converter.converters.network import NetworkConverter

# --- data ---------------------------------------------------------------


def test_json_to_csv_quotes_commas_and_quotes(json_file, tmp_path):
    out = Path(convert_file(str(json_file), str(tmp_path / "out.csv")))
    rows = list(csv.DictReader(io.StringIO(out.read_text(encoding="utf-8"))))

    assert len(rows) == 2
    assert rows[0]["name"] == "Ada, Lovelace"
    assert rows[1]["name"] == 'Grace "Amazing" Hopper'
    # A key that appears only on a later record must not be dropped.
    assert rows[1]["note"] == "key only on the second record"
    assert rows[0]["note"] == ""


def test_json_csv_json_round_trip_preserves_field_names(json_file, tmp_path):
    as_csv = convert_file(str(json_file), str(tmp_path / "out.csv"))
    back = json.loads(Path(convert_file(as_csv, str(tmp_path / "back.json"))).read_text())

    original = json.loads(json_file.read_text())
    assert [set(row) for row in back] == [
        set(original[0]) | {"note"},
        set(original[1]),
    ]
    assert back[0]["name"] == original[0]["name"]


def test_nested_values_are_json_not_python_reprs(json_file, tmp_path):
    text = Path(convert_file(str(json_file), str(tmp_path / "out.csv"))).read_text()
    assert "['math']" not in text  # the old Python repr
    assert '""math""' in text  # JSON, CSV-quoted


def test_csv_with_embedded_quotes_survives_to_json(csv_file, tmp_path):
    rows = json.loads(
        Path(convert_file(str(csv_file), str(tmp_path / "out.json"))).read_text()
    )
    assert rows[0]["name"] == "Hopper, Grace"
    assert rows[1]["name"] == 'She said "go"'


def test_json_xml_json_round_trip(json_file, tmp_path):
    as_xml = convert_file(str(json_file), str(tmp_path / "out.xml"))
    back = json.loads(Path(convert_file(as_xml, str(tmp_path / "back.json"))).read_text())
    # XML has no types, so compare stringified values.
    assert back["root"][0]["name"] == "Ada, Lovelace"


def test_json_yaml_json_round_trip_is_lossless(json_file, tmp_path):
    pytest.importorskip("yaml")
    as_yaml = convert_file(str(json_file), str(tmp_path / "out.yaml"))
    back = json.loads(Path(convert_file(as_yaml, str(tmp_path / "back.json"))).read_text())
    assert back == json.loads(json_file.read_text())


def test_json_to_html_is_a_self_contained_table(json_file, tmp_path):
    html = Path(convert_file(str(json_file), str(tmp_path / "out.html"))).read_text()
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html and "http" not in html.split("<style>")[1][:400]
    assert "<th>name</th>" in html
    assert "Ada, Lovelace" in html


def test_html_escapes_dangerous_content(tmp_path):
    source = tmp_path / "evil.json"
    source.write_text(json.dumps([{"x": "<script>alert(1)</script>"}]))
    html = Path(convert_file(str(source), str(tmp_path / "out.html"))).read_text()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_json_to_markdown_table(json_file, tmp_path):
    md = Path(convert_file(str(json_file), str(tmp_path / "out.md"))).read_text()
    lines = md.strip().splitlines()
    assert lines[0].startswith("| name | lang |")
    assert set(lines[1]) <= set("| -")
    assert len(lines) == 4


def test_xml_tag_names_are_sanitised(tmp_path):
    source = tmp_path / "weird.json"
    source.write_text(json.dumps({"has space": 1, "1leading": 2}))
    xml = Path(convert_file(str(source), str(tmp_path / "out.xml"))).read_text()
    assert "has_space" in xml and "_1leading" in xml


def test_empty_list_produces_empty_csv(tmp_path):
    source = tmp_path / "empty.json"
    source.write_text("[]")
    assert Path(convert_file(str(source), str(tmp_path / "out.csv"))).read_text() == ""


def test_tsv_uses_tabs(json_file, tmp_path):
    text = Path(convert_file(str(json_file), str(tmp_path / "out.tsv"))).read_text()
    assert "\t" in text.splitlines()[0]


# --- markup -------------------------------------------------------------


def test_html_to_markdown_keeps_structure_and_drops_scripts(html_file, tmp_path):
    md = Path(convert_file(str(html_file), str(tmp_path / "out.md"))).read_text()
    assert "# Heading" in md
    assert "**world**" in md
    assert "alert(1)" not in md
    assert "color:red" not in md
    assert "| a | b |" in md


def test_html_to_text_unescapes_entities(html_file, tmp_path):
    text = Path(convert_file(str(html_file), str(tmp_path / "out.txt"))).read_text()
    assert "&amp;" not in text and "& friends" in text
    assert "<" not in text


def test_markdown_to_html_renders_the_subset(markdown_file, tmp_path):
    html = Path(convert_file(str(markdown_file), str(tmp_path / "out.html"))).read_text()
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html
    assert "<code>code</code>" in html
    assert "<li>one</li>" in html
    assert '<a href="https://example.com">link</a>' in html


def test_markdown_html_markdown_round_trip(markdown_file, tmp_path):
    html = convert_file(str(markdown_file), str(tmp_path / "round.html"))
    back = Path(convert_file(html, str(tmp_path / "back.md"))).read_text()
    assert "# Title" in back
    assert "**bold**" in back
    assert "- one" in back


def test_text_to_html_escapes(tmp_path):
    source = tmp_path / "raw.txt"
    source.write_text("a < b & c\n\nsecond block\n")
    html = Path(convert_file(str(source), str(tmp_path / "out.html"))).read_text()
    assert "&lt; b &amp; c" in html
    assert html.count("<p>") == 2


def test_url_to_json(tmp_path):
    source = tmp_path / "site.url"
    source.write_text("https://example.com:8443/a/b?x=1&x=2&y=3#frag")
    data = json.loads(Path(convert_file(str(source), str(tmp_path / "out.json"))).read_text())
    assert data["scheme"] == "https"
    assert data["hostname"] == "example.com"
    assert data["port"] == 8443
    assert data["query"] == {"x": ["1", "2"], "y": ["3"]}
    assert data["fragment"] == "frag"


# --- bioinformatics -----------------------------------------------------


def test_fasta_fastq_fasta_round_trip(fasta_file, tmp_path):
    fastq = convert_file(str(fasta_file), str(tmp_path / "out.fastq"))
    back = Path(convert_file(fastq, str(tmp_path / "back.fasta"))).read_text()

    def sequences(text):
        return "".join(l for l in text.splitlines() if not l.startswith(">"))

    assert sequences(back) == sequences(fasta_file.read_text())
    assert back.count(">") == 2


def test_fasta_handles_wrapped_sequences(fasta_file, tmp_path):
    records = json.loads(
        Path(convert_file(str(fasta_file), str(tmp_path / "out.json"))).read_text()
    )
    assert records[0]["id"] == "seq1"
    assert records[0]["length"] == 20  # two 10-base lines joined
    assert len(records) == 2


def test_fasta_genbank_fasta_round_trip(fasta_file, tmp_path):
    genbank = convert_file(str(fasta_file), str(tmp_path / "out.genbank"))
    assert "ORIGIN" in Path(genbank).read_text()
    back = Path(convert_file(genbank, str(tmp_path / "back.fasta"))).read_text()
    assert "ACGTACGTACGTACGTACGT" in back.replace("\n", "")


def test_vcf_to_json_and_bed(vcf_file, tmp_path):
    variants = json.loads(
        Path(convert_file(str(vcf_file), str(tmp_path / "out.json"))).read_text()
    )
    assert variants[0]["pos"] == 100
    assert variants[0]["info"] == {"DP": "10", "AF": "0.5"}
    assert variants[1]["qual"] is None

    bed = Path(convert_file(str(vcf_file), str(tmp_path / "out.bed"))).read_text()
    # BED is 0-based half-open.
    assert bed.splitlines()[0].split("\t")[:3] == ["chr1", "99", "100"]


def test_bed_to_vcf_emits_valid_alt(tmp_path):
    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t10\t20\tregionA\n")
    vcf = Path(convert_file(str(bed), str(tmp_path / "out.vcf"))).read_text()
    body = [l for l in vcf.splitlines() if not l.startswith("#")][0].split("\t")
    assert body[1] == "11"  # 0-based -> 1-based
    assert body[4] == "<NON_REF>"
    assert "\\." not in vcf  # the old code emitted the literal <\.\>


# --- gis ----------------------------------------------------------------


def test_geojson_kml_geojson_round_trip(geojson_file, tmp_path):
    kml = convert_file(str(geojson_file), str(tmp_path / "out.kml"))
    assert "<Placemark>" in Path(kml).read_text()

    back = json.loads(Path(convert_file(kml, str(tmp_path / "back.geojson"))).read_text())
    original = json.loads(geojson_file.read_text())
    assert len(back["features"]) == len(original["features"])
    assert back["features"][0]["geometry"] == original["features"][0]["geometry"]
    assert back["features"][0]["properties"]["name"] == "Bengaluru"


def test_geojson_gpx_geojson_round_trip(geojson_file, tmp_path):
    gpx = convert_file(str(geojson_file), str(tmp_path / "out.gpx"))
    back = json.loads(Path(convert_file(gpx, str(tmp_path / "back.geojson"))).read_text())
    types = {f["geometry"]["type"] for f in back["features"]}
    assert types == {"Point", "LineString"}


def test_geojson_to_csv_flattens_coordinates(geojson_file, tmp_path):
    rows = list(
        csv.DictReader(
            io.StringIO(
                Path(convert_file(str(geojson_file), str(tmp_path / "out.csv"))).read_text()
            )
        )
    )
    assert rows[0] == {
        "name": "Bengaluru",
        "type": "Point",
        "longitude": "77.5946",
        "latitude": "12.9716",
    }


# --- database -----------------------------------------------------------


def test_sqlite_to_json(sqlite_file, tmp_path):
    data = json.loads(
        Path(convert_file(str(sqlite_file), str(tmp_path / "out.json"))).read_text()
    )
    assert [row["name"] for row in data["items"]] == ["widget", "O'Brien's bolt", None]


def test_sqlite_to_sql_is_executable_and_escapes_quotes(sqlite_file, tmp_path):
    dump = Path(convert_file(str(sqlite_file), str(tmp_path / "out.sql"))).read_text()
    assert "O''Brien" in dump  # escaped, not injected
    assert "TEXT" in dump and "REAL" in dump  # real types, not bare column names

    # The only honest assertion for generated DDL: feed it back to SQLite.
    replay = sqlite3.connect(":memory:")
    replay.executescript(dump)
    assert replay.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 3
    assert replay.execute("SELECT price FROM items WHERE id=1").fetchone()[0] == 9.99
    replay.close()


def test_sqlite_to_sql_to_json(sqlite_file, tmp_path):
    dump = convert_file(str(sqlite_file), str(tmp_path / "out.sql"))
    data = json.loads(Path(convert_file(dump, str(tmp_path / "out.json"))).read_text())
    assert data["items"]["columns"] == ["id", "name", "price"]
    assert data["items"]["rows"][1][1] == "O'Brien's bolt"


def test_sqlite_source_is_not_modified(sqlite_file, tmp_path):
    before = sqlite_file.read_bytes()
    convert_file(str(sqlite_file), str(tmp_path / "out.json"))
    assert sqlite_file.read_bytes() == before


def test_sql_to_markdown_escapes_pipes(tmp_path):
    dump = tmp_path / "dump.sql"
    dump.write_text(
        'CREATE TABLE t (a TEXT, b TEXT);\n'
        "INSERT INTO t VALUES ('pipe|inside', 'plain');\n"
    )
    md = Path(convert_file(str(dump), str(tmp_path / "out.md"))).read_text()
    assert r"pipe\|inside" in md
    assert "| a | b |" in md


def test_xlsx_to_csv(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "book.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["city", "population"])
    sheet.append(["Bengaluru", 13600000])
    workbook.save(path)

    rows = list(
        csv.DictReader(
            io.StringIO(Path(convert_file(str(path), str(tmp_path / "out.csv"))).read_text())
        )
    )
    assert rows == [{"city": "Bengaluru", "population": "13600000"}]


# --- config -------------------------------------------------------------


def test_env_to_json_and_back(env_file, tmp_path):
    pytest.importorskip("yaml")
    data = json.loads(
        Path(convert_file(str(env_file), str(tmp_path / "out.json"))).read_text()
    )
    assert data == {"APP_NAME": "demo", "PORT": "8080", "DEBUG": "true"}


def test_json_toml_json_round_trip(tmp_path):
    pytest.importorskip("tomli_w")
    source = tmp_path / "conf.json"
    source.write_text(json.dumps({"server": {"host": "0.0.0.0", "port": 8080}}))
    as_toml = convert_file(str(source), str(tmp_path / "conf.toml"))
    assert "[server]" in Path(as_toml).read_text()

    back = json.loads(Path(convert_file(as_toml, str(tmp_path / "back.json"))).read_text())
    assert back == {"server": {"host": "0.0.0.0", "port": 8080}}


def test_multi_document_yaml_is_preserved(tmp_path):
    pytest.importorskip("yaml")
    source = tmp_path / "manifests.yaml"
    source.write_text("kind: Service\nname: a\n---\nkind: Deployment\nname: b\n")
    data = json.loads(
        Path(convert_file(str(source), str(tmp_path / "out.json"))).read_text()
    )
    assert [doc["kind"] for doc in data] == ["Service", "Deployment"]


def test_terraform_to_json(tmp_path):
    source = tmp_path / "main.tf"
    source.write_text(
        'resource "aws_s3_bucket" "assets" {\n  bucket = "my-assets"\n  acl = "private"\n}\n'
    )
    data = json.loads(
        Path(convert_file(str(source), str(tmp_path / "out.json"))).read_text()
    )
    assert data["aws_s3_bucket"]["assets"]["bucket"] == "my-assets"


# --- images -------------------------------------------------------------


def test_image_format_conversion(png_file, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    out = Path(convert_file(str(png_file), str(tmp_path / "photo.jpg")))
    with Image.open(out) as img:
        assert img.format == "JPEG"
        assert img.size == (400, 300)


def test_resize_never_overwrites_the_source(png_file):
    Image = pytest.importorskip("PIL.Image")
    from universal_converter import resize_image

    before = png_file.read_bytes()
    out = Path(resize_image(str(png_file), width=100))

    assert out != png_file
    assert png_file.read_bytes() == before  # the regression this test exists for
    with Image.open(out) as img:
        assert img.size == (100, 75)


def test_thumbnail_never_overwrites_the_source(png_file):
    Image = pytest.importorskip("PIL.Image")
    from universal_converter import thumbnail

    before = png_file.read_bytes()
    out = Path(thumbnail(str(png_file)))
    assert png_file.read_bytes() == before
    with Image.open(out) as img:
        assert max(img.size) <= 128


def test_rgba_to_jpeg_flattens_alpha(png_file, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    out = Path(convert_file(str(png_file), str(tmp_path / "flat.jpg")))
    with Image.open(out) as img:
        assert img.mode == "RGB"


# --- media --------------------------------------------------------------

ffmpeg_required = pytest.mark.skipif(
    not __import__("shutil").which("ffmpeg"), reason="ffmpeg not on PATH"
)


@pytest.fixture
def wav_file(tmp_path):
    path = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1", str(path)],
        check=True,
    )
    return path


@ffmpeg_required
def test_audio_conversion_returns_a_result_not_a_typeerror(wav_file, tmp_path):
    # Regression: run_command returns a CompletedProcess, and every media call
    # site subscripted it as a dict -- all 72 audio/video pairs raised
    # TypeError: 'CompletedProcess' object is not subscriptable.
    from universal_converter.converters.audio import AudioConverter

    result = AudioConverter().convert_format(str(wav_file), "mp3")
    assert result.success, result.error
    assert Path(result.output_path).stat().st_size > 0


@ffmpeg_required
def test_audio_failure_is_reported_not_raised(tmp_path):
    from universal_converter.converters.audio import AudioConverter

    broken = tmp_path / "not-audio.wav"
    broken.write_text("this is not a wav file")
    result = AudioConverter().convert_format(str(broken), "mp3")
    assert result.success is False
    assert result.error


@ffmpeg_required
def test_video_conversion_and_probe(tmp_path):
    from universal_converter.converters.video import VideoConverter

    source = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi",
         "-i", "testsrc=size=64x64:rate=10:duration=1", str(source)],
        check=True,
    )

    converted = VideoConverter().convert_format(str(source), "mkv")
    assert converted.success, converted.error

    info = VideoConverter().get_info(str(source))
    assert info.success, info.error
    # Only parseable because run_command now passes text=True.
    assert info.data["streams"][0]["codec_type"] == "video"


@ffmpeg_required
def test_video_to_webm_picks_a_legal_codec(tmp_path):
    from universal_converter.converters.video import VideoConverter

    source = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi",
         "-i", "testsrc=size=64x64:rate=10:duration=1", str(source)],
        check=True,
    )
    result = VideoConverter().convert_format(str(source), "webm")
    assert result.success, result.error


# --- every advertised pair has a code path -------------------------------


def test_no_pair_is_advertised_without_an_implementation(tmp_path):
    """The registry must not list a pair whose converter answers 'Unsupported'.

    33 advertised pairs used to fall straight through to an ``Unsupported``
    result; this walks the whole map and fails if any still does.
    """
    from universal_converter.converters import ConversionTask

    registry = get_registry()
    source = tmp_path / "probe"
    source.write_text("")
    offenders = []

    for src, targets in registry.supported_conversions().items():
        for tgt in targets:
            converter = registry.find_converter(src, tgt)
            task = ConversionTask(
                source_path=source,
                target_path=tmp_path / f"out.{tgt}",
                source_format=src,
                target_format=tgt,
            )
            result = converter().convert(task)
            # Failing on empty input is fine; claiming the pair is unknown is not.
            if result.error and result.error.lower().startswith("unsupported"):
                offenders.append(f"{src}->{tgt} ({converter.__name__})")

    assert not offenders, offenders


def test_console_script_module_entry_point_runs():
    result = subprocess.run(
        [sys.executable, "-m", "universal_converter", "--list"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "direct pairs" in result.stdout
    assert "json" in result.stdout


def test_helpers_still_exported():
    assert DataConverter().can_handle("json", "csv")
    assert NetworkConverter.html_to_text("<p>hi</p>") == "hi"
    from universal_converter.converters import DataConverter as Lazy

    assert Lazy is DataConverter
