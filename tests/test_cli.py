"""CLI behaviour: dispatch, routing, doctor, and the error paths."""

import json

import pytest

from universal_converter import main


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"a": 1, "b": 2}, {"a": 3, "b": 4}]))
    return path


def test_list_prints_the_capability_map(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "direct pairs" in out and "routed" in out
    assert "json" in out and "csv" in out


def test_doctor_reports_readiness(capsys):
    assert main(["--doctor"]) == 0
    out = capsys.readouterr().out
    assert "conversion pairs are runnable" in out
    assert "DataConverter" in out


def test_route_command_shows_a_multi_hop_path(capsys):
    assert main(["--route", "csv", "pdf"]) == 0
    out = capsys.readouterr().out
    assert "csv -> html -> pdf" in out
    assert "2 hops" in out


def test_route_command_reports_unreachable(capsys):
    assert main(["--route", "mp3", "pdf"]) == 1
    assert "no route" in capsys.readouterr().err


def test_cli_dispatches_by_target_format(sample, tmp_path):
    assert main([str(sample), "-t", "csv", "-o", str(tmp_path / "out.csv")]) == 0
    assert (tmp_path / "out.csv").read_text().startswith("a,b")


def test_cli_infers_target_from_output_extension(sample, tmp_path):
    assert main([str(sample), "-o", str(tmp_path / "out.yaml")]) == 0
    assert "a:" in (tmp_path / "out.yaml").read_text()


def test_cli_routes_images_to_the_image_converter(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    source = tmp_path / "pic.png"
    Image.new("RGB", (8, 8), "red").save(source)

    # Regression: the CLI hardcoded DataConverter, so a PNG was fed to the
    # JSON/CSV text converter and failed with a decode error.
    assert main([str(source), "-t", "jpg", "-o", str(tmp_path / "pic.jpg")]) == 0
    with Image.open(tmp_path / "pic.jpg") as img:
        assert img.format == "JPEG"


def test_cli_uses_a_multi_hop_route_automatically(tmp_path, capsys):
    source = tmp_path / "rows.csv"
    source.write_text("city,pop\nBengaluru,13600000\n")
    assert main([str(source), "-t", "md", "-o", str(tmp_path / "rows.md")]) == 0
    assert "| city | pop |" in (tmp_path / "rows.md").read_text()


def test_direct_flag_refuses_a_chained_route(tmp_path, capsys):
    source = tmp_path / "rows.csv"
    source.write_text("city,pop\nBengaluru,1\n")
    assert main([str(source), "-t", "pdf", "--direct", "-o", str(tmp_path / "x.pdf")]) == 1
    assert "no conversion path" in capsys.readouterr().err


def test_from_override_is_respected(tmp_path):
    # Extension lies about the content; --from corrects it.
    source = tmp_path / "actually.json"
    source.write_text(json.dumps({"k": "v"}))
    assert main([str(source), "--from", "json", "-t", "yaml", "-o", str(tmp_path / "o.yaml")]) == 0
    assert "k: v" in (tmp_path / "o.yaml").read_text()


def test_missing_input_file_is_an_error(capsys, tmp_path):
    assert main([str(tmp_path / "nope.json"), "-t", "csv"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_unknown_pair_is_an_error(capsys, tmp_path):
    source = tmp_path / "a.mp3"
    source.write_bytes(b"\x00")
    assert main([str(source), "-t", "pdf"]) == 1
    assert "no conversion path" in capsys.readouterr().err


def test_refuses_to_write_over_the_input(capsys, sample):
    assert main([str(sample), "-o", str(sample)]) == 1
    assert "same as the input" in capsys.readouterr().err


def test_no_arguments_prints_help(capsys):
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_quiet_suppresses_success_output(sample, tmp_path, capsys):
    assert main([str(sample), "-o", str(tmp_path / "o.csv"), "-q"]) == 0
    assert capsys.readouterr().out == ""
