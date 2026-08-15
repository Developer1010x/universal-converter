"""Registry discovery, priority ordering, alias handling and route planning."""

import doctest

import pytest

from universal_converter import registry as registry_module
from universal_converter.converters import BaseConverter
from universal_converter.registry import ConverterRegistry, format_route


@pytest.fixture(scope="module")
def registry():
    return ConverterRegistry()


def test_registry_docstring_examples_run():
    results = doctest.testmod(registry_module, verbose=False)
    assert results.failed == 0


def test_discovery_finds_every_converter_module(registry):
    names = {converter.__name__ for converter in registry.converters}
    assert {
        "AudioConverter",
        "BioinformaticsConverter",
        "CloudConverter",
        "DataConverter",
        "DatabaseConverter",
        "DocumentConverter",
        "GISConverter",
        "ImageConverter",
        "NetworkConverter",
        "VideoConverter",
    } <= names


def test_no_module_failed_to_import(registry):
    assert registry.import_errors == {}


def test_priority_is_ascending_and_specialists_win(registry):
    priorities = [converter.PRIORITY for converter in registry.converters]
    assert priorities == sorted(priorities)
    # Regression: the sort used to be descending, so the generic DataConverter
    # (PRIORITY 50) outranked every specialist.
    assert registry.find_converter("tiff", "png").__name__ == "ImageConverter"
    assert registry.find_converter("png", "jpg").__name__ == "ImageConverter"


def test_aliases_resolve_to_canonical_formats(registry):
    assert registry.detect_format("config.YML") == "yaml"
    assert registry.detect_format("photo.JPEG") == "jpg"
    assert registry.detect_format("notes.markdown") == "md"
    assert registry.detect_format("readme.text") == "txt"
    assert registry.detect_format(".env") == "env"
    assert registry.detect_format("noextension") == "noextension"


def test_alias_targets_are_reachable(registry):
    # NetworkConverter used to declare 'markdown'/'text', which detect_format
    # normalises away -- making those pairs unreachable from any real path.
    assert registry.can_convert("html", registry.detect_format("out.markdown"))
    assert registry.can_convert("html", registry.detect_format("out.text"))


def test_every_declared_format_name_is_canonical(registry):
    for converter in registry.converters:
        for source, targets in converter.SUPPORTED_CONVERSIONS.items():
            assert source == registry.normalize(source), converter
            for target in targets:
                assert target == registry.normalize(target), converter


def test_no_duplicate_alias_subclasses(registry):
    # Four no-op subclasses used to register as separate converters.
    names = [converter.__name__ for converter in registry.converters]
    assert len(names) == len(set(names))
    for converter in registry.converters:
        parents = [
            base
            for base in converter.__mro__[1:]
            if issubclass(base, BaseConverter) and base is not BaseConverter
        ]
        assert not parents, f"{converter.__name__} is an alias subclass"


def test_direct_route_is_a_single_step(registry):
    route = registry.find_route("json", "csv")
    assert len(route) == 1
    assert route[0].source == "json" and route[0].target == "csv"


def test_multi_hop_route_is_found_and_shortest(registry):
    route = registry.find_route("csv", "pdf")
    assert format_route(route) == "csv -> html -> pdf"
    assert len(route) == 2


def test_route_returns_none_when_unreachable(registry):
    assert registry.find_route("mp3", "pdf", max_hops=3) is None


def test_route_respects_max_hops(registry):
    assert registry.find_route("sqlite", "md", max_hops=1) is None
    assert registry.find_route("sqlite", "md", max_hops=2) is not None


def test_identity_route_is_empty(registry):
    assert registry.find_route("json", "json") == []


def test_reachable_from_reports_hop_counts(registry):
    reachable = registry.reachable_from("json", max_hops=2)
    assert reachable["csv"] == 1
    assert reachable["pdf"] == 2


def test_supported_conversions_is_sorted_and_deduplicated(registry):
    conversions = registry.supported_conversions()
    assert list(conversions) == sorted(conversions)
    for targets in conversions.values():
        assert targets == sorted(set(targets))


def test_pair_count_matches_the_map(registry):
    assert registry.pair_count() == sum(
        len(t) for t in registry.supported_conversions().values()
    )
