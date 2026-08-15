"""Geospatial vector converters (GeoJSON, KML, GPX, CSV).

Everything here runs on the standard library. Shapefile support was removed
rather than left advertised: it needs GDAL/fiona, which cannot be installed
from PyPI alone on most platforms, and the previous code path could never
succeed without it.
"""

import csv
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import BaseConverter, ConversionResult, ConversionTask

_KML_NS = "http://www.opengis.net/kml/2.2"
_GPX_NS = "http://www.topografix.com/GPX/1/1"


class GISConverter(BaseConverter):
    """Converter between GeoJSON, KML, GPX and flat CSV point lists."""

    SUPPORTED_CONVERSIONS = {
        "geojson": ["kml", "gpx", "csv"],
        "kml": ["geojson", "gpx", "csv"],
        "gpx": ["geojson", "kml", "csv"],
    }
    PRIORITY = 25

    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()

        try:
            readers = {
                "geojson": self._read_geojson,
                "kml": self._read_kml,
                "gpx": self._read_gpx,
            }
            writers = {
                "geojson": self._write_geojson,
                "kml": self._write_kml,
                "gpx": self._write_gpx,
                "csv": self._write_csv,
            }
            if source not in readers or target not in writers:
                return ConversionResult(
                    success=False, error=f"Unsupported: {source} -> {target}"
                )

            features = readers[source](Path(task.source_path).read_text(encoding="utf-8"))
            Path(task.target_path).write_text(writers[target](features), encoding="utf-8")
            return ConversionResult(
                success=True,
                output_path=str(task.target_path),
                metadata={"features": len(features)},
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    # --- readers ---------------------------------------------------------

    @staticmethod
    def _read_geojson(content: str) -> List[Dict[str, Any]]:
        data = json.loads(content)
        if data.get("type") == "FeatureCollection":
            return data.get("features", [])
        if data.get("type") == "Feature":
            return [data]
        # A bare geometry object is still valid GeoJSON.
        return [{"type": "Feature", "geometry": data, "properties": {}}]

    @classmethod
    def _read_kml(cls, content: str) -> List[Dict[str, Any]]:
        root = ET.fromstring(content)
        features: List[Dict[str, Any]] = []

        for placemark in root.iter(f"{{{_KML_NS}}}Placemark"):
            name = placemark.find(f"{{{_KML_NS}}}name")
            properties = {"name": name.text if name is not None else None}

            for tag, geom_type in (
                ("Point", "Point"),
                ("LineString", "LineString"),
                ("Polygon", "Polygon"),
            ):
                node = placemark.find(f".//{{{_KML_NS}}}{tag}")
                if node is None:
                    continue
                coord_node = node.find(f".//{{{_KML_NS}}}coordinates")
                if coord_node is None or not coord_node.text:
                    continue
                coords = cls._parse_kml_coordinates(coord_node.text)
                if geom_type == "Point":
                    geometry = {"type": "Point", "coordinates": coords[0]}
                elif geom_type == "LineString":
                    geometry = {"type": "LineString", "coordinates": coords}
                else:
                    geometry = {"type": "Polygon", "coordinates": [coords]}
                features.append(
                    {"type": "Feature", "geometry": geometry, "properties": properties}
                )
                break
        return features

    @staticmethod
    def _parse_kml_coordinates(text: str) -> List[List[float]]:
        coords = []
        for chunk in text.split():
            parts = chunk.split(",")
            if len(parts) >= 2:
                coords.append([float(parts[0]), float(parts[1])])
        return coords

    @staticmethod
    def _read_gpx(content: str) -> List[Dict[str, Any]]:
        root = ET.fromstring(content)
        features: List[Dict[str, Any]] = []

        def name_of(node) -> Optional[str]:
            child = node.find(f"{{{_GPX_NS}}}name")
            return child.text if child is not None else None

        for wpt in root.iter(f"{{{_GPX_NS}}}wpt"):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(wpt.get("lon")), float(wpt.get("lat"))],
                    },
                    "properties": {"name": name_of(wpt)},
                }
            )

        for container, point_tag in (("rte", "rtept"), ("trkseg", "trkpt")):
            for node in root.iter(f"{{{_GPX_NS}}}{container}"):
                coords = [
                    [float(p.get("lon")), float(p.get("lat"))]
                    for p in node.iter(f"{{{_GPX_NS}}}{point_tag}")
                ]
                if coords:
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": {"name": name_of(node)},
                        }
                    )
        return features

    # --- writers ---------------------------------------------------------

    @staticmethod
    def _write_geojson(features: List[Dict[str, Any]]) -> str:
        return json.dumps(
            {"type": "FeatureCollection", "features": features}, indent=2
        )

    @classmethod
    def _write_kml(cls, features: List[Dict[str, Any]]) -> str:
        from xml.sax.saxutils import escape

        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<kml xmlns="{_KML_NS}"><Document>',
        ]
        for feature in features:
            geometry = feature.get("geometry") or {}
            name = escape(str((feature.get("properties") or {}).get("name") or "Unnamed"))
            body = cls._kml_geometry(geometry)
            if body:
                parts.append(f"<Placemark><name>{name}</name>{body}</Placemark>")
        parts.append("</Document></kml>")
        return "\n".join(parts)

    @staticmethod
    def _kml_geometry(geometry: Dict[str, Any]) -> str:
        def joined(points) -> str:
            return " ".join(f"{p[0]},{p[1]}" for p in points)

        geom_type = geometry.get("type")
        coords = geometry.get("coordinates") or []
        if geom_type == "Point" and coords:
            return f"<Point><coordinates>{coords[0]},{coords[1]}</coordinates></Point>"
        if geom_type == "LineString" and coords:
            return f"<LineString><coordinates>{joined(coords)}</coordinates></LineString>"
        if geom_type == "Polygon" and coords:
            return (
                "<Polygon><outerBoundaryIs><LinearRing><coordinates>"
                f"{joined(coords[0])}"
                "</coordinates></LinearRing></outerBoundaryIs></Polygon>"
            )
        return ""

    @staticmethod
    def _write_gpx(features: List[Dict[str, Any]]) -> str:
        from xml.sax.saxutils import escape

        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<gpx version="1.1" creator="universal-converter" xmlns="{_GPX_NS}">',
        ]
        for feature in features:
            geometry = feature.get("geometry") or {}
            name = escape(str((feature.get("properties") or {}).get("name") or "Unnamed"))
            coords = geometry.get("coordinates") or []
            if geometry.get("type") == "Point" and coords:
                parts.append(
                    f'<wpt lat="{coords[1]}" lon="{coords[0]}"><name>{name}</name></wpt>'
                )
            elif geometry.get("type") == "LineString" and coords:
                parts.append(f"<rte><name>{name}</name>")
                parts.extend(f'<rtept lat="{c[1]}" lon="{c[0]}"/>' for c in coords)
                parts.append("</rte>")
        parts.append("</gpx>")
        return "\n".join(parts)

    @staticmethod
    def _write_csv(features: List[Dict[str, Any]]) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["name", "type", "longitude", "latitude"])
        for feature in features:
            geometry = feature.get("geometry") or {}
            name = (feature.get("properties") or {}).get("name") or ""
            coords = geometry.get("coordinates") or []
            geom_type = geometry.get("type")
            if geom_type == "Point" and coords:
                writer.writerow([name, geom_type, coords[0], coords[1]])
            elif geom_type == "LineString":
                for point in coords:
                    writer.writerow([name, geom_type, point[0], point[1]])
            elif geom_type == "Polygon" and coords:
                for point in coords[0]:
                    writer.writerow([name, geom_type, point[0], point[1]])
        return buffer.getvalue()

    def convert_format(
        self, path: str, to_format: str, output: Optional[str] = None
    ) -> ConversionResult:
        """Convert a geospatial file to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)
