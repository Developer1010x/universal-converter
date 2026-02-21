"""GIS and geospatial file format conversion module"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from . import BaseConverter, ConversionTask, ConversionResult


class GISConverter(BaseConverter):
    """Converter for GIS/geospatial file formats"""
    
    SUPPORTED_CONVERSIONS = {
        "shp": ["geojson", "kml", "gpx"],
        "geojson": ["shp", "kml", "gpx"],
        "kml": ["geojson", "shp"],
        "gpx": ["geojson", "shp"],
        "tif": ["png", "jpg", "jp2"],
        "tiff": ["png", "jpg", "jp2"],
        "jp2": ["tif", "png"],
    }
    PRIORITY = 25
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()
        
        try:
            if source in ["shp", "geojson", "kml", "gpx"] and target in ["shp", "geojson", "kml", "gpx"]:
                return self._vector_convert(task)
            elif source in ["tif", "tiff", "jp2"] and target in ["png", "jpg", "jp2", "tif"]:
                return self._raster_convert(task)
            
            return ConversionResult(success=False, error=f"Unsupported: {source} → {target}")
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _vector_convert(self, task: ConversionTask) -> ConversionResult:
        try:
            import fiona
            import json
            
            with fiona.open(task.source_path) as src:
                features = list(src)
                meta = src.meta
            
            if task.target_format == "geojson":
                geojson = {
                    "type": "FeatureCollection",
                    "features": features
                }
                task.target_path.write_text(json.dumps(geojson, indent=2))
            elif task.target_format == "kml":
                kml = self._to_kml(features)
                task.target_path.write_text(kml)
            elif task.target_format == "gpx":
                gpx = self._to_gpx(features)
                task.target_path.write_text(gpx)
            else:
                with fiona.open(task.target_path, 'w', **meta) as dst:
                    dst.writerecords(features)
            
            return ConversionResult(success=True, output_path=str(task.target_path))
        except ImportError:
            return ConversionResult(
                success=False,
                error="Fiona/Shapely required. Install: pip install universal-converter[gis]"
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _to_kml(self, features: List[Dict]) -> str:
        kml_parts = ['<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>']
        
        for feat in features:
            name = feat.get('properties', {}).get('name', 'Unnamed')
            geom = feat.get('geometry', {})
            geom_type = geom.get('type')
            coords = geom.get('coordinates', [])
            
            if geom_type == 'Point':
                kml_parts.append(f'<Placemark><name>{name}</name><Point><coordinates>{coords[0]},{coords[1]}</coordinates></Point></Placemark>')
            elif geom_type == 'LineString':
                coord_str = ' '.join([f"{c[0]},{c[1]}" for c in coords])
                kml_parts.append(f'<Placemark><name>{name}</name><LineString><coordinates>{coord_str}</coordinates></LineString></Placemark>')
            elif geom_type == 'Polygon':
                outer = coords[0] if coords else []
                coord_str = ' '.join([f"{c[0]},{c[1]}" for c in outer])
                kml_parts.append(f'<Placemark><name>{name}</name><Polygon><outerBoundaryIs><LinearRing><coordinates>{coord_str}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
        
        kml_parts.append('</Document></kml>')
        return '\n'.join(kml_parts)
    
    def _to_gpx(self, features: List[Dict]) -> str:
        gpx_parts = ['<?xml version="1.0" encoding="UTF-8"?><gpx version="1.1">']
        
        for feat in features:
            name = feat.get('properties', {}).get('name', 'Unnamed')
            geom = feat.get('geometry', {})
            geom_type = geom.get('type')
            coords = geom.get('coordinates', [])
            
            if geom_type == 'Point':
                gpx_parts.append(f'<wpt lat="{coords[1]}" lon="{coords[0]}"><name>{name}</name></wpt>')
            elif geom_type == 'LineString':
                gpx_parts.append(f'<rte><name>{name}</name>')
                for c in coords:
                    gpx_parts.append(f'<rtept lat="{c[1]}" lon="{c[0]}"/>')
                gpx_parts.append('</rte>')
        
        gpx_parts.append('</gpx>')
        return '\n'.join(gpx_parts)
    
    def _raster_convert(self, task: ConversionTask) -> ConversionResult:
        try:
            from PIL import Image
            
            img = Image.open(task.source_path)
            
            if task.target_format in ["jpg", "jpeg"]:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
            
            img.save(task.target_path)
            return ConversionResult(success=True, output_path=str(task.target_path))
        except ImportError:
            return ConversionResult(
                success=False,
                error="Pillow required. Install: pip install universal-converter[images]"
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def convert_format(self, path: str, to_format: str, output: Optional[str] = None) -> ConversionResult:
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format
        )
        return self.convert(task)


class GeoJSONConverter(GISConverter):
    """Alias for GeoJSON-specific conversions"""
    pass


class GPXConverter(BaseConverter):
    """Converter for GPX files"""
    
    SUPPORTED_CONVERSIONS = {
        "gpx": ["geojson", "kml", "csv"],
    }
    PRIORITY = 30
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(task.source_path)
            root = tree.getroot()
            
            ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
            
            features = []
            for wpt in root.findall('.//gpx:wpt', ns):
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(wpt.get('lon')), float(wpt.get('lat'))]
                    },
                    "properties": {
                        "name": wpt.find('gpx:name', ns).text if wpt.find('gpx:name', ns) is not None else None
                    }
                }
                features.append(feat)
            
            import json
            geojson = {"type": "FeatureCollection", "features": features}
            
            task.target_path.write_text(json.dumps(geojson, indent=2))
            return ConversionResult(success=True, output_path=str(task.target_path))
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
