#!/usr/bin/env python3
"""
Universal Converter - Convert Anything to Anything
OS-neutral, minimal dependencies, comprehensive conversion tool

Install extras:
    pip install universal-converter[images]     # Image conversions
    pip install universal-converter[audio]     # Audio conversions
    pip install universal-converter[video]     # Video conversions
    pip install universal-converter[cloud]     # Cloud storage
    pip install universal-converter[ai]        # AI/NLP features
    pip install universal-converter[all]        # All features
"""

import os
import sys
import json
import csv
import xml.etree.ElementTree as ET
import base64
import hashlib
import shutil
import tempfile
import subprocess
import logging
import struct
import zipfile
import tarfile
import configparser
import re
import mimetypes
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import traceback
import io
import stat
from contextlib import contextmanager


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversionError(Exception):
    """Custom exception for conversion errors"""
    def __init__(self, message: str, source_format: str = None, target_format: str = None, details: str = None):
        self.source_format = source_format
        self.target_format = target_format
        self.details = details
        super().__init__(message)
    
    def __str__(self):
        msg = super().__str__()
        if self.source_format and self.target_format:
            msg = f"{msg} ({self.source_format} → {self.target_format})"
        return msg


class DependencyError(ConversionError):
    """Raised when an optional dependency is missing"""
    def __init__(self, package: str, feature: str):
        self.package = package
        self.feature = feature
        super().__init__(
            f"Feature '{feature}' requires '{package}'. Install with: pip install universal-converter[{feature}]"
        )


@dataclass
class ConversionResult:
    """Result of a conversion operation"""
    success: bool
    data: Any = None
    output_path: str = None
    error: str = None
    metadata: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@contextmanager
def safe_tempdir():
    """Safe temporary directory that always cleans up"""
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


def check_dependency(package: str, import_name: str = None):
    """Check if a dependency is available, raise helpful error if not"""
    if import_name is None:
        import_name = package
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def require_dependency(package: str, feature: str = None):
    """Require a dependency, raise DependencyError if missing"""
    import_name = package.replace('-', '_').replace(' ', '_')
    if not check_dependency(package, import_name):
        raise DependencyError(package, feature or package)


# ============ CORE DATA CONVERSIONS (NO DEPENDENCIES) ============


# ============ AI / NLP CONVERSIONS ============

def text_to_embeddings(text: str, model: str = "sentence-transformers") -> List[float]:
    """Convert text to embeddings"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ConversionError("sentence-transformers required: pip install sentence-transformers")
    
    model_obj = SentenceTransformer(model)
    embeddings = model_obj.encode(text)
    return embeddings.tolist()


def text_to_tokens(text: str, model: str = "gpt2") -> List[int]:
    """Convert text to tokens"""
    try:
        import tiktoken
    except ImportError:
        raise ConversionError("tiktoken required: pip install tiktoken")
    
    enc = tiktoken.get_encoding(model)
    return enc.encode(text)


def tokens_to_text(tokens: List[int], model: str = "gpt2") -> str:
    """Convert tokens to text"""
    try:
        import tiktoken
    except ImportError:
        raise ConversionError("tiktoken required: pip install tiktoken")
    
    enc = tiktoken.get_encoding(model)
    return enc.decode(tokens)


def audio_to_transcription(audio_path: str, language: str = "en") -> str:
    """Convert audio to transcription using Whisper AI"""
    try:
        import whisper
    except ImportError:
        raise ConversionError("openai-whisper required: pip install openai-whisper")
    
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=language)
    return result["text"]


def text_to_speech(text: str, output_path: str = None, language: str = "en") -> str:
    """Convert text to speech (TTS)"""
    try:
        from gtts import gTTS
    except ImportError:
        raise ConversionError("gtts required: pip install gtts")
    
    if output_path is None:
        output_path = "speech.mp3"
    
    tts = gTTS(text=text, lang=language)
    tts.save(output_path)
    return output_path


def text_to_speech_offline(text: str, output_path: str = None) -> str:
    """Convert text to speech using offline TTS"""
    try:
        import pyttsx3
    except ImportError:
        raise ConversionError("pyttsx3 required: pip install pyttsx3")
    
    if output_path is None:
        output_path = "speech.mp3"
    
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path


# ============ CLOUD STORAGE CONVERSIONS ============

def upload_to_s3(file_path: str, bucket: str, key: str = None, aws_access_key: str = None, aws_secret_key: str = None) -> str:
    """Upload file to AWS S3"""
    try:
        import boto3
    except ImportError:
        raise ConversionError("boto3 required: pip install boto3")
    
    if key is None:
        key = Path(file_path).name
    
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    s3.upload_file(file_path, bucket, key)
    return f"s3://{bucket}/{key}"


def download_from_s3(bucket: str, key: str, output_path: str = None, aws_access_key: str = None, aws_secret_key: str = None) -> str:
    """Download file from AWS S3"""
    try:
        import boto3
    except ImportError:
        raise ConversionError("boto3 required: pip install boto3")
    
    if output_path is None:
        output_path = key.split('/')[-1]
    
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    s3.download_file(bucket, key, output_path)
    return output_path


def upload_to_gcs(file_path: str, bucket_name: str, destination_name: str = None, credentials_path: str = None) -> str:
    """Upload file to Google Cloud Storage"""
    try:
        from google.cloud import storage
    except ImportError:
        raise ConversionError("google-cloud-storage required: pip install google-cloud-storage")
    
    if destination_name is None:
        destination_name = Path(file_path).name
    
    client = storage.Client.from_service_account_json(credentials_path) if credentials_path else storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_name)
    blob.upload_from_filename(file_path)
    return f"gs://{bucket_name}/{destination_name}"


def download_from_gcs(bucket_name: str, source_name: str, output_path: str = None, credentials_path: str = None) -> str:
    """Download file from Google Cloud Storage"""
    try:
        from google.cloud import storage
    except ImportError:
        raise ConversionError("google-cloud-storage required: pip install google-cloud-storage")
    
    if output_path is None:
        output_path = source_name.split('/')[-1]
    
    client = storage.Client.from_service_account_json(credentials_path) if credentials_path else storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_name)
    blob.download_to_filename(output_path)
    return output_path


def upload_to_azure(file_path: str, container: str, blob_name: str = None, connection_string: str = None) -> str:
    """Upload file to Azure Blob Storage"""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ConversionError("azure-storage-blob required: pip install azure-storage-blob")
    
    if blob_name is None:
        blob_name = Path(file_path).name
    
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
    blob_client.upload_blob_from_path(file_path, overwrite=True)
    return f"azure://{container}/{blob_name}"


def download_from_azure(container: str, blob_name: str, output_path: str = None, connection_string: str = None) -> str:
    """Download file from Azure Blob Storage"""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ConversionError("azure-storage-blob required: pip install azure-storage-blob")
    
    if output_path is None:
        output_path = blob_name.split('/')[-1]
    
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
    with open(output_path, "wb") as f:
        f.write(blob_client.download_blob().readall())
    return output_path


# ============ BLOCKCHAIN / CRYPTO CONVERSIONS ============

def generate_key_pair() -> Dict:
    """Generate Ethereum key pair"""
    try:
        from eth_account import Account
    except ImportError:
        raise ConversionError("eth-account required: pip install eth-account")
    
    acct = Account.create()
    return {
        'address': acct.address,
        'private_key': acct.key.hex()
    }


def private_key_to_address(private_key: str) -> str:
    """Convert private key to address"""
    try:
        from eth_account import Account
    except ImportError:
        raise ConversionError("eth-account required: pip install eth-account")
    
    acct = Account.from_key(private_key)
    return acct.address


def text_to_keccak_hash(text: str) -> str:
    """Convert text to Keccak-256 hash"""
    try:
        from web3 import Web3
    except ImportError:
        raise ConversionError("web3 required: pip install web3")
    
    w3 = Web3()
    return w3.keccak(text.encode()).hex()


def hex_to_bytes(hex_string: str) -> bytes:
    """Convert hex string to bytes"""
    return bytes.fromhex(hex_string.replace('0x', ''))


def bytes_to_hex(data: bytes) -> str:
    """Convert bytes to hex string"""
    return '0x' + data.hex()


# ============ BIOINFORMATICS CONVERSIONS ============

def fasta_to_dict(fasta_path: str) -> Dict[str, str]:
    """Parse FASTA file to dictionary"""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    
    if current_id:
        sequences[current_id] = ''.join(current_seq)
    
    return sequences


def dict_to_fasta(sequences: Dict[str, str], output_path: str = None) -> str:
    """Convert dictionary to FASTA format"""
    lines = []
    for seq_id, seq in sequences.items():
        lines.append(f">{seq_id}")
        for i in range(0, len(seq), 80):
            lines.append(seq[i:i+80])
    
    fasta_str = '\n'.join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(fasta_str)
    
    return fasta_str


def genbank_to_fasta(genbank_path: str, output_path: str = None) -> str:
    """Convert GenBank to FASTA"""
    try:
        from Bio import SeqIO
    except ImportError:
        raise ConversionError("biopython required: pip install biopython")
    
    records = list(SeqIO.parse(genbank_path, "genbank"))
    return SeqIO.write(records, output_path or genbank_path.replace('.gb', '.fasta'), "fasta")


def fasta_to_sequence(fasta_path: str) -> List[Dict]:
    """Convert FASTA to sequence objects"""
    try:
        from Bio import SeqIO
    except ImportError:
        raise ConversionError("biopython required: pip install biopython")
    
    records = list(SeqIO.parse(fasta_path, "fasta"))
    return [{'id': r.id, 'seq': str(r.seq), 'description': r.description} for r in records]


# ============ DOCKER / CONTAINER CONVERSIONS ============

def dockerfile_to_image(dockerfile_path: str, image_name: str, tag: str = "latest") -> str:
    """Build Docker image from Dockerfile"""
    import subprocess
    
    image_full = f"{image_name}:{tag}"
    subprocess.run(["docker", "build", "-t", image_full, "-f", dockerfile_path, "."], check=True)
    return image_full


def image_to_tar(image_name: str, output_path: str = None) -> str:
    """Export Docker image to tar file"""
    import subprocess
    
    if output_path is None:
        output_path = f"{image_name.replace('/', '_').replace(':', '_')}.tar"
    
    subprocess.run(["docker", "save", "-o", output_path, image_name], check=True)
    return output_path


def tar_to_image(tar_path: str) -> str:
    """Import tar file to Docker image"""
    import subprocess
    
    subprocess.run(["docker", "load", "-i", tar_path], check=True)
    return "Image loaded successfully"


# ============ STRUCTURED DATA CONVERSIONS ============

def struct_to_bytes(format_string: str, *args) -> bytes:
    """Pack data into binary format using struct"""
    return struct.pack(format_string, *args)


def bytes_to_struct(format_string: str, data: bytes) -> tuple:
    """Unpack binary data using struct"""
    return struct.unpack(format_string, data)


def json_to_prometheus(json_data: Dict, metric_name: str) -> str:
    """Convert JSON to Prometheus format"""
    lines = []
    for key, value in json_data.items():
        if isinstance(value, (int, float)):
            lines.append(f"{metric_name}_{key} {value}")
    return '\n'.join(lines)


def parse_syslog(syslog_line: str) -> Dict:
    """Parse syslog line to structured data"""
    import re
    
    pattern = r'^(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+(\S+):\s+(.*)$'
    match = re.match(pattern, syslog_line)
    
    if match:
        return {
            'timestamp': match.group(1),
            'hostname': match.group(2),
            'process': match.group(3),
            'message': match.group(4)
        }
    return {'raw': syslog_line}


# ============ KAFKA / REDIS / MQTT ============

def send_to_kafka(topic: str, message: str, bootstrap_servers: List[str] = ['localhost:9092']) -> bool:
    """Send message to Kafka topic"""
    try:
        from kafka import KafkaProducer
    except ImportError:
        raise ConversionError("kafka-python required: pip install kafka-python")
    
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    producer.send(topic, message.encode())
    producer.flush()
    return True


def consume_from_kafka(topic: str, bootstrap_servers: List[str] = ['localhost:9092'], max_messages: int = 10) -> List[str]:
    """Consume messages from Kafka topic"""
    try:
        from kafka import KafkaConsumer
    except ImportError:
        raise ConversionError("kafka-python required: pip install kafka-python")
    
    consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, auto_offset_reset='earliest')
    messages = []
    for i, message in enumerate(consumer):
        if i >= max_messages:
            break
        messages.append(message.value.decode())
    return messages


def set_redis(key: str, value: str, host: str = 'localhost', port: int = 6379) -> bool:
    """Set value in Redis"""
    try:
        import redis
    except ImportError:
        raise ConversionError("redis required: pip install redis")
    
    r = redis.Redis(host=host, port=port)
    r.set(key, value)
    return True


def get_redis(key: str, host: str = 'localhost', port: int = 6379) -> str:
    """Get value from Redis"""
    try:
        import redis
    except ImportError:
        raise ConversionError("redis required: pip install redis")
    
    r = redis.Redis(host=host, port=port)
    return r.get(key).decode() if r.get(key) else None


def publish_mqtt(topic: str, message: str, broker: str = "localhost", port: int = 1883) -> bool:
    """Publish message to MQTT topic"""
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        raise ConversionError("paho-mqtt required: pip install paho-mqtt")
    
    client = mqtt.Client()
    client.connect(broker, port, 60)
    client.publish(topic, message)
    client.disconnect()
    return True


# ============ CAD / GRAPHICS FORMATS ============

def dxf_to_svg(dxf_path: str, output_path: str = None) -> str:
    """Convert DXF to SVG"""
    try:
        import ezdxf
    except ImportError:
        raise ConversionError("ezdxf required: pip install ezdxf")
    
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    svg_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<svg xmlns="http://www.w3.org/2000/svg">']
    
    for entity in msp:
        if entity.dxftype() == 'LINE':
            svg_lines.append(f'<line x1="{entity.dxf.start.x}" y1="{entity.dxf.start.y}" x2="{entity.dxf.end.x}" y2="{entity.dxf.end.y}" stroke="black"/>')
    
    svg_lines.append('</svg>')
    
    svg_content = '\n'.join(svg_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(svg_content)
    
    return svg_content


# ============ EDI / HL7 / XBRL ============

def edi_to_json(edi_path: str) -> Dict:
    """Parse EDI to JSON"""
    try:
        import pyx12
    except ImportError:
        raise ConversionError("pyx12 required: pip install pyx12")
    
    with open(edi_path, 'r') as f:
        edi_data = f.read()
    
    segments = [s.split('*') for s in edi_data.split('~') if s.strip()]
    return {'segments': segments}


def hl7_to_dict(hl7_path: str) -> Dict:
    """Parse HL7 message to dictionary"""
    with open(hl7_path, 'r') as f:
        hl7_data = f.read()
    
    lines = hl7_data.split('\r')
    result = {'segments': []}
    
    for line in lines:
        if line.strip():
            fields = line.split('|')
            result['segments'].append({
                'type': fields[0] if fields else None,
                'fields': fields[1:] if len(fields) > 1 else []
            })
    
    return result


# ============ MATH / SYMBOLIC CONVERSIONS ============

def latex_to_image(latex: str, output_path: str = None) -> str:
    """Convert LaTeX equation to image"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ConversionError("Pillow required: pip install pillow")
    
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 30), latex, fill='black', font=font)
    
    if output_path is None:
        output_path = "equation.png"
    
    img.save(output_path)
    return output_path


def equation_to_latex(equation: str) -> str:
    """Convert equation string to LaTeX"""
    latex = equation.replace('^', '^{').replace('sqrt', '\\sqrt')
    
    if '{' in latex and '}' not in latex.replace('{}', ''):
        latex += '}'
    
    return latex


# ============ GEOSPATIAL / GIS CONVERSIONS ============

def geojson_to_dict(geojson_path: str) -> Dict:
    """Read GeoJSON file to dictionary"""
    with open(geojson_path, 'r') as f:
        return json.load(f)


def dict_to_geojson(data: Dict, output_path: str = None) -> str:
    """Convert dictionary to GeoJSON"""
    geojson_str = json.dumps(data, indent=2)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(geojson_str)
    
    return geojson_str


def coordinates_to_geojson(lat: float, lon: float, properties: Dict = None) -> Dict:
    """Convert coordinates to GeoJSON Point"""
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": properties or {}
    }
    return feature


def geojson_to_kml(geojson_path: str, output_path: str = None) -> str:
    """Convert GeoJSON to KML"""
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)
    
    kml_lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                '<kml xmlns="http://www.opengis.net/kml/2.2">',
                '<Document>']
    
    if geojson.get('type') == 'FeatureCollection':
        for feature in geojson.get('features', []):
            if feature.get('geometry', {}).get('type') == 'Point':
                coords = feature['geometry']['coordinates']
                name = feature.get('properties', {}).get('name', 'Unnamed')
                kml_lines.append(f'<Placemark><name>{name}</name><Point><coordinates>{coords[0]},{coords[1]}</coordinates></Point></Placemark>')
    
    kml_lines.extend(['</Document>', '</kml>'])
    kml_str = '\n'.join(kml_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(kml_str)
    
    return kml_str


def kml_to_geojson(kml_path: str) -> Dict:
    """Convert KML to GeoJSON"""
    import re
    
    with open(kml_path, 'r') as f:
        kml_content = f.read()
    
    features = []
    placemarks = re.findall(r'<Placemark>(.*?)</Placemark>', kml_content, re.DOTALL)
    
    for placemark in placemarks:
        name_match = re.search(r'<name>(.*?)</name>', placemark)
        coords_match = re.search(r'<coordinates>(.*?)</coordinates>', placemark)
        
        if coords_match:
            coords = coords_match.group(1).split(',')
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(coords[0]), float(coords[1])]
                },
                "properties": {
                    "name": name_match.group(1) if name_match else "Unnamed"
                }
            }
            features.append(feature)
    
    return {"type": "FeatureCollection", "features": features}


def shapefile_to_geojson(shp_path: str) -> Dict:
    """Convert Shapefile to GeoJSON"""
    try:
        import geopandas as gpd
    except ImportError:
        raise ConversionError("geopandas required: pip install geopandas")
    
    gdf = gpd.read_file(shp_path)
    return json.loads(gdf.to_json())


def wkt_to_geojson(wkt_string: str) -> Dict:
    """Convert WKT to GeoJSON"""
    try:
        from shapely import wkt
        from shapely.geometry import mapping
    except ImportError:
        raise ConversionError("shapely required: pip install shapely")
    
    geom = wkt.loads(wkt_string)
    return mapping(geom)


def geojson_to_wkt(geojson: Dict) -> str:
    """Convert GeoJSON to WKT"""
    try:
        from shapely.geometry import shape
    except ImportError:
        raise ConversionError("shapely required: pip install shapely")
    
    geom = shape(geojson)
    return geom.wkt


# ============ FONT CONVERSIONS ============

def ttf_to_base64(font_path: str) -> str:
    """Convert TTF font to base64"""
    with open(font_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def base64_to_ttf(base64_str: str, output_path: str) -> str:
    """Convert base64 to TTF font file"""
    with open(output_path, 'wb') as f:
        f.write(base64.b64decode(base64_str))
    return output_path


def list_font_info(font_path: str) -> Dict:
    """Get font information"""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        raise ConversionError("fonttools required: pip install fonttools")
    
    font = TTFont(font_path)
    return {
        'family_name': font['name'].getNameByID(1),
        'style_name': font['name'].getNameByID(2),
        'version': font['name'].getNameID(5),
        'glyph_count': len(font.getGlyphSet()),
    }


# ============ 3D MODEL CONVERSIONS ============

def obj_to_stl(obj_path: str, output_path: str = None) -> str:
    """Convert OBJ to STL"""
    try:
        import trimesh
    except ImportError:
        raise ConversionError("trimesh required: pip install trimesh")
    
    mesh = trimesh.load(obj_path)
    
    if output_path is None:
        output_path = obj_path.rsplit('.', 1)[0] + '.stl'
    
    mesh.export(output_path)
    return output_path


def stl_to_obj(stl_path: str, output_path: str = None) -> str:
    """Convert STL to OBJ"""
    try:
        import trimesh
    except ImportError:
        raise ConversionError("trimesh required: pip install trimesh")
    
    mesh = trimesh.load(stl_path)
    
    if output_path is None:
        output_path = stl_path.rsplit('.', 1)[0] + '.obj'
    
    mesh.export(output_path)
    return output_path


def gltf_to_obj(gltf_path: str, output_path: str = None) -> str:
    """Convert GLTF to OBJ"""
    try:
        import trimesh
    except ImportError:
        raise ConversionError("trimesh required: pip install trimesh")
    
    mesh = trimesh.load(gltf_path)
    
    if output_path is None:
        output_path = gltf_path.rsplit('.', 1)[0] + '.obj'
    
    mesh.export(output_path)
    return output_path


def mesh_to_stl(mesh_data: Dict, output_path: str = None) -> str:
    """Convert mesh data to STL"""
    try:
        import trimesh
    except ImportError:
        raise ConversionError("trimesh required: pip install trimesh")
    
    vertices = mesh_data.get('vertices', [])
    faces = mesh_data.get('faces', [])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    if output_path is None:
        output_path = "mesh.stl"
    
    mesh.export(output_path)
    return output_path


# ============ CALENDAR CONVERSIONS ============

def ics_to_dict(ics_path: str) -> Dict:
    """Parse ICS calendar file to dictionary"""
    with open(ics_path, 'r') as f:
        content = f.read()
    
    events = []
    current_event = {}
    in_event = False
    
    for line in content.split('\n'):
        if line.startswith('BEGIN:VEVENT'):
            in_event = True
            current_event = {}
        elif line.startswith('END:VEVENT'):
            in_event = False
            events.append(current_event)
        elif in_event:
            if ':' in line:
                key, value = line.split(':', 1)
                current_event[key] = value
    
    return {'events': events}


def dict_to_ics(events: List[Dict], output_path: str = None, calendar_name: str = "Calendar") -> str:
    """Convert dictionary to ICS calendar format"""
    ics_lines = [
        'BEGIN:VCALENDAR',
        'VERSION:2.0',
        'PRODID:-//Universal Converter//EN',
        f'X-WR-CALNAME:{calendar_name}'
    ]
    
    for event in events:
        ics_lines.append('BEGIN:VEVENT')
        if 'summary' in event:
            ics_lines.append(f"SUMMARY:{event['summary']}")
        if 'description' in event:
            ics_lines.append(f"DESCRIPTION:{event['description']}")
        if 'dtstart' in event:
            ics_lines.append(f"DTSTART:{event['dtstart']}")
        if 'dtend' in event:
            ics_lines.append(f"DTEND:{event['dtend']}")
        if 'location' in event:
            ics_lines.append(f"LOCATION:{event['location']}")
        ics_lines.append('END:VEVENT')
    
    ics_lines.append('END:VCALENDAR')
    ics_str = '\n'.join(ics_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(ics_str)
    
    return ics_str


# ============ EMAIL CONVERSIONS ============

def eml_to_text(eml_path: str) -> str:
    """Convert EML email to plain text"""
    try:
        from email import policy
        from email.parser import Parser
    except ImportError:
        raise ConversionError("Python email module required")
    
    with open(eml_path, 'r') as f:
        msg = Parser(policy=policy.default).parse(f)
    
    text = f"Subject: {msg['subject']}\n"
    text += f"From: {msg['from']}\n"
    text += f"To: {msg['to']}\n"
    text += f"Date: {msg['date']}\n\n"
    
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text += part.get_content()
                break
    else:
        text += msg.get_content()
    
    return text


def eml_to_dict(eml_path: str) -> Dict:
    """Convert EML email to dictionary"""
    try:
        from email import policy
        from email.parser import Parser
    except ImportError:
        raise ConversionError("Python email module required")
    
    with open(eml_path, 'r') as f:
        msg = Parser(policy=policy.default).parse(f)
    
    return {
        'subject': msg['subject'],
        'from': msg['from'],
        'to': msg['to'],
        'date': msg['date'],
        'body': msg.get_content() if not msg.is_multipart() else str(msg),
    }


def msg_to_text(msg_path: str) -> str:
    """Convert MSG email to plain text"""
    try:
        from extract_msg import Message
    except ImportError:
        raise ConversionError("extract-msg required: pip install extract-msg")
    
    msg = Message(msg_path)
    return f"Subject: {msg.subject}\nFrom: {msg.sender}\n\n{msg.body}"


# ============ FINANCE / TRADING CONVERSIONS ============

def csv_to_trading_view(csv_path: str, output_path: str = None) -> str:
    """Convert CSV to TradingView format"""
    try:
        import pandas as pd
    except ImportError:
        raise ConversionError("pandas required: pip install pandas")
    
    df = pd.read_csv(csv_path)
    
    tv_format = "datetime,open,high,low,close,volume\n"
    for _, row in df.iterrows():
        tv_format += f"{row.iloc[0]},{row.iloc[1]},{row.iloc[2]},{row.iloc[3]},{row.iloc[4]},{row.iloc[5]}\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(tv_format)
    
    return tv_format


def crypto_to_csv(crypto_data: List[Dict], output_path: str = None) -> str:
    """Convert crypto API data to CSV"""
    if not crypto_data:
        return ""
    
    keys = crypto_data[0].keys()
    lines = [','.join(keys)]
    
    for item in crypto_data:
        lines.append(','.join(str(item.get(k, '')) for k in keys))
    
    csv_str = '\n'.join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(csv_str)
    
    return csv_str


# ============ CONFIGURATION FILE CONVERSIONS ============

def env_to_dict(env_path: str) -> Dict:
    """Parse .env file to dictionary"""
    config = {}
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip().strip('"').strip("'")
    
    return config


def dict_to_env(data: Dict, output_path: str = None) -> str:
    """Convert dictionary to .env format"""
    lines = []
    for key, value in data.items():
        if ' ' in str(value):
            lines.append(f'{key}="{value}"')
        else:
            lines.append(f'{key}={value}')
    
    env_str = '\n'.join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(env_str)
    
    return env_str


def properties_to_dict(properties_path: str) -> Dict:
    """Parse Java properties file to dictionary"""
    config = {}
    
    with open(properties_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('!'):
                if '=' in line or ':' in line:
                    sep = '=' if '=' in line else ':'
                    key, value = line.split(sep, 1)
                    config[key.strip()] = value.strip()
    
    return config


def dict_to_properties(data: Dict, output_path: str = None) -> str:
    """Convert dictionary to Java properties format"""
    lines = []
    for key, value in data.items():
        lines.append(f"{key}={value}")
    
    props_str = '\n'.join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(props_str)
    
    return props_str


def json_to_env(json_path: str, output_path: str = None) -> str:
    """Convert JSON to .env format"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return dict_to_env(data, output_path)


def env_to_json(env_path: str, output_path: str = None) -> str:
    """Convert .env to JSON format"""
    data = env_to_dict(env_path)
    json_str = json.dumps(data, indent=2)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(json_str)
    
    return json_str


# ============ NETWORK / PACKET CAPTURE ============

def pcap_to_json(pcap_path: str) -> List[Dict]:
    """Convert PCAP file to JSON"""
    try:
        from scapy.all import rdpcap, IP, TCP, UDP
    except ImportError:
        raise ConversionError("scapy required: pip install scapy")
    
    packets = rdpcap(pcap_path)
    results = []
    
    for pkt in packets:
        if IP in pkt:
            packet_info = {
                'src': pkt[IP].src,
                'dst': pkt[IP].dst,
                'protocol': pkt[IP].proto,
                'len': pkt[IP].len
            }
            
            if TCP in pkt:
                packet_info['sport'] = pkt[TCP].sport
                packet_info['dport'] = pkt[TCP].dport
            elif UDP in pkt:
                packet_info['sport'] = pkt[UDP].sport
                packet_info['dport'] = pkt[UDP].dport
            
            results.append(packet_info)
    
    return results


# ============ PROTOCOL BUFFERS / FLATBUFFERS ============

def protobuf_to_dict(pb_path: str, proto_file: str = None) -> Dict:
    """Parse Protocol Buffer to dictionary"""
    try:
        from google.protobuf import descriptor_pb2
        from google.protobuf.internal.decoder import _DecodeVarint32
    except ImportError:
        raise ConversionError("protobuf required: pip install protobuf")
    
    with open(pb_path, 'rb') as f:
        data = f.read()
    
    return {"raw": base64.b64encode(data).decode()}


def dict_to_json_schema(data: Dict, output_path: str = None) -> str:
    """Generate JSON Schema from dictionary"""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {}
    }
    
    for key, value in data.items():
        if isinstance(value, bool):
            schema["properties"][key] = {"type": "boolean"}
        elif isinstance(value, int):
            schema["properties"][key] = {"type": "integer"}
        elif isinstance(value, float):
            schema["properties"][key] = {"type": "number"}
        elif isinstance(value, str):
            schema["properties"][key] = {"type": "string"}
        elif isinstance(value, list):
            schema["properties"][key] = {"type": "array"}
        elif isinstance(value, dict):
            schema["properties"][key] = {"type": "object"}
    
    json_str = json.dumps(schema, indent=2)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(json_str)
    
    return json_str


# ============ MAIN CLI ============

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Converter - Convert Anything to Anything')
    parser.add_argument('input', help='Input file, data, or directory')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('-f', '--from', dest='from_fmt', help='Input format')
    parser.add_argument('-t', '--to', dest='to_fmt', help='Output format')
    parser.add_argument('-l', '--list', action='store_true', help='List supported formats')
    parser.add_argument('-i', '--info', action='store_true', help='Show file info')
    parser.add_argument('--validate', action='store_true', help='Validate file format')
    parser.add_argument('--hash', choices=['md5', 'sha1', 'sha256', 'sha512'], help='File hash')
    parser.add_argument('--resize', action='store_true', help='Resize image')
    parser.add_argument('--width', type=int, help='Target width')
    parser.add_argument('--height', type=int, help='Target height')
    parser.add_argument('--scale', type=float, help='Scale factor (e.g., 0.5)')
    parser.add_argument('--batch', nargs='+', help='Batch convert')
    parser.add_argument('--output-dir', help='Output directory')
    
    args = parser.parse_args()
    
    if args.list:
        formats = get_formats()
        print("Supported Formats:")
        for cat, fmts in formats.items():
            print(f"\n{cat.upper()} ({len(fmts)}):")
            print("  " + ", ".join(fmts))
        return
    
    if args.info:
        print(json.dumps(file_info(args.input), indent=2))
        return
    
    if args.validate:
        print(json.dumps(validate_file(args.input), indent=2))
        return
    
    if args.hash:
        print(f"{args.hash}: {hash_file(args.input, args.hash)}")
        return
    
    if args.resize:
        result = resize_image(args.input, args.output, args.width, args.height, args.scale)
        print(f"Resized: {result}")
        return
    
    if args.batch:
        results = batch_convert(args.batch, args.output_dir, args.to_fmt or 'json')
        for r in results:
            status = "✓" if r['success'] else "✗"
            print(f"{status} {r['file']} → {r.get('output') or r.get('error')}")
        return
    
    try:
        result = convert_file(args.input, args.output, args.from_fmt, args.to_fmt)
        print(f"Converted: {result}")
    except ConversionError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
