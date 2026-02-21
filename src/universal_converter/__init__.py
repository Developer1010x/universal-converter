#!/usr/bin/env python3
"""
Universal Converter - Convert Anything to Anything
OS-neutral, minimal dependencies, comprehensive conversion tool
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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


REQUIRED_PACKAGES = {
    'pillow': 'Pillow',
    'pypdf2': 'PyPDF2',
    'reportlab': 'reportlab',
    'python-docx': 'docx',
    'pandas': 'pandas',
    'openpyxl': 'openpyxl',
    'xlrd': 'xlrd',
    'python-pptx': 'pptx',
    'markdown': 'markdown',
    'weasyprint': 'weasyprint',
    'odfpy': 'odf',
    'cairosvg': 'cairosvg',
    'pydub': 'pydub',
    'speechrecognition': 'SpeechRecognition',
    'moviepy': 'moviepy',
    'rarfile': 'rarfile',
    'pyzipper': 'pyzipper',
    'beautifulsoup4': 'bs4',
    'html2text': 'html2text',
    'requests': 'requests',
    'xmltodict': 'xmltodict',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'h5py': 'h5py',
    'pyarrow': 'pyarrow',
    'qrcode': 'qrcode',
    'pyzbar': 'pyzbar',
    'bcrypt': 'bcrypt',
    'argon2': 'argon2',
    'pyjwt': 'jwt',
    'ebooklib': 'ebooklib',
    'pint': 'pint',
    'forex-python': 'forex_python',
    'pytesseract': 'pytesseract',
    'pdf2image': 'pdf2image',
    'msgpack': 'msgpack',
    'sentence-transformers': 'sentence_transformers',
    'tiktoken': 'tiktoken',
    'openai-whisper': 'whisper',
    'gtts': 'gtts',
    'pyttsx3': 'pyttsx3',
    'boto3': 'boto3',
    'google-cloud-storage': 'google.cloud.storage',
    'azure-storage-blob': 'azure.storage.blob',
    'eth-account': 'eth_account',
    'web3': 'web3',
    'biopython': 'Bio',
    'ezdxf': 'ezdxf',
    'kafka-python': 'kafka',
    'redis': 'redis',
    'paho-mqtt': 'paho.mqtt.client',
    'pyx12': 'pyx12',
}


def ensure_dependencies():
    """Check and install missing dependencies"""
    import importlib
    
    missing = []
    for pkg, import_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.info(f"Installing missing dependencies: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to install some packages. Install manually: pip install {' '.join(missing)}")


def ensure_package(package_name: str):
    """Ensure a specific package is installed"""
    import importlib
    
    pkg_info = REQUIRED_PACKAGES.get(package_name.lower())
    if not pkg_info:
        pkg_info = package_name
    
    try:
        importlib.import_module(pkg_info)
        return True
    except ImportError:
        try:
            logger.info(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            return True
        except subprocess.CalledProcessError:
            return False


    # Auto-check dependencies on import
    try:
        ensure_dependencies()
    except Exception:
        pass


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
