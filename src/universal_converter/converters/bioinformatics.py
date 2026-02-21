"""Bioinformatics file format conversion module"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from . import BaseConverter, ConversionTask, ConversionResult


class BioinformaticsConverter(BaseConverter):
    """Converter for bioinformatics file formats"""
    
    SUPPORTED_CONVERSIONS = {
        "fasta": ["fastq", "genbank", "fasta"],
        "fastq": ["fasta", "fastq"],
        "genbank": ["fasta", "json"],
        "bed": ["bigbed", "vcf"],
        "vcf": ["bed", "json"],
", "bcf        "bcf": ["vcf"],
        "sam": ["bam", "cram"],
        "bam": ["sam", "cram"],
    }
    PRIORITY = 30
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()
        
        try:
            if source == "fasta" and target == "fastq":
                return self._fasta_to_fastq(task)
            elif source == "fasta" and target == "genbank":
                return self._fasta_to_genbank(task)
            elif source == "fastq" and target == "fasta":
                return self._fastq_to_fasta(task)
            elif source == "genbank" and target == "fasta":
                return self._genbank_to_fasta(task)
            elif source == "vcf" and target == "json":
                return self._vcf_to_json(task)
            elif source == "bed" and target == "vcf":
                return self._bed_to_vcf(task)
            
            return ConversionResult(success=False, error=f"Unsupported: {source} → {target}")
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def _fasta_to_fastq(self, task: ConversionTask) -> ConversionResult:
        content = task.source_path.read_text()
        lines = content.strip().split('\n')
        
        fastq_lines = []
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                header = lines[i]
                sequence = lines[i + 1]
                quality = 'I' * len(sequence)
                fastq_lines.append(header.replace('>', '@'))
                fastq_lines.append(sequence)
                fastq_lines.append('+')
                fastq_lines.append(quality)
        
        task.target_path.write_text('\n'.join(fastq_lines))
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _fasta_to_genbank(self, task: ConversionTask) -> ConversionResult:
        content = task.source_path.read_text()
        lines = content.strip().split('\n')
        
        header = lines[0].replace('>', '')
        sequence = ''.join(lines[1:])
        
        genbank = f"""LOCUS       {header.split()[0]}  {len(sequence)} bp  DNA  linear  UNK
DEFINITION  {header}
ACCESSION   {header.split()[0]}
FEATURES             Location/Qualifiers
     source          1..{len(sequence)}
                     /organism="unknown"
ORIGIN
{self._sequence_to_genbank(sequence)}
//
"""
        task.target_path.write_text(genbank)
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _sequence_to_genbank(self, seq: str) -> str:
        lines = []
        for i in range(0, len(seq), 60):
            chunk = seq[i:i+60]
            line_num = str(i + 1).rjust(9)
            formatted = ' '.join([chunk[j:j+10] for j in range(0, len(chunk), 10)])
            lines.append(f"{line_num}  {formatted}")
        return '\n'.join(lines)
    
    def _fastq_to_fasta(self, task: ConversionTask) -> ConversionResult:
        content = task.source_path.read_text()
        lines = content.strip().split('\n')
        
        fasta_lines = []
        for i in range(0, len(lines), 4):
            if i + 1 < len(lines):
                header = lines[i]
                sequence = lines[i + 1]
                fasta_lines.append(header.replace('@', '>'))
                fasta_lines.append(sequence)
        
        task.target_path.write_text('\n'.join(fasta_lines))
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _genbank_to_fasta(self, task: ConversionTask) -> ConversionResult:
        content = task.source_path.read_text()
        
        import re
        match = re.search(r'LOCUS\s+(\S+)', content)
        if not match:
            return ConversionResult(success=False, error="Invalid GenBank file")
        
        seq_match = re.search(r'ORIGIN\s+(.+?)(?://|$)', content, re.DOTALL)
        if not seq_match:
            return ConversionResult(success=False, error="No sequence found")
        
        seq = re.sub(r'[\d\s/]', '', seq_match.group(1)).upper()
        
        fasta = f">{match.group(1)}\n{self._wrap_sequence(seq)}\n"
        task.target_path.write_text(fasta)
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _wrap_sequence(self, seq: str, width: int = 80) -> str:
        return '\n'.join([seq[i:i+width] for i in range(0, len(seq), width)])
    
    def _vcf_to_json(self, task: ConversionTask) -> ConversionResult:
        import json
        
        content = task.source_path.read_text()
        lines = [l for l in content.split('\n') if l and not l.startswith('#')]
        
        records = []
        for line in lines:
            parts = line.split('\t')
            record = {
                "chrom": parts[0],
                "pos": int(parts[1]),
                "id": parts[2],
                "ref": parts[3],
                "alt": parts[4].split(','),
                "qual": float(parts[5]) if parts[5] != '.' else None,
                "filter": parts[6],
                "info": self._parse_info(parts[7])
            }
            records.append(record)
        
        task.target_path.write_text(json.dumps(records, indent=2))
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def _parse_info(self, info: str) -> Dict[str, str]:
        if info == '.':
            return {}
        return dict(item.split('=') if '=' in item else (item, True) 
                   for item in info.split(';'))
    
    def _bed_to_vcf(self, task: ConversionTask) -> ConversionResult:
        content = task.source_path.read_text()
        lines = [l for l in content.split('\n') if l and not l.startswith('#')]
        
        vcf_header = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        vcf_lines = [vcf_header]
        
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 3:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                name = parts[3] if len(parts) > 3 else '.'
                vcf_lines.append(f"{chrom}\t{start + 1}\t{name}\tN\t<\.\>\t.\t.\t.\n")
        
        task.target_path.write_text(''.join(vcf_lines))
        return ConversionResult(success=True, output_path=str(task.target_path))
    
    def convert_format(self, path: str, to_format: str, output: Optional[str] = None) -> ConversionResult:
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format
        )
        return self.convert(task)


class PDBConverter(BaseConverter):
    """Converter for protein database files"""
    
    SUPPORTED_CONVERSIONS = {
        "pdb": ["mmtf", "cif", "json"],
        "mmtf": ["pdb", "cif"],
        "cif": ["pdb", "mmtf"],
    }
    PRIORITY = 35
    
    def convert(self, task: ConversionTask) -> ConversionResult:
        return ConversionResult(
            success=False,
            error="Biopython required. Install: pip install universal-converter[bioinformatics]"
        )
