"""Bioinformatics sequence and variant format converters.

Every pair below is implemented with the standard library. The binary formats
that used to be advertised here (BAM, CRAM, BCF, bigBed) need samtools/htslib
and had no code path at all, so they are gone rather than listed and broken.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import BaseConverter, ConversionResult, ConversionTask


class BioinformaticsConverter(BaseConverter):
    """Converter for FASTA, FASTQ, GenBank, BED and VCF text formats."""

    SUPPORTED_CONVERSIONS = {
        "fasta": ["fastq", "genbank", "json"],
        "fastq": ["fasta", "json"],
        "genbank": ["fasta", "json"],
        "bed": ["vcf", "json"],
        "vcf": ["json", "bed"],
    }
    PRIORITY = 30

    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()

        try:
            handlers = {
                ("fasta", "fastq"): self._fasta_to_fastq,
                ("fasta", "genbank"): self._fasta_to_genbank,
                ("fasta", "json"): self._sequences_to_json,
                ("fastq", "fasta"): self._fastq_to_fasta,
                ("fastq", "json"): self._sequences_to_json,
                ("genbank", "fasta"): self._genbank_to_fasta,
                ("genbank", "json"): self._sequences_to_json,
                ("vcf", "json"): self._vcf_to_json,
                ("vcf", "bed"): self._vcf_to_bed,
                ("bed", "vcf"): self._bed_to_vcf,
                ("bed", "json"): self._bed_to_json,
            }
            handler = handlers.get((source, target))
            if handler is None:
                return ConversionResult(
                    success=False, error=f"Unsupported: {source} -> {target}"
                )
            return handler(task)
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    # --- sequence parsing ------------------------------------------------

    @staticmethod
    def _parse_fasta(content: str) -> List[Tuple[str, str]]:
        """Yield ``(header, sequence)``; sequences may wrap over many lines."""
        records: List[Tuple[str, str]] = []
        header: Optional[str] = None
        chunks: List[str] = []
        for line in content.splitlines():
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:].strip(), []
            elif line.strip() and header is not None:
                chunks.append(line.strip())
        if header is not None:
            records.append((header, "".join(chunks)))
        return records

    @staticmethod
    def _parse_fastq(content: str) -> List[Tuple[str, str, str]]:
        """Yield ``(header, sequence, quality)`` from four-line FASTQ records."""
        lines = [line for line in content.splitlines() if line.strip()]
        records: List[Tuple[str, str, str]] = []
        for index in range(0, len(lines) - 3, 4):
            records.append(
                (lines[index].lstrip("@").strip(), lines[index + 1], lines[index + 3])
            )
        return records

    def _records(self, task: ConversionTask) -> List[Dict[str, Any]]:
        fmt = task.source_format.lower()
        content = Path(task.source_path).read_text(encoding="utf-8")
        if fmt == "fasta":
            return [
                {"id": h.split()[0] if h else "", "description": h, "sequence": s,
                 "length": len(s)}
                for h, s in self._parse_fasta(content)
            ]
        if fmt == "fastq":
            return [
                {"id": h.split()[0] if h else "", "description": h, "sequence": s,
                 "quality": q, "length": len(s)}
                for h, s, q in self._parse_fastq(content)
            ]
        header, sequence = self._read_genbank(content)
        return [{"id": header, "description": header, "sequence": sequence,
                 "length": len(sequence)}]

    def _sequences_to_json(self, task: ConversionTask) -> ConversionResult:
        records = self._records(task)
        Path(task.target_path).write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"records": len(records)},
        )

    # --- sequence conversions -------------------------------------------

    def _fasta_to_fastq(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        # FASTA carries no quality scores; 'I' is Phred 40 in Sanger encoding.
        quality_char = task.options.get("quality_char", "I")
        lines: List[str] = []
        records = self._parse_fasta(content)
        for header, sequence in records:
            lines += [f"@{header}", sequence, "+", quality_char * len(sequence)]

        Path(task.target_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"records": len(records), "quality_char": quality_char},
            warnings=["FASTA has no quality scores; a constant score was substituted"],
        )

    def _fastq_to_fasta(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        records = self._parse_fastq(content)
        lines: List[str] = []
        for header, sequence, _ in records:
            lines.append(f">{header}")
            lines.extend(self._wrap(sequence))

        Path(task.target_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"records": len(records)},
        )

    @staticmethod
    def _wrap(sequence: str, width: int = 60) -> Iterator[str]:
        for index in range(0, len(sequence), width):
            yield sequence[index : index + width]

    def _fasta_to_genbank(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        records = self._parse_fasta(content)
        if not records:
            return ConversionResult(success=False, error="no FASTA records found")

        header, sequence = records[0]
        locus = header.split()[0] if header else "UNKNOWN"
        genbank = (
            f"LOCUS       {locus:<16}{len(sequence)} bp    DNA     linear   UNK\n"
            f"DEFINITION  {header}\n"
            f"ACCESSION   {locus}\n"
            "FEATURES             Location/Qualifiers\n"
            f"     source          1..{len(sequence)}\n"
            '                     /organism="unknown"\n'
            "ORIGIN\n"
            f"{self._sequence_to_genbank(sequence)}\n"
            "//\n"
        )
        Path(task.target_path).write_text(genbank, encoding="utf-8")
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            warnings=(
                ["only the first FASTA record was converted"]
                if len(records) > 1
                else []
            ),
        )

    @staticmethod
    def _sequence_to_genbank(sequence: str) -> str:
        lines = []
        for index in range(0, len(sequence), 60):
            chunk = sequence[index : index + 60].lower()
            blocks = " ".join(chunk[i : i + 10] for i in range(0, len(chunk), 10))
            lines.append(f"{str(index + 1).rjust(9)} {blocks}")
        return "\n".join(lines)

    @staticmethod
    def _read_genbank(content: str) -> Tuple[str, str]:
        locus = re.search(r"LOCUS\s+(\S+)", content)
        if not locus:
            raise ValueError("invalid GenBank file: no LOCUS line")
        origin = re.search(r"^ORIGIN.*?$(.*?)(?:^//|\Z)", content, re.DOTALL | re.M)
        if not origin:
            raise ValueError("invalid GenBank file: no ORIGIN block")
        return locus.group(1), re.sub(r"[^A-Za-z]", "", origin.group(1)).upper()

    def _genbank_to_fasta(self, task: ConversionTask) -> ConversionResult:
        locus, sequence = self._read_genbank(
            Path(task.source_path).read_text(encoding="utf-8")
        )
        body = "\n".join(self._wrap(sequence))
        Path(task.target_path).write_text(f">{locus}\n{body}\n", encoding="utf-8")
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"length": len(sequence)},
        )

    # --- variants --------------------------------------------------------

    @staticmethod
    def _read_vcf(content: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for line in content.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            records.append(
                {
                    "chrom": parts[0],
                    "pos": int(parts[1]),
                    "id": parts[2],
                    "ref": parts[3],
                    "alt": parts[4].split(","),
                    "qual": float(parts[5]) if parts[5] not in (".", "") else None,
                    "filter": parts[6],
                    "info": BioinformaticsConverter._parse_info(parts[7]),
                }
            )
        return records

    @staticmethod
    def _parse_info(info: str) -> Dict[str, Any]:
        if info in (".", ""):
            return {}
        fields: Dict[str, Any] = {}
        for item in info.split(";"):
            if not item:
                continue
            key, _, value = item.partition("=")
            fields[key] = value if value else True
        return fields

    def _vcf_to_json(self, task: ConversionTask) -> ConversionResult:
        records = self._read_vcf(Path(task.source_path).read_text(encoding="utf-8"))
        Path(task.target_path).write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"variants": len(records)},
        )

    def _vcf_to_bed(self, task: ConversionTask) -> ConversionResult:
        records = self._read_vcf(Path(task.source_path).read_text(encoding="utf-8"))
        lines = [
            # BED is 0-based half-open; VCF is 1-based inclusive.
            "\t".join(
                [
                    r["chrom"],
                    str(r["pos"] - 1),
                    str(r["pos"] - 1 + len(r["ref"])),
                    r["id"] if r["id"] != "." else f"{r['chrom']}_{r['pos']}",
                ]
            )
            for r in records
        ]
        Path(task.target_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"variants": len(records)},
        )

    @staticmethod
    def _read_bed(content: str) -> List[Dict[str, Any]]:
        features: List[Dict[str, Any]] = []
        for line in content.splitlines():
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            features.append(
                {
                    "chrom": parts[0],
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "name": parts[3] if len(parts) > 3 else ".",
                    "score": parts[4] if len(parts) > 4 else None,
                    "strand": parts[5] if len(parts) > 5 else None,
                }
            )
        return features

    def _bed_to_json(self, task: ConversionTask) -> ConversionResult:
        features = self._read_bed(Path(task.source_path).read_text(encoding="utf-8"))
        Path(task.target_path).write_text(
            json.dumps(features, indent=2), encoding="utf-8"
        )
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"features": len(features)},
        )

    def _bed_to_vcf(self, task: ConversionTask) -> ConversionResult:
        features = self._read_bed(Path(task.source_path).read_text(encoding="utf-8"))
        lines = [
            "##fileformat=VCFv4.2",
            '##ALT=<ID=NON_REF,Description="Interval imported from BED; '
            'no alternate allele is known">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        ]
        for feature in features:
            # BED start is 0-based, VCF POS is 1-based. <NON_REF> is the
            # symbolic allele for "no ALT known" -- the old code emitted the
            # literal string <\.\>, which is not valid VCF.
            end = feature["end"]
            lines.append(
                "\t".join(
                    [
                        feature["chrom"],
                        str(feature["start"] + 1),
                        feature["name"] or ".",
                        "N",
                        "<NON_REF>",
                        ".",
                        ".",
                        f"END={end}",
                    ]
                )
            )
        Path(task.target_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"features": len(features)},
        )

    def convert_format(
        self, path: str, to_format: str, output: Optional[str] = None
    ) -> ConversionResult:
        """Convert a bioinformatics file to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)
