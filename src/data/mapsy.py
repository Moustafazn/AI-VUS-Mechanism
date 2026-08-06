"""
SpliceVarMech — MaPSy (Massively Parallel Splicing Assay) Parser

Independent cross-dataset evaluation using experimentally validated
exonic splice mutation (ESM) variants from Soemedi et al., Nature Genetics 2017.

MaPSy Dataset:
  - 231 exonic variants tested in a massively parallel splicing assay
  - Each variant classified as ESM (Exonic Splice Mutation) or non-ESM
  - ESM = variant experimentally shown to disrupt normal splicing
  - 8 ESM-positive variants, 223 ESM-negative variants
  - Variants identified by dbSNP rs IDs
  - Completely independent from our training data

Label definition (from Soemedi et al. 2017):
    ESM = 1 → exonic splice mutation / splice-disrupting (label=1)
    ESM = 0 → no splice effect (label=0)

Scientific context:
    MaPSy tests variants in a minigene reporter assay. Each variant is
    introduced into a three-exon minigene construct, and the ratio of
    exon inclusion to skipping is measured. Variants that significantly
    reduce exon inclusion are classified as ESM.

    This is complementary to BRCA1 SGE (which tests one gene exhaustively)
    — MaPSy tests variants across multiple genes but at smaller scale.

Data source:
    Soemedi et al., "Pathogenic variants that alter protein code often
    disrupt splicing", Nature Genetics, 2017.
    DOI: 10.1038/ng.3837
    Supplementary Table 1

Download:
    Automatically downloaded by the pipeline, or manually:
    curl -L -o data/external/mapsy_soemedi2017.xlsx \\
        "https://static-content.springer.com/esm/art%3A10.1038%2Fng.3837/MediaObjects/41588_2017_BFng3837_MOESM2_ESM.xlsx"

Usage:
    from src.data.mapsy import load_mapsy_variants
    variants = load_mapsy_variants()
    print(f"ESM: {sum(1 for v in variants if v.label == 1)}")
    print(f"Non-ESM: {sum(1 for v in variants if v.label == 0)}")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MAPSY_PATH = "data/external/mapsy_soemedi2017.xlsx"


@dataclass
class MaPSyVariant:
    """A single MaPSy experimentally tested exonic variant."""
    dbsnp_id: str                  # rs ID (e.g., rs3207775)
    ref_allele: str
    alt_allele: str
    esm: int                       # 1=exonic splice mutation, 0=no effect
    label: int                     # Same as esm: 1=splice-disrupting, 0=normal


def load_mapsy_variants(
    path: str = MAPSY_PATH,
    verbose: bool = True,
) -> list[MaPSyVariant]:
    """
    Load MaPSy variants from Soemedi et al. 2017 supplementary data.

    Args:
        path: Path to the Excel file
        verbose: Print summary

    Returns:
        List of MaPSyVariant with experimental ESM labels
    """
    data_path = Path(path)

    if not data_path.exists():
        if verbose:
            print(f"\n  ⚠️  MaPSy data not found at {path}")
            print(f"  Download with:")
            print(f'  curl -L -o data/external/mapsy_soemedi2017.xlsx \\')
            print(f'    "https://static-content.springer.com/esm/'
                  f'art%3A10.1038%2Fng.3837/MediaObjects/'
                  f'41588_2017_BFng3837_MOESM2_ESM.xlsx"')
        return []

    try:
        import openpyxl
    except ImportError:
        if verbose:
            print("  ⚠️  openpyxl required: pip install openpyxl")
        return []

    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb['Sheet1']

    # Row 1 is title, Row 2 has headers, data starts from Row 3
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    headers = list(rows[0])
    data_rows = rows[1:]
    wb.close()

    # Build column index
    ci = {str(h).strip(): i for i, h in enumerate(headers) if h is not None}

    variants = []
    for row in data_rows:
        if not row[0]:  # Skip empty rows
            continue

        dbsnp = str(row[ci['dbSNP']]) if row[ci['dbSNP']] else ''
        ref = str(row[ci['ref']]) if row[ci['ref']] else ''
        alt = str(row[ci['alt']]) if row[ci['alt']] else ''

        # ESM: 0 or 1
        try:
            esm = int(row[ci['ESM']])
        except (ValueError, TypeError):
            continue

        variants.append(MaPSyVariant(
            dbsnp_id=dbsnp,
            ref_allele=ref,
            alt_allele=alt,
            esm=esm,
            label=esm,  # ESM=1 means splice-disrupting
        ))

    if verbose:
        _print_mapsy_summary(variants)

    return variants


def _print_mapsy_summary(variants: list[MaPSyVariant]) -> None:
    """Print summary of MaPSy dataset."""
    n_total = len(variants)
    n_esm = sum(1 for v in variants if v.label == 1)
    n_normal = n_total - n_esm

    print(f"\n  MaPSy Dataset (Soemedi et al., Nature Genetics 2017):")
    print(f"    Total variants: {n_total}")
    print(f"    ESM (label=1): {n_esm} ({n_esm/max(n_total,1):.1%})")
    print(f"    Non-ESM (label=0): {n_normal} ({n_normal/max(n_total,1):.1%})")
    print(f"    Assay: Massively parallel minigene splicing reporter")
    print(f"    Ground truth: Exon inclusion ratio measurement")


def mapsy_to_causal_features(
    mapsy_variants: list[MaPSyVariant],
    verbose: bool = True,
) -> list:
    """
    Convert MaPSy variants to CausalFeatures for cross-dataset evaluation.

    MaPSy variants are exonic (position=0) and lack tool scores,
    but have gold-standard experimental ground truth.
    """
    from src.causal.dag import CausalFeatures

    features = []
    for v in mapsy_variants:
        feat = CausalFeatures(
            variant_name=f"MaPSy:{v.dbsnp_id}",
            position=0,                 # Exonic variants
            splice_strength=None,
            ese_ess_score=None,
            conservation=None,
            ise_iss_score=None,
            splice_ai=None,
            squirls=None,
            mmsplice=None,
            cadd_splice=None,
            all_scores={},
            diffusion_aberrant_fraction=None,
            label=v.label,
            variant_type="Exonic",
            donor_or_acceptor="Unknown",
        )
        features.append(feat)

    if verbose:
        n_pos = sum(1 for f in features if f.label == 1)
        n_neg = sum(1 for f in features if f.label == 0)
        print(f"\n  MaPSy → CausalFeatures: "
              f"{n_pos} ESM + {n_neg} non-ESM = {len(features)} total")

    return features


if __name__ == "__main__":
    print("=" * 70)
    print("MaPSy Dataset (Soemedi et al., Nature Genetics 2017)")
    print("=" * 70)
    variants = load_mapsy_variants(verbose=True)
