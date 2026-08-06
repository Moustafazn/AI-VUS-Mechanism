# SpliceVarMech

**A Causal Generative Framework for Mechanistic Interpretation of Non-Canonical Splicing Variants in Male Infertility**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![PyMC 5.x](https://img.shields.io/badge/PyMC-5.x-orange.svg)]()
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()

---

## Overview

SpliceVarMech is the first computational framework that combines **discrete sequence diffusion models** with **Bayesian causal inference** for mechanistic interpretation of non-canonical splicing variants (NCSVs). Unlike existing tools that output scores, SpliceVarMech generates the **predicted aberrant mRNA sequence**, identifies the **causal mechanism**, and provides **calibrated uncertainty** — delivering a complete clinical report from variant to ACMG classification.

### Clinical Case

A hemizygous **TEX11 c.1156+16G>T** variant (position +16 in intron) in a male with non-obstructive azoospermia — classified as a VUS because no existing tool can confidently predict whether this deep intronic variant disrupts splicing.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  MODULE 1: Biological Diffusion Model (D3PM)                │
│  Dual-stream encoder: WT vs MUT cross-attention comparison  │
│  → Variant highlight (Gaussian σ=3bp) + substitution embed  │
│  → Multi-scale CNN (local/regional/structural spliceosome)  │
│  → Contrastive embedding distance (primary disruption metric│
│  → D3PM reverse sampling → mechanism classification         │
│  → EMA-smoothed weights for stable inference                │
├─────────────────────────────────────────────────────────────┤
│  MODULE 2: Bayesian Causal Inference (PyMC + MCMC)          │
│  Takes DIFFUSION FEATURES as primary evidence:              │
│    contrastive_distance + aberrant_fraction + mechanism     │
│    + position + variant_type → P(disruption)                │
│  → No splice tool scores in the Bayesian model              │
│  → Full MCMC posterior with 95% credible interval           │
├─────────────────────────────────────────────────────────────┤
│  MODULE 3: Explainable AI                                   │
│  Cross-attention attribution, causal paths, ACMG mapping    │
│  → Clinical report: mechanism + uncertainty + classification │
└─────────────────────────────────────────────────────────────┘
```

### End-to-End Pipeline Flow

```
  VARIANT INPUT (e.g., TEX11 c.1156+16G>T — held-out test case)
       │
       ▼
  ┌─ MODULE 1: Diffusion Model ──────────────────────────┐
  │  1. Construct WT + mutant pre-mRNA contexts          │
  │  2. Generate N mRNA samples from mutant context      │
  │     (D3PM reverse process with self-conditioning)    │
  │  3. Classify each sample via NW alignment to WT      │
  │  4. Compute disruption_score (log-likelihood ratio)  │
  │  5. Compute aberrant_fraction & mechanism probs      │
  └──────────────────────────────────────────────────────┘
       │  aberrant_fraction, mechanism, disruption_score
       ▼
  ┌─ MODULE 2: Bayesian Inference ───────────────────────┐
  │  MCMC posterior using diffusion outputs as evidence:  │
  │    P(disruption | aberrant_frac, mechanism, position) │
  │  → Posterior mean + 95% credible interval             │
  └──────────────────────────────────────────────────────┘
       │  P(disruption) = 0.87 [0.72, 0.95]
       ▼
  ┌─ MODULE 3: Clinical Report ──────────────────────────┐
  │  • Mechanism: exon skipping → frameshift → NMD       │
  │  • ACMG criteria: PP3_Strong + PS3_Moderate + ...    │
  │  • Classification: Likely Pathogenic                  │
  │  • Recommendation: Reclassify VUS → LP               │
  └──────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/Moustafazn/AI-VUS-Mechanism.git
cd AI-VUS-Mechanism

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### External Data Setup

The pipeline uses multiple external data sources for pre-training, training augmentation, and cross-dataset evaluation.

```bash
# 1. GENCODE v44 — real splice junctions for diffusion model pre-training
wget -P data/external/ https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
gunzip data/external/gencode.v44.annotation.gtf.gz
wget -P data/external/ https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz
gunzip data/external/GRCh38.primary_assembly.genome.fa.gz

# 2. gnomAD v4.1 — benign intronic negatives for training augmentation
#    Uses gnomAD GraphQL API — no large VCF downloads needed (~2-5 minutes)
python scripts/fetch_gnomad_api.py

# 3. MFASS — 27,733 experimentally tested splice variants (cross-dataset eval)
curl -L -o data/external/mfass_snv_data_clean.txt \
  "https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/snv_data_clean.txt"

# 4. ClinVar — clinically classified splice variants (training augmentation)
wget -P data/external/ https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz

# 5. BRCA1 SGE — saturation genome editing (cross-dataset evaluation)
curl -L -o data/external/brca1_sge_findlay2018.xlsx \
  "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0461-z/MediaObjects/41586_2018_461_MOESM3_ESM.xlsx"

# 6. MaPSy — massively parallel splicing assay (cross-dataset evaluation)
curl -L -o data/external/mapsy_soemedi2017.xlsx \
  "https://static-content.springer.com/esm/art%3A10.1038%2Fng.3837/MediaObjects/41588_2017_BFng3837_MOESM2_ESM.xlsx"
```

> **Note:** GENCODE and gnomAD are auto-detected. When present, the pipeline uses real data. When absent, it shows download instructions.

### Tissue-Conditioned Generation

The model supports tissue-specific splice prediction via a learned tissue embedding. Set the default tissue in `config.yaml`:

```yaml
tissue:
  default: "testis"  # Options: universal, testis, brain, liver, heart, muscle, blood, kidney, lung, ovary
```

During inference, specify tissue type:
```python
from src.diffusion.sampling import SpliceSampler
sampler = SpliceSampler(model)
samples = sampler.generate_samples(context, tissue="testis")
```

### Run Full Pipeline (Production)

```bash
python main.py --all            # Run all phases with the required settings
```

### Run Individual Phases

```bash
python main.py --phase 1        # Parse primary dataset (2,404 variants)
python main.py --phase 2        # Splice tool coverage analysis
python main.py --phase 3        # Bayesian model diagnostics
python main.py --phase 4        # Training (production: 20 epochs pre-train, 30 fine-tune)
python main.py --phase 5        # TEX11 prediction + clinical report (full MCMC)
python main.py --baselines      # 17-tool baseline evaluation (AUROC/AUPRC)
python main.py --spliceai       # SpliceAI head-to-head evaluation
python main.py --loo            # Leave-one-out cross-validation
python main.py --ablation       # Run ablation studies (actual execution)
python main.py --eval           # Comprehensive evaluation (leakage, calibration, cross-dataset)
python main.py --ablation --id 2  # Run specific ablation (#2: Bayesian only)
python main.py --xai            # Explainability analysis (cross-attention)
python main.py --benchmark      # SOTA benchmarking (2019-2026 literature)
```

---

## Training

### Defaults

Running `python main.py --phase 4` trains the full model with the required settings:

| Setting | Value |
|---------|-------|
| **Model** | d_model=256, n_heads=8, n_layers=6, d_ff=1024, Pre-LayerNorm |
| **Diffusion** | 100 timesteps, cosine schedule, self-conditioning (Sahoo et al. 2024) |
| **Sampling** | Proper D3PM reverse transitions, vectorized, non-nucleotide masking |
| **Parameters** | ~9.2M trainable |
| **Pre-training** | 100K GENCODE splice junctions (or synthetic fallback), 10 epochs |
| **Fine-tuning** | Gold-standard + Study 6 + augmentation, 20 epochs |
| **Regularization** | EMA (decay=0.9999), validation split (15%), early stopping (patience=10) |
| **Scheduler** | Linear warmup (100 steps) → cosine annealing |
| **Device** | Auto-detected (CUDA → MPS → CPU) |

### GENCODE Pre-training (Recommended)

For best results, pre-train on real human splice junctions from GENCODE:

1. Download the files:
   ```bash
   # GENCODE v44 annotation
   wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
   gunzip gencode.v44.annotation.gtf.gz

   # GRCh38 reference genome
   wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz
   gunzip GRCh38.primary_assembly.genome.fa.gz
   ```

2. Place them in `data/external/`:
   ```
   data/external/gencode.v44.annotation.gtf
   data/external/GRCh38.primary_assembly.genome.fa
   ```

3. Install FASTA access library:
   ```bash
   pip install pysam    # or: pip install pyfaidx
   ```

4. Run training — GENCODE is **auto-detected**:
   ```bash
   python main.py --phase 4
   ```

The trainer automatically searches `data/external/` for GENCODE GTF and reference FASTA files. When found, it extracts up to 100,000 real exon-intron-exon splice junctions for pre-training instead of synthetic data.

### Checkpoints

Trained models are saved to `experiments/checkpoints/splice_diffusion_model.pt` and automatically used by Phase 5 (prediction) and ablation studies.

---

## Ablation Studies

Running `python main.py --ablation` **executes** ablation experiments:

| # | Ablation | What's Removed | Tests |
|---|----------|----------------|-------|
| 1 | Diffusion Only | Bayesian causal model | Does causal reasoning improve over generation alone? |
| 2 | Bayesian Only | Diffusion model | Does the diffusion model add value over tool scores? |
| 3 | No External Data | Study 6 variants | Does external data augmentation help? |
| 4 | **No Pre-training** | GENCODE Stage 1 (252K junctions) | Does pre-training on normal splicing help? |
| 5 | No Class Balancing | Balanced weights | How critical is class balancing? |
| 6 | No Shrinkage Prior | Hierarchical prior | Does regularization prevent overfitting? |

```bash
python main.py --ablation           # Run all ablations
python main.py --ablation --id 4    # Without GENCODE pre-training (pretrained vs scratch)
```


## Evaluation Suite

Running `python main.py --eval` executes a comprehensive evaluation addressing top-tier publication standards:

| Metric | What It Tests | Status |
|--------|--------------|--------|
| **Formal Leakage Analysis** | Verify no information leakage between features and labels | ✅ |
| **Calibration (ECE)** | Expected Calibration Error + reliability diagrams | ✅ |
| **Per-Mechanism Metrics** | Performance stratified by variant type (Mis/Intron/Syn) | ✅ |
| **Cold-Gene Evaluation** | Leave-One-Gene-Out CV for unseen gene generalization | ✅ |
| **Feature-Group Ablation** | Isolate contribution of each feature category | ✅ |
| **Cross-Dataset Testing** | Train on primary → test on Study 6/Study 4 (independent cohorts) | ✅ |
| **XAI Stability** | Rank correlation of attributions across seeds/bootstraps | ✅ |
| **Component Training** | Document frozen vs fine-tuned components | ✅ |

### Cross-Dataset Generalization

| Train Set | Test Set | Type |
|-----------|----------|------|
| S7+S2 (primary) | S7+S2 (LOO-CV) | In-distribution |
| S7+S2 (primary) | Study 6 genes | Cross-cohort |
| S7+S2 (primary) | Study 4 TESE outcomes | Cross-outcome |
| S7+S2 + ClinVar | BRCA1 SGE (3,644 variants) | **Cross-dataset (independent gene)** |
| S7+S2 + ClinVar | MaPSy (231 variants) | **Cross-dataset (independent assay)** |

### External Data Sources

| Dataset | Variants | Source | Use |
|---------|----------|--------|-----|
| **GENCODE v44** | 252K splice junctions | GENCODE | Pre-training (Stage 1) ✅ |
| **gnomAD v4.1** | ~1,000+ benign intronic (AF>1%) | Google Cloud | Training augmentation (benign negatives) ✅ |
| **ClinVar** | 394,686 splice variants | NCBI variant_summary.txt.gz | Training augmentation ✅ |
| **MFASS** | 27,733 variants (13,875 near-canonical ±3-20) | Cheung et al., Mol Cell 2019 | Cross-dataset evaluation ✅ |
| **BRCA1 SGE** | 3,644 variants (823 LOF + 2,821 FUNC) | Findlay et al., Nature 2018 | Cross-dataset evaluation ✅ |
| **MaPSy** | 231 variants (8 ESM + 223 non-ESM) | Soemedi et al., Nat Gen 2017 | Cross-dataset evaluation ✅ |
| **GTEx v8** | 55,374 genes × 54 tissues | GTEx Portal | Tissue-specific expression ✅ |

---

## Dataset

### Primary Data

| Table | Content | Records | Role |
|-------|---------|---------|------|
| **S1** | Curated pathogenic variants | 2,404 × 63 | Feature matrix (17 splice tool scores) |
| **S2** | Negative controls (experimentally normal) | 14 usable | Gold-standard negatives |
| **S7** | Validated NCSVs with aberrant mRNA | 40 variants | Gold-standard positives + diffusion targets |
| **S3** | Patient-level variants | 58 × 95 | Biological context |
| **S4** | Clinical semen analysis | 13 patients | Phenotype correlation |
| **S5** | Extended variant landscape | 6,310 × 101 | Broader context |

**Source:** *"Mapping the Non-Canonical Splicing Variants"*, Advanced Science, 2024.

### External Data (Literature Integration)

| Study | Data | Examples | Use |
|-------|------|----------|-----|
| Study 6 (Splice defects in infertility) | 341 splice variants + SpliceAI/CADD | 183 positives added | Training augmentation |
| Study 4 (TESE outcomes, n=571) | 326 variants + TESE outcome | 50 negatives added | Evaluation + weak negatives |
| RBP Table 1 (RNA-binding proteins) | 8 TEX11 + 11 splice-affecting variants | Hardcoded | XAI validation |

**Combined training data:** 284 examples (before augmentation), ~1,900+ with augmentation.

---

## Project Structure

```
AI-VUS-Mechanism/
├── main.py                            # Unified CLI entry point (all phases)
├── config.yaml                        # Central configuration (model, training, data paths)
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
│
├── data/
│   ├── raw/ADVS-13-e15512-s001.xlsx   # Primary dataset
│   ├── external/                      # GENCODE, gnomAD, MFASS, BRCA1 SGE, MaPSy, ClinVar
│   └── processed/                     # Parsed outputs
│
├── scripts/
│   └── fetch_gnomad_api.py            # gnomAD v4.1 GraphQL API fetcher (benign negatives)
│                                      #   Exponential backoff, 190 genes, ~2K variants
│
├── src/
│   ├── config.py                      # Configuration loader (reads config.yaml)
│   ├── data/
│   │   ├── parser.py                  # Primary dataset parser (S1/S2/S3-S8)
│   │   ├── external_parser.py         # Study 6 + Study 4 literature data parser
│   │   ├── clinvar.py                 # ClinVar splice variant parser + augmented training
│   │   ├── gnomad.py                  # gnomAD v4.1 benign negatives loader
│   │   ├── mfass.py                   # MFASS splice assay (27,733 variants, Cheung 2019)
│   │   ├── brca1_sge.py              # BRCA1 SGE (3,644 variants, Findlay Nature 2018)
│   │   ├── mapsy.py                   # MaPSy (231 variants, Soemedi Nat Gen 2017)
│   │   ├── hg38_context.py           # GRCh38 genomic context extraction (real exon-intron)
│   │   └── gtex.py                    # GTEx v8 tissue-specific expression mapper
│   ├── features/
│   │   └── splice_scores.py           # 16-tool splice score analysis + coverage
│   ├── causal/
│   │   ├── dag.py                     # Bayesian causal model (diffusion-integrated SCM)
│   │   ├── diagnostics.py            # Class imbalance investigation
│   │   └── loo_cv.py                  # Leave-one-out cross-validation
│   ├── diffusion/
│   │   ├── model.py                   # D3PM architecture (~9.2M params, Pre-LayerNorm)
│   │   ├── training.py                # Pre-train + fine-tune pipeline (GENCODE auto-detect)
│   │   └── sampling.py               # D3PM reverse sampling + mechanism classification
│   ├── pipeline/
│   │   └── predict.py                 # End-to-end TEX11 prediction + clinical report
│   ├── xai/
│   │   └── attribution.py            # Cross-attention attribution + causal paths
│   └── baselines/
│       ├── tool_evaluation.py         # 17-tool AUROC/AUPRC evaluation
│       ├── spliceai_evaluation.py     # SpliceAI head-to-head (eval only)
│       ├── ablation.py                # 6 ablation experiments (actual execution)
│       ├── evaluation_metrics.py      # Comprehensive eval (leakage, calibration, cross-dataset)
│       └── benchmark.py              # SOTA benchmarking (2019-2026 literature)
│
├── notebooks/
│   ├── base_proposal.md               # Full project proposal
│   └── model_documentation.md         # Comprehensive model documentation
│
├── tests/
│   └── test_parser.py                 # 29 validation tests
│
├── experiments/                       # Training checkpoints + ablation outputs
├── figures/                           # Generated figures
└── paper/                             # Manuscript drafts
```

---

## Key Results

### Why Diffusion > Tool Scores

Tool scores alone cannot reliably separate true NCSVs from false positives — the negatives were selected *because* tools flagged them (selection bias). The diffusion model provides **sequence-level evidence** that goes beyond aggregate scores by actually generating the predicted mRNA and classifying the splice mechanism.

### TEX11 Validation (Held-Out Test Case)

TEX11 c.1156+16G>T is **not in the training data** — it is the clinical case study used exclusively for testing. Independently confirmed as causative by:
- **Study 4**: 4/4 TEX11 variants → TESE-negative (no sperm)
- **Study 7**: TEX11 listed among 5 confirmed NOA causative genes
- **RBP Table 1**: 8 pathogenic TEX11 variants including 2 splice-disrupting

---

## Dependencies

```
openpyxl>=3.1.0     # Excel parsing
pandas>=2.0.0       # Data manipulation
numpy>=1.24.0       # Numerical computing
pymc>=5.0.0         # Bayesian inference (MCMC)
arviz>=0.15.0       # MCMC diagnostics
scipy>=1.10.0       # Statistical tests
torch>=2.0.0        # Deep learning (diffusion model)
einops>=0.7.0       # Tensor operations
scikit-learn>=1.3.0  # Evaluation metrics
pytest>=7.0.0       # Testing
```

Optional for GENCODE pre-training:
```
pysam>=0.22.0       # FASTA access (recommended)
pyfaidx>=0.7.0      # FASTA access (alternative)
```

---

## Citation

If you use this framework, please cite:

```bibtex
@article{splicevarmech2026,
  title={SpliceVarMech: A Causal Generative Framework for Mechanistic
         Interpretation of Non-Canonical Splicing Variants in Male Infertility},
  author={Zein, Moustafa and Hassanien, Aboul Ella},
  year={2026},
  note={Manuscript in preparation}
}
```

### Key References

- Primary dataset: *"Mapping the Non-Canonical Splicing Variants"*, Advanced Science, 2024
- D3PM: Austin et al., NeurIPS 2021
- SpliceAI: Jaganathan et al., Cell 2019
- Bayesian causal inference: Pearl, *Causality*, 2009
- See `notebooks/base_proposal.md` Section 12 for complete reference list

---

## License

This project is for academic research purposes.
