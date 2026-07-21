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
│  MODULE 1: Discrete Diffusion Model (D3PM)                  │
│  Pre-mRNA context → Generate predicted mRNA (N samples)     │
│  → Mechanism classification (exon skip / intron retain / …) │
├─────────────────────────────────────────────────────────────┤
│  MODULE 2: Bayesian Causal Inference (PyMC SCM)             │
│  Structural causal model: V→S, V→E, V→I, D→O               │
│  → P(disruption | evidence) with 95% credible interval      │
├─────────────────────────────────────────────────────────────┤
│  MODULE 3: Explainable AI                                   │
│  Sequence attribution, causal paths, ACMG criteria mapping  │
│  → Clinical report with mechanism + uncertainty              │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/AI-VUS-Mechanism.git
cd AI-VUS-Mechanism

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Project Summary

```bash
python main.py              # Show module status and usage
```

### Run Individual Phases

```bash
python main.py --phase 1    # Parse primary dataset (2,404 variants)
python main.py --phase 2    # Splice tool coverage analysis
python main.py --phase 3    # Bayesian model diagnostics (no MCMC)
python main.py --phase 4    # Diffusion architecture validation
python main.py --phase 5    # Training pipeline (pre-train + fine-tune)
python main.py --phase 6    # TEX11 prediction + clinical report
python main.py --baselines  # 17-tool baseline evaluation (AUROC/AUPRC)
python main.py --xai        # Explainability analysis
python main.py --benchmark  # SOTA benchmarking (2019-2026 literature)
python main.py --all        # Run all phases sequentially
```

### Run Specific Modules

```bash
python -m src.data.parser              # Parse primary dataset
python -m src.data.external_parser     # Parse external studies (4 & 6)
python -m src.features.splice_scores   # Splice tool analysis
python -m src.causal.diagnostics       # Diagnostic analysis
python -m src.causal.dag               # Full Bayesian model comparison (V1 vs V2)
python -m src.baselines.tool_evaluation # Individual tool AUROC evaluation
python -m src.baselines.benchmark      # Literature benchmarking + validation
```

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
├── main.py                            # Unified CLI entry point
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
│
├── data/
│   ├── raw/ADVS-13-e15512-s001.xlsx   # Primary dataset
│   ├── external/                      # Literature supplementary data
│   │   ├── study6_splice_variants.xlsx
│   │   └── study4_tese_panel.xlsx
│   └── processed/                     # Parsed outputs
│
├── src/
│   ├── data/
│   │   ├── parser.py                  # Primary dataset parser
│   │   └── external_parser.py         # External literature data parser
│   ├── features/
│   │   └── splice_scores.py           # 16-tool splice score analysis
│   ├── causal/
│   │   ├── dag.py                     # Bayesian SCM (V1 + V2 improved)
│   │   └── diagnostics.py            # Class imbalance investigation
│   ├── diffusion/
│   │   ├── model.py                   # D3PM architecture (9.2M params)
│   │   ├── training.py                # Pre-train + fine-tune pipeline
│   │   └── sampling.py               # Multi-sample inference + mechanism
│   ├── pipeline/
│   │   └── predict.py                 # End-to-end TEX11 prediction
│   ├── xai/
│   │   └── attribution.py            # Sequence attribution + causal paths
│   └── baselines/
│       ├── tool_evaluation.py         # 17-tool AUROC/AUPRC evaluation
│       └── benchmark.py              # SOTA benchmarking (2019-2026)
│
├── notebooks/
│   ├── base_proposal.md               # Full project proposal
│   └── model_documentation.md         # Comprehensive model documentation
│
├── tests/
│   └── test_parser.py                 # 29 validation tests
│
├── experiments/                       # Training checkpoints
├── figures/                           # Generated figures
└── paper/                             # Manuscript drafts
```

---

## Key Results

### Bayesian Causal Model (Phase 3)

| Model | Accuracy | Balanced Acc | Sensitivity | Specificity | MCC |
|-------|----------|-------------|-------------|-------------|-----|
| V1 (Original) | 77.4% | 56.2% | 100.0% | 12.5% | +0.310 |
| **V2 (Improved, optimal)** | **80.6%** | **74.7%** | **87.0%** | **62.5%** | **+0.495** |

**Key diagnostic finding:** Tool scores alone cannot separate true NCSVs from false positives due to selection bias — the negatives were chosen *because* tools flagged them. This validates the need for the diffusion model.

### TEX11 Validation

TEX11 c.1156+16G>T independently confirmed as causative by:
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

---

## Citation

If you use this framework, please cite:

```bibtex
@article{splicevarmech2026,
  title={SpliceVarMech: A Causal Generative Framework for Mechanistic
         Interpretation of Non-Canonical Splicing Variants in Male Infertility},
  author = {Zein, Moustafa and Hassanien, Aboul Ella},
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
