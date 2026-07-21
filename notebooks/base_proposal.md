# SpliceVarMech: A Causal Generative Framework for Mechanistic Interpretation of Non-Canonical Splicing Variants in Male Infertility

## 1. The Biological Problem

### 1.1 Male Infertility and Spermatogenesis

Male infertility affects approximately 7% of men worldwide, and in roughly 40% of cases the underlying genetic cause remains unidentified — termed **idiopathic male infertility**. The production of mature sperm (spermatogenesis) is one of the most complex differentiation programs in human biology, requiring the coordinated expression of over 2,000 genes across a tightly regulated developmental timeline:

1. **Spermatogonial stem cells** undergo mitotic divisions to maintain the progenitor pool
2. **Primary spermatocytes** enter meiosis I, where homologous chromosomes must pair and undergo recombination (crossover formation) — this step is essential for genetic diversity and correct chromosome segregation
3. **Secondary spermatocytes** complete meiosis II to produce haploid spermatids
4. **Spermiogenesis** transforms round spermatids into mature spermatozoa with specialized structures (acrosome, flagellum, condensed chromatin)

Disruption at any stage causes **spermatogenic failure**, ranging from reduced sperm count (oligozoospermia) to complete absence of sperm (azoospermia). Genetic variants in genes essential for meiotic recombination are particularly devastating because they cause complete meiotic arrest — no mature sperm are produced at all.

### 1.2 TEX11 and Meiotic Recombination

**TEX11** (Testis Expressed 11) is an X-linked gene located at Xp11.22 that encodes a protein essential for the formation of crossover-specific recombination intermediates during meiosis I. Specifically, TEX11:

- Interacts with SHOC1 to promote the transition from early recombination intermediates to crossover-designated intermediates
- Is required for proper loading of the mismatch repair machinery (MLH1/MLH3) onto recombination sites
- Functions exclusively during male meiosis — the protein is testis-specific with no detectable expression in other tissues (confirmed by Human Protein Atlas)

Loss of TEX11 function causes **Spermatogenic Failure 2** (OMIM #309120): complete meiotic arrest at the pachytene/metaphase I stage, resulting in non-obstructive azoospermia. Because TEX11 is X-linked, males carry only one copy — a single damaging variant results in complete loss of gene function.

### 1.3 The Clinical Case

A 34-year-old male with consanguineous parentage presents with primary infertility due to **azoospermia** (zero sperm in ejaculate). Clinical workup:
- **Karyotype:** 46,XY (normal)
- **Hormone profile:** elevated FSH (consistent with spermatogenic failure)
- **Testicular biopsy:** meiotic arrest
- **Whole-exome sequencing:** identified a hemizygous variant in TEX11: **c.1156+16G>T**

This variant sits at **position +16 within the intron**, downstream of the exon-intron boundary. It is classified as a **Variant of Uncertain Significance (VUS)** because:
- It falls outside the canonical ±1/±2 splice sites that clinical laboratories routinely evaluate
- No existing computational tool can confidently determine whether a variant at this position disrupts RNA splicing
- No functional studies have been performed for this specific variant

**The unsolved question:** Does TEX11 c.1156+16G>T disrupt mRNA splicing, and if so, what is the exact molecular mechanism? Answering this would reclassify the VUS, explain the patient's azoospermia, and directly guide clinical management (e.g., genetic counseling, reproductive options).

---

## 2. Molecular Biology of RNA Splicing

### 2.1 The Splicing Process

Every protein-coding gene in the human genome is transcribed as a **pre-mRNA** that contains alternating exons (coding regions) and introns (non-coding regions). Before the mRNA can be translated into protein, introns must be precisely excised and exons joined — a process called **RNA splicing**, catalyzed by the **spliceosome**, a large ribonucleoprotein complex.

The spliceosome recognizes intron-exon boundaries through specific sequence signals:

| Signal | Location | Consensus | Function |
|--------|----------|-----------|----------|
| **5' donor splice site** | Exon-intron junction | GT at +1/+2, extended consensus to +6 | Recognized by U1 snRNP base-pairing |
| **3' acceptor splice site** | Intron-exon junction | AG at -1/-2 | Recognized by U2AF35 |
| **Polypyrimidine tract** | -5 to -40 upstream of 3' AG | Y-rich (C/T runs) | Bound by U2AF65 |
| **Branch point** | -18 to -44 upstream of 3' AG | YNYURAY (A is branch nucleotide) | Recognized by SF1, then U2 snRNP |
| **Exonic Splicing Enhancers (ESEs)** | Within exons | Varied (SR protein binding motifs) | Recruit SR proteins to promote exon inclusion |
| **Exonic Splicing Silencers (ESSs)** | Within exons | Varied (hnRNP binding motifs) | Recruit hnRNPs to promote exon skipping |
| **Intronic Splicing Enhancers (ISEs)** | Within introns, often near splice sites | Varied | Promote splice site recognition |
| **Intronic Splicing Silencers (ISSs)** | Within introns | Varied | Suppress splice site usage |

### 2.2 Why Non-Canonical Variants Can Disrupt Splicing

The canonical splice site (the GT at +1/+2 and AG at -1/-2) is only part of the recognition signal. The spliceosome relies on a **combinatorial code** of multiple sequence elements working together. This means variants at positions outside the canonical ±1/±2 can disrupt splicing through several mechanisms:

**a) Extended donor site disruption (positions +3 to +20):**
U1 snRNP base-pairs with the 5' splice site across positions -3 to +6. However, the extended intronic sequence from +7 to approximately +20 contains auxiliary signals that stabilize U1 binding and recruit additional factors (e.g., TIA-1, CELF proteins). Variants in this region — like c.1156+16G>T — can weaken these auxiliary interactions, reducing splice site strength below the recognition threshold.

**b) Exonic splicing enhancer (ESE) disruption:**
Missense variants that appear to only change the amino acid can simultaneously destroy ESE motifs. SR proteins (SRSF1, SRSF2, etc.) bind these motifs to recruit U1/U2 snRNP to nearby splice sites. Losing an ESE can cause the spliceosome to skip the entire exon. This explains why 25 of the 40 validated NCSVs in the reference dataset are missense variants.

**c) Synonymous variant effects:**
Even "silent" variants that don't change the amino acid can alter ESE/ESS balance. A synonymous change that converts an ESE hexamer into an ESS hexamer will flip the exon from "included" to "skipped." Three validated NCSVs in the reference dataset are synonymous.

**d) Deep intronic variants creating cryptic splice sites:**
A variant deep in the intron can create a new GT or AG dinucleotide with sufficient surrounding consensus to be recognized as a novel splice site, leading to inclusion of intronic sequence (pseudoexon activation) or partial intron retention.

### 2.3 Testis-Specific Splicing — A Unique Challenge

The testis has the most complex transcriptome of any human tissue, with:
- The highest number of tissue-specific alternative splicing events
- Expression of testis-specific splicing factors (e.g., RBMXL2, T-STAR/KHDRBS3)
- Unique chromatin accessibility patterns during meiosis that expose regulatory elements not active in other tissues
- Widespread use of alternative promoters and polyadenylation sites

This means **splice prediction tools trained on general tissue data systematically underperform on testis-expressed genes**. A variant that has no effect in somatic tissues may be devastating in the testis because of tissue-specific splicing factor expression and regulatory context. This biological reality is a key motivation for building a specialized model.

### 2.4 The Specific Biology of Position +16

For the TEX11 c.1156+16G>T variant specifically, the position +16 within the intron falls in a region where:

- **ISE motifs** frequently occur (intronic splicing enhancer elements typically cluster in the first 50-100 nucleotides of introns)
- **RNA secondary structures** can form between the 5' splice site region and downstream intronic sequences, stabilizing U1 snRNP binding
- **Regulatory protein binding sites** for factors like TIA-1 (which binds U-rich sequences downstream of weak 5' splice sites) may be present

A G>T transversion at this position could:
1. Disrupt an ISE motif, weakening splice site recognition → **exon skipping**
2. Alter local RNA secondary structure, destabilizing U1 snRNP binding → **exon skipping or intron retention**
3. Create or destroy a binding site for a splicing regulatory protein → **variable outcome depending on the factor**

The actual outcome can only be definitively determined by experimental validation (RT-PCR, minigene assay) — or, as we propose, by a computational model that has learned the rules of splice site recognition from large-scale data.

---

## 3. The Computational Gap

### 3.1 Current Splice Prediction Tools

At least 17 computational tools exist for predicting splice-disrupting variants:

| Tool | Approach | Strengths | Weaknesses |
|------|----------|-----------|------------|
| **SpliceAI** | Deep learning (ResNet) on pre-mRNA | Best for canonical/near-canonical sites | Produces false positives for deep NCSVs; only 14% coverage in reference dataset |
| **MaxEntScan** | Maximum entropy modeling of splice site motifs | Fast, interpretable | Only considers ±20bp around splice sites |
| **CADD-splice** | Integrated deleteriousness score | Broad feature integration | Not specialized for splicing |
| **Squirls** | Hexamer-based + positional features | Considers ESE/ESS | Limited by motif databases |
| **MMSplice** | Modular neural network | Models multiple splicing modules | Misses long-range interactions |
| **dbscSNV** | Ada/RF trained on known splice variants | Good for canonical sites | Poor at non-canonical positions |
| **Kipoi-splice** | Deep learning ensemble | Flexible architecture | Limited training on NCSVs |
| **SPiCE** | Probability-based integration | Combines multiple signals | Calibration issues |
| **GeneSplicer** | Markov models + decision trees | Fast | Outdated training data |
| **ESRseq** | Exonic regulatory sequence scores | Quantifies ESE/ESS | Exon-only, no intronic context |
| **Spliceogen** | Integration of donor/acceptor models | Multi-site analysis | Missing data issues |
| **SCAP** | Sequence-based classifier | Conservation-aware | Low coverage (19%) |
| **RegSNP** | Regulatory SNP predictor | Functional annotations | Very low coverage (5%) |
| **dpsi (SPOT)** | Tissue-specific PSI prediction | Tissue-aware | Limited to well-studied tissues |

### 3.2 The False Positive / False Negative Problem

The reference dataset documents a critical failure mode of current tools:

**False Negatives (tools miss real NCSVs):**
- 40 experimentally validated non-canonical variants DO disrupt splicing
- Many of these receive LOW scores from individual tools (SpliceAI < 0.5 for variants at positions > +10)
- No single tool reliably detects all 40

**False Positives (tools flag benign variants):**
- 14 variants in Table S2 were predicted to disrupt splicing but experimental validation showed **no effect**
- Example: AR:c.1768G>A received SpliceAI = 0.83 (high confidence) but outcome = "Normal"
- Example: variants with SPCards consensus scores > 10/17 showing no splice disruption

**The clinical consequence:** Clinicians cannot trust any single tool's prediction for non-canonical variants. This is why TEX11 c.1156+16G>T remains a VUS — tools disagree, and there is no reliable way to resolve the disagreement.

### 3.3 What No Existing Tool Does

Every current tool outputs a **score** — a number representing the probability of splice disruption. But clinicians need to know:

1. **What happens to the mRNA?** — Which exon is skipped? How many bases of intron are retained? Is a cryptic splice site activated?
2. **What happens to the protein?** — Does it cause a frameshift? An in-frame deletion? Loss of a critical domain?
3. **Why should we trust this prediction?** — What sequence features drive it? Which tools agree/disagree and why?

No existing tool generates the predicted aberrant mRNA sequence. No tool explains the mechanism. No tool provides end-to-end interpretation from variant to clinical consequence.

---

## 4. Dataset

### 4.1 Primary Dataset

The primary dataset comes from the supplementary materials of:

> *"Mapping the Non-Canonical Splicing Variants: Decrypting the Hidden Genetic Architecture of Idiopathic Male Infertility"* — Advanced Science, 2024

**File:** `ADVS-13-e15512-s001.xlsx` — 8 sheets containing curated genetic and splicing data.

### 4.2 Sheet-by-Sheet Description

| Sheet | Content | Records | Key Columns | Role in This Study |
|-------|---------|---------|-------------|-------------------|
| **Table S1** | Curated pathogenic variants in hereditary male infertility | 2,404 variants × 63 columns | Genomic coordinates, 17 splice tool scores, pathogenicity predictors, population frequencies, ClinVar, functional annotations | Semi-supervised pre-training; feature matrix for meta-learner |
| **Table S2** | Negative controls — computationally predicted but experimentally disproven | 25 variants (14 "Normal" + 11 "Failed") | Position, gene, SpliceAI score, SPCards score, outcome | Gold-standard negative examples (14 usable) |
| **Table S3** | Patient-level filtered variants (11 patients) | 58 variants × 95 columns | Patient ID, genotype, all Table S1 columns + tissue expression, pathways, GO terms, mouse phenotypes, pLI/pRec scores | Biological context; tissue-specific features |
| **Table S4** | Clinical semen analysis (13 patients) | 13 patients × 12 parameters | Volume, concentration, motility, morphology | Clinical phenotype correlation |
| **Table S5** | Extended patient variants (all types) | 6,310 variants × 101 columns | Same as Table S3 but comprehensive | Broader variant landscape |
| **Table S6** | ICSI clinical outcomes (1 patient) | 1 patient | Treatment outcomes | Clinical validation |
| **Table S7** | **Gold-standard positive NCSVs** | 40 variants × 6 columns | Gene:variant, type, splicing outcome, RT-PCR primers, **aberrant mRNA sequences** | Gold-standard positives; mechanism labels; diffusion model training targets |
| **Table S8** | Lab reagents and antibodies | 57 entries | Reagent details | Not computationally relevant |

### 4.3 The 17 Splice Prediction Tool Scores (Table S1, columns 35-54)

Each of the 2,404 variants is annotated with scores from up to 17 splice prediction tools. Coverage varies significantly:

| Coverage Level | Tools | Coverage |
|---------------|-------|----------|
| **High (>80%)** | splice_number, CADDsplice_phred, MaxEntScan, GeneSplicer, ESRseq, Spliceogen, Squirls_max_score, mmsplice_delta_logit_psi | 82-100% |
| **Medium (40-65%)** | Kipoisplice_pathogenic, dpsi_max_tissue, dpsi_zscore | 41-64% |
| **Low (<20%)** | SCAP_max, spliceAI_max_score, dbscSNV_ADA, dbscSNV_RF, max_SPiCEprobability, regsnp_fpr | 5-19% |

The high missingness in key tools (SpliceAI at 14%, dbscSNV at 8%) is itself informative — these tools often return no score for non-canonical variant positions, creating systematic blind spots.

### 4.4 Gold-Standard Labels

**Positive NCSVs (Table S7) — 40 variants with validated splicing outcomes:**

| Variant Type | Count | Example | Documented Outcome |
|-------------|-------|---------|-------------------|
| Missense causing splice disruption | 25 | LHCGR:c.265A>T | Exon 3 skipping |
| Intronic (non-canonical position) | 12 | MAP3K1:c.634-8T>A | Intron 2, 6bp retention |
| Synonymous causing splice disruption | 3 | NR5A1:c.990G>A | Exon 5 skipping |

Each positive NCSV includes the **actual aberrant mRNA sequence** — this is the critical training target for the diffusion model.

**Negative Controls (Table S2) — 14 usable "Normal" outcome variants:**

These are variants where tools predicted splice disruption but experimental validation showed no effect. They include variants with high SpliceAI scores (up to 0.83) — documented false positives.

### 4.5 External Data Sources for Augmentation

| Source | What It Provides | Role |
|--------|-----------------|------|
| **GENCODE v44** | ~250,000 annotated transcript isoforms with exon-intron structures | Pre-training: millions of (pre-mRNA context → mature mRNA) pairs for the diffusion model |
| **ClinVar** | ~2,000-5,000 classified splice-affecting variants across all diseases | Training augmentation: expand labeled positives |
| **gnomAD v4** | Population-frequency data for intronic variants | Training augmentation: common intronic variants as negatives (benign by frequency) |
| **UCSC Genome Browser** | PhyloP/PhastCons conservation, regulatory tracks, transcript structures | Feature extraction: conservation at variant position, ±200bp sequence context |
| **Human Protein Atlas** | Tissue-specific gene expression (RNA and protein) | Feature extraction: confirm testis-specificity, identify tissue-specific splicing patterns |
| **Ensembl / RefSeq** | Reference genome sequences (hg38) | Sequence extraction: ±200bp genomic context around each variant |

---

## 5. Proposed Approach

### 5.1 Core Insight: Splicing as a Sequence Transformation — and a Causal Question

The biological process of RNA splicing is fundamentally a **sequence transformation**: the cell takes a pre-mRNA sequence (exons + introns) and produces a mature mRNA sequence (exons only, correctly joined). The spliceosome reads sequence signals to decide where to cut and join. A genetic variant alters these sequence signals, potentially changing the transformation output.

But more importantly, the clinical question — *"Does this variant CAUSE splice disruption?"* — is inherently a **causal inference** question, not just a prediction task. We need to model the causal chain:

```
Variant (G>T at +16)
    → Changes local sequence context
        → Disrupts specific regulatory element (ISE/ESE/splice signal)
            → Alters spliceosome recognition
                → Produces aberrant mRNA
                    → Abnormal/truncated protein
                        → Loss of biological function
                            → Disease phenotype (azoospermia)
```

Each arrow is a causal relationship with associated uncertainty. Our framework models this entire chain, not just the endpoint.

### 5.2 Why This Architecture (Diffusion + Bayesian Causal Inference)

**Why not simple ML (XGBoost, Random Forest, Logistic Regression)?**

Simple ML approaches would fail here for several concrete reasons:

1. **54 gold-standard examples is below the reliability threshold** for discriminative ML. With 40 positives and 14 negatives, even leave-one-out cross-validation produces estimates with high variance. XGBoost on sparse features (many tools have >80% missing values) will overfit to the training idiosyncrasies rather than learning generalizable patterns.

2. **Simple ML produces correlations, not causal explanations.** A Random Forest might learn that "high MaxEntScan + low SpliceAI → splice disruption" but this tells you nothing about WHY the variant disrupts splicing. The clinical question demands mechanism, not correlation.

3. **No uncertainty quantification.** A simple classifier outputs P(disruption) = 0.78, but how confident is it in that 0.78? With 54 training examples, the model should be LESS confident than with 5,000 examples. Simple ML doesn't capture this epistemic uncertainty — Bayesian methods do.

4. **Cannot generate the biological outcome.** Classifiers predict labels, not sequences. The diffusion model generates the actual predicted aberrant mRNA, which IS the biological answer.

**Why Diffusion + Bayesian Causal Inference together?**

| Component | What It Answers | Why It's Needed |
|-----------|----------------|-----------------|
| **Diffusion Model** | WHAT happens to the mRNA? (generates the aberrant sequence) | The biological question requires knowing the exact molecular outcome, not just yes/no |
| **Bayesian Causal Model** | WHY does it happen? (models the causal chain with uncertainty) | Clinical reclassification requires understanding the causal mechanism with calibrated confidence |
| **XAI Layer** | WHERE in the sequence? (identifies causal nucleotides) | Clinicians need to see which specific sequence elements are disrupted |

### 5.3 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         SpliceVarMech Framework                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  MODULE 1: Discrete Sequence Diffusion Model                       │  │
│  │  (Generative Core — WHAT happens to the mRNA?)                     │  │
│  │                                                                    │  │
│  │  Pre-mRNA Context (±200bp) ──→ Conditional Diffusion Process       │  │
│  │       with variant                                                 │  │
│  │         ↓                                                          │  │
│  │  Generated Mature mRNA (sampled N times)                           │  │
│  │         ↓                                                          │  │
│  │  Alignment to wild-type mRNA → Mechanism Identification            │  │
│  │  (exon skipping / intron retention / cryptic site / normal)        │  │
│  │         ↓                                                          │  │
│  │  Outcome Distribution: 75% exon skip, 20% normal, 5% retention    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                          ↓                                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  MODULE 2: Bayesian Causal Inference Engine                        │  │
│  │  (Causal Reasoning — WHY does it happen?)                          │  │
│  │                                                                    │  │
│  │  Structural Causal Model (DAG):                                    │  │
│  │                                                                    │  │
│  │  Variant ──→ Sequence Context Change ──→ Regulatory Element        │  │
│  │     │              │                      Disruption               │  │
│  │     │              ↓                         │                     │  │
│  │     │        Conservation Signal              ↓                    │  │
│  │     │              │                   Splice Site Strength         │  │
│  │     │              ↓                      Change                   │  │
│  │     └──→ Position Context ──────────→     │                        │  │
│  │                                           ↓                        │  │
│  │                                    Splicing Outcome                 │  │
│  │                                    (with posterior                  │  │
│  │                                     uncertainty)                   │  │
│  │                                                                    │  │
│  │  Bayesian Inference:                                               │  │
│  │  • Prior: biological knowledge (splice site rules, motif DBs)      │  │
│  │  • Likelihood: diffusion model output + sequence features          │  │
│  │  • Posterior: P(disruption | evidence) with credible intervals     │  │
│  │                                                                    │  │
│  │  Interventional Query (do-calculus):                               │  │
│  │  • P(aberrant mRNA | do(G→T at +16)) — the causal effect          │  │
│  │  • Counterfactual: "Would splicing be normal if G were restored?"  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                          ↓                                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  MODULE 3: Explainable AI & Clinical Interpretation                │  │
│  │  (WHERE in the sequence and WHAT it means clinically)              │  │
│  │                                                                    │  │
│  │  • Sequence attribution maps → causal nucleotides identified       │  │
│  │  • Causal path visualization → which regulatory element disrupted  │  │
│  │  • Protein consequence → frameshift / domain loss / NMD            │  │
│  │  • Uncertainty visualization → credible intervals for clinicians   │  │
│  │  • Clinical evidence grade → ACMG criteria mapping                 │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                          ↓                                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  OUTPUT: Causal Clinical Report with Uncertainty                   │  │
│  │                                                                    │  │
│  │  "TEX11 c.1156+16G>T CAUSES exon N skipping                      │  │
│  │   (posterior probability: 0.89, 95% CI: [0.79, 0.95])            │  │
│  │   through disruption of ISE motif at +14-18                       │  │
│  │   → frameshift → premature stop → loss of recombination domain   │  │
│  │   → spermatogenic failure → azoospermia                           │  │
│  │   Classification: Likely Pathogenic (PP3_Strong + PM2)"           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Module 1: Discrete Sequence Diffusion Model

#### Why Discrete Diffusion for DNA/RNA

DNA and RNA sequences are inherently **discrete** — each position is one of four nucleotides (A, C, G, T/U). Standard continuous diffusion models (like DDPM) operate in continuous space and are not naturally suited for discrete sequences. We use a **discrete diffusion** framework (building on D3PM — Discrete Denoising Diffusion Probabilistic Models) that operates directly on the nucleotide alphabet.

The key advantage over continuous diffusion: no need for embedding/de-embedding — the model reasons directly about nucleotide identities and their biological meaning (e.g., GT dinucleotide = splice donor signal).

#### The Diffusion Process Applied to Splicing

**Training Phase — Learning the Rules of Splicing:**

The model learns from paired examples of (pre-mRNA context → mature mRNA):

1. **Large-scale pre-training** on normal splicing: Using GENCODE-annotated transcripts, we extract millions of examples where the input is a pre-mRNA region (e.g., exon-intron-exon spanning ~400-600bp) and the target output is the correctly spliced mRNA (exon-exon junction). The model learns the general rules: recognize GT/AG boundaries, respect ESE/ESS motifs, identify branch points.

2. **Fine-tuning on variant effects**: Using the 40 gold-standard NCSVs from Table S7, where we have (mutant pre-mRNA context → experimentally validated aberrant mRNA). The model learns how specific variants alter splicing outcomes. The 14 "Normal" negatives from Table S2 teach the model that not all variants at non-canonical positions disrupt splicing.

**Inference Phase — Predicting Splice Outcome for TEX11:**

1. Input the pre-mRNA context around TEX11 c.1156+16G>T (±200bp including the variant)
2. The diffusion model generates predicted mature mRNA through iterative denoising
3. Sample multiple times (e.g., 1,000 samples) to capture the probability distribution over possible outcomes
4. Align generated mRNAs to the expected wild-type TEX11 mRNA
5. The alignment reveals: Is an exon skipped? Is intronic sequence retained? Where exactly?

**Why sampling multiple times matters:** In biology, a splice-disrupting variant doesn't always cause 100% aberrant splicing. Often, the outcome is a mixture — e.g., 60% normal transcript + 40% exon-skipped transcript. By generating many samples and counting the frequency of each outcome, the model naturally estimates this ratio, which has direct clinical relevance (partial vs. complete loss of function).

#### Diffusion for Data Augmentation

Beyond its role as the core generative model, the diffusion framework also addresses the small dataset limitation:

1. **Synthetic variant generation**: After pre-training on normal splice junctions, the model understands the sequence features that define splice sites. We can introduce synthetic variants at various positions and generate predicted outcomes, creating thousands of labeled (variant → outcome) pairs that expand the training distribution.

2. **Counterfactual sequence generation**: Generate "what if" scenarios — what would the mRNA look like if the variant were at +14 instead of +16? At +18? This creates biologically plausible augmented data while respecting the sequence constraints the model has learned.

### 5.5 Module 2: Bayesian Causal Inference Engine

#### Why Causal Inference, Not Just Prediction

The clinical question is causal: *"Does this variant CAUSE the splicing defect?"* This is fundamentally different from *"Is this variant associated with a splicing defect?"* Causal inference provides:

1. **Interventional predictions**: P(aberrant mRNA | do(G→T at +16)) — what happens if we introduce this variant? This is the do-calculus formulation of the clinical question.

2. **Counterfactual reasoning**: "If this patient had the wild-type G instead of T at +16, would splicing be normal?" — directly answering whether the variant explains the phenotype.

3. **Calibrated uncertainty**: Bayesian posteriors with credible intervals, not just point estimates. With 54 gold-standard examples, the model SHOULD express uncertainty — and clinicians need to see that uncertainty to make informed decisions.

#### Structural Causal Model (SCM)

We define a biologically grounded causal DAG (Directed Acyclic Graph) that encodes the known biology of splice site recognition:

**Causal Variables (Nodes):**

| Node | Biological Meaning | Measurement |
|------|-------------------|-------------|
| **V** — Variant | The nucleotide change (G>T) | Binary: present/absent |
| **P** — Position | Location relative to splice site (+16) | Integer: distance from exon-intron boundary |
| **C** — Conservation | Evolutionary constraint at this position | PhyloP / PhastCons score |
| **S** — Splice Site Strength | How well the spliceosome recognizes this junction | MaxEntScan score for the donor/acceptor site |
| **E** — ESE/ESS Balance | Whether exonic regulatory elements are intact | ESRseq score, hexamer analysis |
| **I** — ISE/ISS Impact | Whether intronic regulatory elements are disrupted | Computed from sequence motif analysis |
| **R** — RNA Structure | Whether local secondary structure is altered | RNAfold minimum free energy change |
| **D** — Diffusion Output | Generated mRNA from Module 1 | Aberrant vs. normal classification from diffusion samples |
| **O** — Splicing Outcome | The actual biological result | Normal / Exon skipping / Intron retention / Cryptic site |

**Causal Edges (Arrows):**

```
V → S  (variant changes splice site strength)
V → E  (variant may disrupt ESE/ESS in exon)
V → I  (variant may disrupt ISE/ISS in intron)
V → R  (variant may alter RNA secondary structure)
P → S  (position determines which splice signal is affected)
P → I  (position determines which ISE/ISS could be disrupted)
C → O  (highly conserved positions more likely to be functionally important)
S → O  (weaker splice site → higher probability of aberrant splicing)
E → O  (ESE loss → exon skipping; ESS loss → exon inclusion)
I → O  (ISE loss → weakened splice recognition)
R → O  (structural change → altered splice factor binding)
D → O  (diffusion model prediction informs outcome)
```

#### Bayesian Inference

**Priors (encoding biological knowledge):**

We set biologically informed priors on each causal variable:

- **P(disruption | position)**: Variants at +1/+2 have ~95% disruption rate; rate decreases with distance but has secondary peaks at known regulatory positions (+5, branch point region). Prior derived from published splice site mutation databases.
- **P(disruption | conservation)**: Highly conserved positions (PhyloP > 2) are more likely functionally important. Prior derived from ClinVar pathogenic vs. benign variants.
- **P(mechanism | position)**: Donor-side variants (+positions) more likely cause exon skipping; acceptor-side variants (-positions) more likely cause intron retention. Prior derived from Table S7 validated outcomes.

**Likelihood (from data and diffusion model):**

The diffusion model output provides the key likelihood term: given the variant context, what is the probability of each splicing outcome? The distribution over 1,000 generated samples directly estimates this likelihood.

**Posterior (what we want):**

Using Bayesian inference (implemented via MCMC or variational inference), we compute:

```
P(O = exon_skipping | V = G>T, P = +16, C, S, E, I, R, D)
```

This posterior comes with a **95% credible interval**, telling the clinician not just "we predict exon skipping" but "we predict exon skipping with probability 0.89 (95% CI: 0.79-0.95)." This calibrated uncertainty is essential for clinical decision-making.

#### Do-Calculus: The Interventional Query

The clinical question translates directly to a do-calculus query:

**P(aberrant mRNA | do(V = G→T at +16))**

This asks: *"If we intervene to change the G to T at position +16 (which is what the patient's genome has), what is the probability of aberrant mRNA?"*

This is different from the observational P(aberrant mRNA | V = G→T), which might be confounded by other factors. The causal query, via the SCM and do-calculus, controls for confounders and gives the true causal effect of the variant.

**Counterfactual:**

*"In this specific patient, if the T at +16 were the wild-type G, would splicing be normal?"*

This counterfactual directly answers whether the variant explains the patient's azoospermia. It's computed by:
1. Abduction: infer the patient's exogenous noise variables from observed data
2. Action: set V = G (wild-type)
3. Prediction: compute the counterfactual splicing outcome

If the counterfactual shows normal splicing → the variant is causal for the phenotype.

### 5.6 Module 3: Explainable AI & Clinical Interpretation

**Explainability is not optional — it IS the clinical contribution.** A prediction without explanation is useless in clinical genetics. We provide four layers of explanation:

1. **Sequence Attribution (on the diffusion model):**
   - Which nucleotides in the pre-mRNA context most influence the generated output?
   - Computed via gradient-based attribution on the diffusion model's denoising network
   - Visualize as a heatmap over the ±200bp region
   - Expected result for TEX11: high attribution at position +16 and surrounding ISE motif positions

2. **Causal Path Visualization (from the SCM):**
   - Which causal path from variant to outcome is strongest?
   - Example: V → I → O (variant disrupts ISE → aberrant splicing) vs. V → S → O (variant weakens splice site directly)
   - Computed from the posterior marginals of each node in the causal graph
   - Clinically: tells the genetic counselor WHICH biological mechanism is disrupted

3. **Mechanism Visualization:**
   - Alignment of generated aberrant mRNA vs. expected wild-type mRNA
   - Graphical representation: exon map showing skipped/retained/truncated regions
   - Protein consequence: reading frame analysis → frameshift / in-frame deletion / domain mapping
   - NMD prediction: will the aberrant transcript be degraded by nonsense-mediated decay?

4. **Uncertainty Visualization:**
   - Posterior distribution over outcomes (pie chart: 75% exon skip, 20% normal, 5% retention)
   - Credible intervals on each prediction
   - Comparison to prior (how much did the data update our belief?)
   - Clinical confidence grade: High / Moderate / Low based on posterior concentration

### 5.7 End-to-End Pipeline for the TEX11 Case

```
TEX11 c.1156+16G>T
        ↓
[Extract ±200bp pre-mRNA context from hg38]
        ↓
[Module 1: Diffusion Model — WHAT happens?]
  → Generate 1,000 predicted mRNA samples
  → 750/1,000 show exon N skipping
  → 200/1,000 show normal splicing
  → 50/1,000 show partial intron retention
  → Primary prediction: Exon skipping (75% of samples)
        ↓
[Module 2: Bayesian Causal Inference — WHY does it happen?]
  → Causal DAG evaluation:
     • V → I → O path (ISE disruption): posterior weight 0.72
     • V → R → O path (RNA structure): posterior weight 0.18
     • V → S → O path (direct splice site): posterior weight 0.10
  → Interventional: P(exon skip | do(G→T)) = 0.89 (95% CI: 0.79-0.95)
  → Counterfactual: P(normal | do(T→G)) = 0.94 → variant IS causal
        ↓
[Module 3: XAI — WHERE and WHAT does it mean?]
  → Attribution: G>T at +16 disrupts ISE motif at positions +14 to +18
  → Causal path: ISE disruption is the primary mechanism (0.72 posterior)
  → Mechanism: Exon N skipping → frameshift → premature stop codon at position Y
  → Protein: loss of recombination domain → disrupted meiotic crossover
  → NMD: transcript likely degraded → complete loss of TEX11 protein
        ↓
[Clinical Report with Calibrated Uncertainty]
  → TEX11 c.1156+16G>T CAUSES exon N skipping
     (posterior: 0.89, 95% CI: [0.79, 0.95])
  → Mechanism: ISE disruption at +14-18 (causal path probability: 0.72)
  → Consequence: frameshift → NMD → no TEX11 protein
  → Clinical: complete loss of meiotic recombination → azoospermia
  → ACMG: PP3_Strong (computational evidence) + PM2 (absent in gnomAD)
  → Recommendation: Reclassify VUS → LIKELY PATHOGENIC
```

---

## 6. Baseline Comparison Strategy

The 17 existing splice prediction tools are NOT part of the core model — they serve as **baselines to demonstrate our method's superiority**. This is a critical distinction: we don't stack broken tools; we replace them with a fundamentally better approach and then prove it.

### 6.1 Baseline Evaluation

Each of the 17 tools will be evaluated individually on the gold-standard set (40 positives + 14 negatives):

| Baseline | Approach | Expected Weakness |
|----------|----------|-------------------|
| **SpliceAI** | Deep learning on pre-mRNA | False positives at non-canonical positions (documented: AR:c.1768G>A → 0.83 but Normal) |
| **MaxEntScan** | Maximum entropy splice site model | Misses variants beyond ±20bp |
| **CADD-splice** | Integrated deleteriousness | Not specialized for splicing mechanism |
| **Best individual tool** | Whichever single tool has highest AUROC | Still limited by its single algorithm's assumptions |
| **Tool majority vote** | Consensus across available tools | Garbage consensus from tools that individually fail |
| **XGBoost on tool scores** | ML ensemble of 17 scores | Sparse features (massive missingness), overfits on 54 examples |
| **DNABERT-2 fine-tuned** | DNA language model classifier | Classification only — no mechanism prediction, no uncertainty |

Our method must beat ALL of these to justify the diffusion + causal approach.

---

## 7. Training Strategy

### 7.1 Phase 1: Pre-training the Diffusion Model on Normal Splicing

| Parameter | Value |
|-----------|-------|
| **Training data** | ~1M splice junction examples from GENCODE v44 (human + mouse) |
| **Input** | Pre-mRNA sequence: exon (50-100bp) + intron (up to 200bp) + exon (50-100bp) |
| **Target** | Correctly spliced mRNA: exon-exon junction |
| **Objective** | Learn the sequence rules of splice site recognition — GT/AG boundaries, ESE/ESS motifs, branch points, polypyrimidine tracts |
| **Architecture** | D3PM-based discrete diffusion with transformer backbone |
| **Expected outcome** | Model generates correct exon-exon junctions for >95% of known splice sites |

### 7.2 Phase 2: Fine-tuning on Variant Effects

| Parameter | Value |
|-----------|-------|
| **Positive examples** | 40 NCSVs from Table S7 (mutant pre-mRNA → experimentally validated aberrant mRNA) |
| **Negative examples** | 14 "Normal" from Table S2 (mutant pre-mRNA → normal mRNA despite tool predictions) |
| **Augmented positives** | ClinVar splice-affecting variants (~2,000-5,000) with known outcomes |
| **Augmented negatives** | gnomAD common intronic variants (~10,000) — benign by population frequency |
| **Synthetic data** | Counterfactual variants generated by the pre-trained diffusion model (variants at positions +3 to +50) |
| **Objective** | Learn how specific variants alter splicing outcomes |
| **Validation** | Leave-one-out cross-validation on gold standard (54 examples) |

### 7.3 Phase 3: Bayesian Causal Model Calibration

| Parameter | Value |
|-----------|-------|
| **DAG structure** | Biologically defined (Section 5.5) — V, P, C, S, E, I, R, D → O |
| **Priors** | Informed by published splice site mutation rates, ClinVar, and conservation databases |
| **Likelihood** | Diffusion model output distributions + computed sequence features (MaxEntScan, ESRseq, PhyloP, RNAfold) |
| **Inference method** | MCMC (PyMC / NumPyro) or Variational Inference (for scalability) |
| **Calibration** | Posterior calibration checked against Table S7 known outcomes — do 90% credible intervals contain the true outcome ≥90% of the time? |
| **Output** | Calibrated posterior P(outcome | evidence) with credible intervals |

### 7.4 Phase 4: Application to TEX11

| Step | Action |
|------|--------|
| Extract TEX11 pre-mRNA context | ±200bp around c.1156+16 from hg38 reference genome |
| Run diffusion model | Generate 1,000 predicted mRNA samples (wild-type context AND mutant context) |
| Compare outputs | Align mutant-context samples to wild-type-context samples → identify differences |
| Compute causal features | MaxEntScan score change, ISE motif analysis, PhyloP at +16, RNAfold ΔG |
| Run Bayesian causal model | Compute posterior P(exon skipping \| all evidence) with credible interval |
| Run counterfactual | P(normal splicing \| do(T→G)) — would restoring wild-type fix splicing? |
| Generate clinical report | Full interpretable output with mechanism, confidence, and ACMG criteria |

---

## 8. Evaluation Plan

### 8.1 Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **AUROC** | Binary discrimination (disrupts vs. doesn't) | > 0.90 (beat best single tool AND best tool ensemble) |
| **AUPRC** | Precision-recall under class imbalance | > 0.85 |
| **Mechanism accuracy** | Correct mechanism type among 40 positives | > 70% on leave-one-out |
| **Sequence identity** | Generated aberrant mRNA vs. Table S7 validated sequence | > 90% nucleotide identity |
| **False positive control** | Correctly rejecting Table S2 "Normal" variants | > 85% (≥12/14 correct) |
| **Calibration** | Do 90% credible intervals contain true outcome ≥90% of the time? | Calibration error < 0.10 |
| **Counterfactual consistency** | Does counterfactual (restore wild-type) predict normal splicing for negatives? | > 90% consistency |

### 8.2 Ablation Studies

| Ablation | What It Tests |
|----------|--------------|
| Diffusion model without Bayesian causal layer | Is causal reasoning needed beyond generation? |
| Bayesian causal model without diffusion (using only computed features) | Does the diffusion model add value over handcrafted features? |
| Pre-training on 100K vs. 500K vs. 1M splice junctions | How much pre-training data is needed? |
| With vs. without ClinVar/gnomAD augmentation | Does data augmentation improve fine-tuning? |
| With vs. without counterfactual synthetic data | Does diffusion-generated augmentation help? |
| Bayesian model with vs. without informative priors | How much do biological priors contribute? |

### 8.3 Comparison Against All 17 Existing Tools

For each tool, we compute AUROC, AUPRC, sensitivity, specificity, and mechanism prediction capability (which no existing tool has) on the gold standard. This table becomes a central figure in the paper, demonstrating that our approach is the first to simultaneously achieve high discrimination AND mechanism prediction.

---

## 9. Novelty Claims

1. **First application of diffusion models to splice variant mechanism prediction** — no published method generates predicted aberrant mRNA sequences from variant context. Existing tools output scores; we output the actual predicted molecular outcome.

2. **Mechanism prediction through generation, not classification** — the mechanism is READ from the generated sequence, not predicted as a discrete label. This enables discovery of novel mechanism types not in the training data (e.g., complex events combining partial exon deletion with intron retention).

3. **Bayesian causal inference for variant pathogenicity** — we model the biological causal chain (variant → regulatory element disruption → splice outcome) with calibrated uncertainty, providing credible intervals rather than point estimates. This is the first integration of structural causal models with generative sequence models for clinical genetics.

4. **Probabilistic splicing outcomes with penetrance estimation** — the diffusion model naturally estimates the ratio of normal to aberrant transcripts through sampling, capturing partial penetrance that no existing tool quantifies.

5. **Dual role of diffusion: prediction AND augmentation** — the same model that generates predictions also generates counterfactual synthetic training data, addressing the small labeled dataset limitation from within the framework rather than relying on external augmentation strategies.

6. **Counterfactual reasoning for clinical genetics** — do-calculus queries ("would splicing be normal if the variant were absent?") directly answer the clinical question of variant causality, going beyond association to establish causal responsibility.

7. **End-to-end clinical interpretability** — from variant to causal mechanism to protein consequence to ACMG criteria, with calibrated uncertainty at every step.

### 9.1 Novelty Assessment Summary

| Novelty Claim | What's New | What Exists | Confidence |
|---|---|---|---|
| Diffusion for splice variant mechanism | Generates predicted aberrant mRNA sequences from variant context | SpliceAI/MMSplice output scores only; EvoDiff/DDSM target proteins not splice variants | ⭐⭐⭐⭐⭐ High |
| Mechanism through generation | Mechanism is READ from alignment of generated vs. wild-type mRNA, not classified as a discrete label | All 17 tools classify (yes/no or score); none generate the molecular outcome | ⭐⭐⭐⭐⭐ High |
| SCM + generative model integration | Structural causal model with do-calculus integrated with generative diffusion for clinical genetics | Bayesian networks in genetics exist; causal inference in genomics discussed; but never combined with generative splice models | ⭐⭐⭐⭐⭐ High |
| Probabilistic penetrance via sampling | Multiple diffusion samples estimate normal:aberrant transcript ratio (e.g., 75% skip, 25% normal) | No existing tool quantifies partial penetrance for splice variants | ⭐⭐⭐⭐⭐ High |
| Dual diffusion (predict + augment) | Same model generates predictions AND counterfactual synthetic training data | Data augmentation exists separately; using the generative model itself for both roles is novel | ⭐⭐⭐⭐ High |
| Counterfactual variant causality | do-calculus queries (P(normal \| do(T→G))) directly answer causal attribution for individual variants | Causal inference applied in GWAS/eQTL; never to single-variant splice causality with counterfactual reasoning | ⭐⭐⭐⭐ Medium-High |
| End-to-end clinical interpretability | Variant → causal mechanism → protein consequence → ACMG criteria, with calibrated uncertainty at every step | Individual tools provide fragments; no tool chains the full causal path with uncertainty | ⭐⭐⭐⭐⭐ High |

---

## 10. Known Challenges & Risk Mitigation

### 10.1 Critical Challenges

| Challenge | Severity | Description | Mitigation Strategy |
|---|---|---|---|
| **Small gold-standard (N=54)** | 🔴 High | 40 positives + 14 negatives is extremely small for deep learning fine-tuning. Leave-one-out on N=54 produces wide confidence intervals (~±2% AUROC per variant). | Pre-train on ~1M GENCODE splice junctions (model learns general splicing rules); augment with ClinVar splice variants (~2-5K) and gnomAD common intronic variants (~10K negatives); Bayesian model is inherently robust to small N. |
| **Variable-length sequence generation** | 🔴 High | Pre-mRNA input is ~400bp but output varies: exon skipping = shorter, intron retention = longer, partial deletion = intermediate. Most D3PM implementations assume fixed-length output. | Use masking/padding approach (generate to max length, learn stop/pad tokens); alternatively, predict splice outcome type first (length hint), then generate conditionally. |
| **No ground truth for TEX11** | 🟡 Medium | The target prediction (TEX11 c.1156+16G>T) cannot be verified without wet-lab experiments (RT-PCR/minigene assay). | Frame as a prediction with calibrated uncertainty awaiting experimental confirmation; demonstrate strong performance on the 54 known variants first; the Bayesian credible interval communicates the confidence level transparently. |
| **Circular reasoning in synthetic augmentation** | 🟡 Medium | Using the pre-trained diffusion model to generate counterfactual training data, then evaluating on related data, risks information leakage. | Strict cross-validation: synthetic data generated from fold-excluded variants only; augmented data never overlaps with test variants; ablation study with vs. without synthetic augmentation. |
| **ClinVar augmentation limitations** | 🟡 Medium | Most ClinVar splice-affecting variants have classification labels but NOT the actual aberrant mRNA sequence. They can inform the Bayesian model but not directly train the diffusion model's generative output. | Use ClinVar for the Bayesian causal model features and for binary classification fine-tuning; reserve generative fine-tuning for the 40 Table S7 variants that have full aberrant mRNA sequences. |
| **Sequence identity target (>90%)** | 🟡 Medium | Full mRNAs are hundreds to thousands of bp. Achieving >90% identity requires correctly generating almost the entire sequence, not just the aberrant junction. | Most of the sequence is correct exonic sequence (learned during pre-training). The real evaluation should focus on junction accuracy — whether the model correctly identifies the aberrant junction location and type. |

### 10.2 Dataset Validation Notes (Corrected from Original README)

During independent validation of the dataset, the following was confirmed:

| README Claim | Actual Data | Status |
|---|---|---|
| Table S1: 2,404 variants × 63 cols | 2,405 data rows × 63 cols (header row offset) | ✅ Confirmed |
| Table S2: 14 "Normal" + 11 "Failed" | 14 Normal + 11 Failed = 25 | ✅ Confirmed |
| Table S7: 40 validated NCSVs with aberrant mRNA sequences | 40 data rows, each with Gene:variant, Type, Outcome, RT-PCR primers, full mRNA sequence | ✅ Confirmed |
| S7 types: "24 missense, 13 intronic, 3 synonymous" | Actual: 25 Mis, 12 Intron, 3 Syn = 40 total | ⚠️ Minor discrepancy (corrected) |
| 17 splice tool score columns in S1 | Cols 35-54 confirmed: splice_number, CADDsplice_phred, MaxEntScan, GeneSplicer, ESRseq, Spliceogen, Squirls, dbscSNV_ADA/RF, Kipoisplice, mmsplice, regsnp, SCAP, dpsi, spliceAI, SPiCE + distance/delta metrics | ✅ Confirmed |
| S7 outcome diversity | 34 distinct outcomes: exon skipping (majority), intron retention, partial deletions, complex events (e.g., "Intron 74 retention and Exon 75 65bp deletion") | ✅ Confirmed |
| Full aberrant mRNA sequences in S7 | Present — full-length mRNA sequences (hundreds to thousands of bp each); abnormal regions indicated by formatting in original | ✅ Confirmed |

---

## 11. Project Structure

```
AI-VUS-Mechanism/
├── README.md                       # This file
├── pyproject.toml                  # Project configuration and dependencies
├── data/
│   ├── raw/                        # Original dataset (ADVS-13-e15512-s001.xlsx)
│   ├── processed/                  # Parsed tables, extracted sequences, labels
│   └── external/                   # ClinVar, gnomAD, GENCODE downloads
├── src/
│   ├── data/                       # Data loading, parsing, sequence extraction
│   ├── diffusion/                  # Discrete diffusion model (D3PM-based)
│   │   ├── model.py                # Diffusion model architecture
│   │   ├── training.py             # Pre-training and fine-tuning loops
│   │   ├── sampling.py             # Inference: generate mRNA samples
│   │   └── augmentation.py         # Counterfactual data generation
│   ├── causal/                     # Bayesian causal inference engine
│   │   ├── dag.py                  # Structural causal model definition
│   │   ├── priors.py               # Biologically informed priors
│   │   ├── inference.py            # MCMC / variational inference
│   │   └── counterfactual.py       # Do-calculus and counterfactual queries
│   ├── features/                   # Sequence feature computation
│   │   ├── splice_strength.py      # MaxEntScan, donor/acceptor scoring
│   │   ├── motifs.py               # ESE/ESS/ISE/ISS motif analysis
│   │   ├── conservation.py         # PhyloP/PhastCons extraction
│   │   └── structure.py            # RNA secondary structure (RNAfold)
│   ├── xai/                        # Explainability modules
│   │   ├── attribution.py          # Sequence attribution maps
│   │   ├── causal_paths.py         # Causal path visualization
│   │   └── clinical_report.py      # End-to-end report generation
│   ├── baselines/                  # 17-tool baseline evaluation
│   │   └── tool_evaluation.py      # Individual and ensemble tool benchmarks
│   └── pipeline/                   # End-to-end inference pipeline
│       └── predict.py              # Full pipeline: variant → clinical report
├── notebooks/                      # Exploration and analysis notebooks
├── experiments/                    # Training configs, hyperparameters, results
├── figures/                        # Generated figures for paper
└── paper/                          # Manuscript drafts
```

---

## 12. References

### Primary Data Source
- *"Mapping the Non-Canonical Splicing Variants: Decrypting the Hidden Genetic Architecture of Idiopathic Male Infertility"* — Advanced Science, 2024

### Clinical Genetics
- TEX11 and spermatogenic failure: OMIM #309120
- ACMG variant classification guidelines: Richards et al., Genetics in Medicine, 2015

### Splice Prediction Tools (Baselines)
- SpliceAI: Jaganathan et al., Cell, 2019
- MaxEntScan: Yeo & Burge, Journal of Computational Biology, 2004
- MMSplice: Cheng et al., Genome Biology, 2019
- CADD: Rentzsch et al., Nucleic Acids Research, 2019
- Pangolin: Zeng & Li, Genome Biology, 2022
- AbSplice: Wagner et al., Nature Genetics, 2023
- CADD-splice v1.7: Rentzsch et al., NAR, 2024
- SpliceBERT: Chen et al., Bioinformatics, 2024
- GPN-MSA: Benegas et al., ICLR, 2024

### DNA/RNA Foundation Models
- DNABERT-2: Zhou et al., ICLR, 2024
- Nucleotide Transformer: Dalla-Torre et al., Nature Methods, 2024
- Evo: Nguyen et al., Science, 2024
- AlphaMissense: Cheng et al., Science, 2023

### Diffusion Models
- D3PM (Discrete Denoising Diffusion): Austin et al., NeurIPS, 2021
- EvoDiff: Alamdari et al., Nature Biotechnology, 2023
- Dirichlet Diffusion Score Model: Avdeyev et al., ICML, 2023

### Bayesian Causal Inference
- Pearl, Judea. *Causality: Models, Reasoning, and Inference*, Cambridge University Press, 2009
- Peters, Janzing & Schölkopf. *Elements of Causal Inference*, MIT Press, 2017
- PyMC: Salvatier et al., PeerJ Computer Science, 2016
- NumPyro: Phan et al., 2019

### RNA Splicing Biology
- Wang & Burge, "Splicing regulation: from a parts list to regulatory networks", RNA, 2008
- Cartegni et al., "Listening to silence and understanding nonsense: ESEs and ESSs", Nature Reviews Genetics, 2002
- Baralle & Giudice, "Alternative splicing as a regulator of development and tissue identity", Nature Reviews Molecular Cell Biology, 2017

### Male Infertility Genetics & Splicing (Recent 2024-2025)
- **[CRITICAL — Splicing & Infertility]** *"Defects in mRNA splicing and implications for infertility: a comprehensive review and in silico analysis"* — Human Reproduction Update, 2024/2025. **73 functionally validated splicing variants in 54 genes; 27 non-canonical splice variants classified as VUS by standard analysis. Directly validates our core premise. Data source: 27 NCSVs for training augmentation.**
- **[CRITICAL — TEX11 Validation]** *"Genetic determinants of testicular sperm extraction outcomes: insights from a large multicentre study of men with non-obstructive azoospermia"* — Human Reproduction, 2025. **571 NOA patients, 145-gene panel, TEX11 confirmed with 10+ TESE-negative cases. Independent validation of our clinical case.**
- **[HIGH — VUS Problem Quantification]** *"The genetic insights of sporadic male infertility: a systematic review of WES and WGS studies (2014–2024)"* — 2025. **143 genes, 47% VUS burden, only 34% functionally validated. Quantifies the clinical gap our framework addresses.**
- **[HIGH — RBP/Splicing Factor Biology]** *"The intricate dance of RNA-binding proteins: unveiling the mechanisms behind male infertility"* — Human Reproduction Update, 2025. **91 VUS in 35 RBP genes, 177 pathogenic variants in 62 RBP genes, 1744 RBP atlas. Provides biological context for XAI causal paths (RBPs are the trans-acting splicing factors).**
- **[MODERATE — TEX11 Independent Confirmation]** *"Systematic molecular analyses for 115 karyotypically normal men with isolated non-obstructive azoospermia"* — Human Reproduction, 2025. **TEX11 identified as causative gene alongside DMRT1, PLK4, SYCP2, USP26.**
- **[MODERATE — Multi-omics Context]** *"Bioinformatics and multi-omics approaches in male infertility: implications for diagnosis and assisted reproduction"* — 2025. **Reviews AI/bioinformatics in reproductive medicine; positions our work in the computational landscape.**
- **[CONTEXT — Clinical Translation]** *"Advances in point-of-care molecular testing for non-obstructive azoospermia: Biomarkers and translational perspectives"* — 2025. **AI-driven clinical decision-making in NOA; validates translational relevance.**
