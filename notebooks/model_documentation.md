# SpliceVarMech — Comprehensive Model Documentation

## For Reviewers: Understanding Every Component

This document explains the complete SpliceVarMech framework from both **biological** and **computational** perspectives. Every mathematical definition, biological assumption, and architectural choice is documented here.

---

## Table of Contents

1. [Biological Foundation](#1-biological-foundation)
2. [Module 1: Discrete Diffusion Model (D3PM)](#2-module-1-discrete-diffusion-model)
3. [Module 2: Bayesian Causal Inference](#3-module-2-bayesian-causal-inference)
4. [Module 3: Explainable AI](#4-module-3-explainable-ai)
5. [Training Strategy](#5-training-strategy)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Mathematical Definitions](#7-mathematical-definitions)

---

## 1. Biological Foundation

### 1.1 RNA Splicing — The Process Our Model Learns

**Pre-mRNA splicing** is the removal of introns (non-coding) and joining of exons (coding) to produce mature mRNA. The spliceosome (a molecular machine) recognizes splice sites through:

| Signal | Sequence | Location | Our DAG Node |
|--------|----------|----------|-------------|
| **5' donor** | GU (GT in DNA) | Exon-intron boundary | **S** (Splice strength) |
| **3' acceptor** | AG | Intron-exon boundary | **S** |
| **Branch point** | YNYURAY | 18-44bp upstream of 3' | **S** |
| **Polypyrimidine tract** | Y-rich | 5-40bp upstream of 3' | **S** |
| **ESE motifs** | SR protein binding | Within exons | **E** (ESE/ESS balance) |
| **ISE motifs** | TIA-1, CELF binding | Within introns | **I** (ISE/ISS impact) |

### 1.2 Non-Canonical Splicing Variants (NCSVs)

NCSVs are variants **outside the canonical ±1/±2 splice sites** that still disrupt splicing. They cause disease through:

- **ESE disruption** (exonic): Missense variants destroy SR protein binding → exon skipping
- **ISE disruption** (intronic +7 to +50): Variants weaken auxiliary splice signals → exon skipping or intron retention
- **Cryptic site activation**: Deep intronic variants create new GT/AG → pseudoexon inclusion

**Our TEX11 case** (c.1156+16G>T) falls in the ISE region (+16), where TIA-1 and CELF proteins bind to stabilize U1 snRNP. A G→T transversion may disrupt this binding.

### 1.3 Why 47% of Variants Remain VUS

From the WES/WGS systematic review (Study 3, 2025):
- 143 genes identified in male infertility
- **47% average VUS burden** across studies
- Only **34% functionally validated**
- Standard analysis misses 27 NCSVs classified as "likely benign" (Study 6)

**Our framework's contribution:** Provide computational functional evidence for reclassification.

---

## 2. Module 1: Discrete Diffusion Model

### 2.1 Why Discrete Diffusion (Not Continuous)

DNA/RNA sequences are **discrete** — each position is one of {A, C, G, T}. Continuous diffusion (DDPM, Dall-E) operates in continuous ℝⁿ space. We use **D3PM** (Discrete Denoising Diffusion Probabilistic Model, Austin et al. NeurIPS 2021) which operates directly on categorical tokens.

**Advantage:** No embedding/de-embedding roundtrip — the model reasons about nucleotide identity directly (e.g., "GT" = donor splice signal).

### 2.2 The Absorbing State Noise Schedule

**Forward process** (corruption): Progressively replace nucleotides with [MASK] tokens.

At timestep *t*, each non-PAD token has probability β(t) of being masked:

```
q(x_t | x_0) = ∏ᵢ q(xᵢ_t | xᵢ_0)

where q(xᵢ_t = MASK | xᵢ_0 = k) = 1 - ᾱ(t)
      q(xᵢ_t = k | xᵢ_0 = k)    = ᾱ(t)
```

**ᾱ(t)** is the cumulative survival probability using a **cosine schedule**:

```
ᾱ(t) = cos²(t/T · π/2)

β(t) = 1 - ᾱ(t)/ᾱ(t-1)     (clipped to [0.0001, 0.999])
```

**Intuition:** At t=0, sequence is clean. At t=T, sequence is fully [MASK]ed. The cosine schedule provides gentle corruption early and aggressive corruption late.

### 2.3 BiologicalDiffusionModel — Spliceosome-Inspired Architecture

**Goal:** Learn p_θ(x_0 | x_t, WT_context, MUT_context, variant_pos, t) — predict the mature mRNA from the corrupted sequence, conditioned on the biological comparison between wild-type and mutant pre-mRNA.

**Key Innovation:** The model simulates how the spliceosome detects and responds to DNA variants. Instead of processing a single context (which made 1-nucleotide changes invisible — 50.3% BalAcc in the old model), it receives BOTH the WT and MUT contexts and explicitly compares them.

**Architecture: 4 Biological Components**

```
Input:  WT_context (wild-type pre-mRNA, [batch, ctx_len])
        MUT_context (mutant pre-mRNA, [batch, ctx_len])
        variant_pos (position of mutation, [batch])
        ref_token / alt_token (ref→alt alleles, [batch])
        x_t (corrupted mRNA, [batch, seq_len])
        t (timestep, [batch])

Component 1 — VARIANT HIGHLIGHT:
  • Learned variant marker embedding at the mutated position
  • Gaussian spread: nearby positions also get signal (σ=3bp, tighter for sharper positional signal)
  • Substitution type embedding: 16 types (G→T, A→C, etc.)
  → Solves: model knows WHERE and WHAT changed

Component 2 — MULTI-SCALE FEATURE EXTRACTION:
  Like the spliceosome's multi-level signal recognition:
  • Local CNN (kernel=5):   U1/U2 binding at ±2bp splice sites
  • Regional CNN (kernel=15): SR/hnRNP binding at ESE/ESS elements
  • Structural CNN (kernel=51): Branch point + polypyrimidine tract
  • Learned gating: each scale's contribution is position-dependent
  → Solves: captures biological signals at appropriate spatial scales

Component 3 — DUAL-STREAM ENCODER:
  Like biological quality control (expected vs actual binding):
  • Shared transformer encoder processes BOTH WT and MUT embeddings
  • Cross-attention: MUT queries attend to WT keys/values
    → Discovers WHAT changed between WT and MUT
  • Variant impact: element-wise difference (MUT_encoded - WT_encoded)
  • Fusion: concat(cross_attended, variant_impact) → fused context
  → Solves: explicit WT/MUT comparison for variant sensitivity

Component 4 — CONDITIONAL mRNA DECODER:
  • Embeds corrupted mRNA tokens + timestep + tissue type
  • Self-conditioning from previous step's prediction
  • Cross-attends to fused biological context
  • Projects to logits [batch, seq_len, vocab_size=7]

Output: Per-position probability over {PAD, A, C, G, T, MASK, SEP}
```

**Contrastive Learning (Component 5):**
During training, a contrastive loss enforces that:
- Disruptive variants → WT and MUT representations are pushed APART
- Benign variants → WT and MUT representations stay CLOSE

```
L_contrastive = is_disruptive × max(0, margin - distance)²
              + (1 - is_disruptive) × distance²
```

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 256 | Balance between capacity and training speed |
| n_heads | 8 | Multi-head attention for diverse splice features |
| n_encoder_layers | 3 | Shared WT/MUT encoder depth |
| n_decoder_layers | 6 | mRNA decoder needs more capacity |
| d_ff | 1024 | Standard 4× expansion |
| kernel_local | 5 | ±2bp splice site recognition |
| kernel_regional | 15 | ±7bp ESE/ESS regulation |
| kernel_structural | 51 | ±25bp branch point / PPT |
| contrastive_weight | 0.3 | λ for L_contrastive |
| contrastive_margin | 0.5 | Margin for disruptive variants (SimCLR/SupCon recommendation) |
| n_timesteps | 100 | Sufficient for gradual denoising |
| max_seq_len | 512 | Covers exon-intron-exon regions |
| self_cond_prob | 0.5 | Self-conditioning applied 50% of training |
| ema_decay | 0.9999 | EMA weight smoothing for inference |

**Total parameters:** ~12-15 million (larger due to dual-stream + multi-scale)

### 2.4 Training Loss

Combined diffusion + contrastive loss:

```
L_total = L_diffusion + λ × L_contrastive

L_diffusion = 𝔼_{t,x_0} [ -∑ᵢ∈masked log p_θ(xᵢ_0 | x_t, WT_ctx, MUT_ctx, var_pos, t) ]

L_contrastive = pushes WT/MUT representations apart when variant is disruptive
```

**Why two losses?**
- L_diffusion teaches the model to generate correct mRNA sequences
- L_contrastive teaches the model to differentiate WT from MUT contexts — the critical capability the old model lacked

**Training data is paired:** Each example includes (WT_context, MUT_context, variant_pos, ref, alt, target_mRNA, label). During pre-training, synthetic paired junctions teach which mutations matter. During fine-tuning, real gold-standard variants + MFASS experimentally validated data provide ground truth.

### 2.5 Sampling (Inference) — Proper D3PM Reverse Process

**Iterative denoising** from fully masked → predicted mRNA using the theoretically correct absorbing-state reverse transitions:

```
x_self_cond = None  # Self-conditioning state

For t = T-1, T-2, ..., 0:
  1. Predict logits = p_θ(x_0 | x_t, context, t, x_self_cond)
  2. Mask out non-nucleotide tokens (PAD, MASK, SEP) from logits
  3. Compute probs = softmax(logits / temperature)
  4. Update x_self_cond = probs  (for next step's self-conditioning)
  5. For each MASKED position independently:
     - P(unmask) = (ᾱ_{t-1} - ᾱ_t) / (1 - ᾱ_t)
     - With probability P(unmask): sample token from probs
     - Otherwise: stay masked
  6. At t=0: unmask all remaining positions
```

**Key design decisions:**
- **Proper D3PM reverse**: Each position independently decides to unmask based on the noise schedule posterior, NOT confidence-based top-k. This is the theoretically correct absorbing-state reverse process (Austin et al. 2021).
- **Self-conditioning**: Each step's prediction feeds back to the next step, improving sequence coherence.
- **Non-nucleotide masking**: PAD, MASK, SEP tokens are blocked from being generated.
- **Fully vectorized**: No per-sample Python loops — all operations are batched on GPU.

### 2.6 Mechanism Classification from Generated Sequences

After generating N samples, each is classified by comparing to the expected wild-type mRNA:

| Generated vs WT | Length Ratio | Classification |
|----------------|--------------|----------------|
| Similar length, >80% identity | ~1.0 | **Normal** |
| Much shorter (>30% loss) | <0.7 | **Exon skipping** |
| Somewhat shorter (10-30% loss) | 0.7-0.9 | **Partial deletion** |
| Longer than WT | >1.1 | **Intron retention** |
| Same length, low identity | ~1.0, <70% sim | **Cryptic splice site** |

**Probabilistic penetrance:** The distribution over N samples estimates the transcript mixture. E.g., 75% exon skip + 20% normal + 5% retention suggests partial penetrance.

---

## 3. Module 2: Bayesian Causal Inference

### 3.1 Structural Causal Model (SCM)

We define a **Directed Acyclic Graph (DAG)** encoding the biology of splice site recognition:

```
Causal Graph:
  V (Variant) → S (Splice strength)
  V → E (ESE/ESS balance)  
  V → I (ISE/ISS impact)
  P (Position) → S
  P → I
  C (Conservation) → O (Outcome)
  S → O
  E → O
  I → O
  D (Diffusion output) → O
```

**Pearl's do-calculus** allows us to answer the interventional question:

```
P(O = disrupted | do(V = G→T at +16))
```

This is different from the observational P(O | V) because the do-operator removes confounding.

### 3.2 Bayesian Logistic Regression

The outcome O is modeled as:

```
logit(P(O=1)) = α + β_S·S + β_E·E + β_I·I + β_C·C + β_P·P + β_D·D
```

where:
- **α** = intercept (prior: Normal(0, 1.5))
- **β_k** = coefficient for feature k
- **P(O=1)** = probability of splice disruption

### 3.3 Priors — Encoding Biological Knowledge

**Hierarchical shrinkage prior** on coefficients:

```
σ_β ~ HalfNormal(0.5)     # Global shrinkage scale
β_k ~ Normal(0, σ_β)       # Per-feature coefficient
```

**Why hierarchical?** With N=31 observations and 16 features, we need regularization. The hierarchical prior learns the appropriate shrinkage level from data — unimportant features get pushed to zero, important ones can be large.

**Interpretation:** This is a Bayesian analog of L2 regularization, but with the scale learned (not fixed).

### 3.4 Class-Balanced Weighted Likelihood

**The imbalance problem:** 23 positives vs 8 negatives. An unweighted model predicts "positive" for everything (100% sensitivity, 12.5% specificity).

**Solution:** Weight the log-likelihood by inverse class frequency:

```
w_pos = N / (2 · N_pos)    # = 31/(2·23) = 0.674
w_neg = N / (2 · N_neg)    # = 31/(2·8)  = 1.938

Weighted LL = Σᵢ wᵢ · [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]
```

This ensures each class contributes equally to the loss, regardless of count.

### 3.5 MCMC Inference

We use **NUTS** (No-U-Turn Sampler, Hoffman & Gelman 2014) via PyMC:

```
n_chains = 2
n_tune = 1000        # Warmup/adaptation steps (discarded)
n_samples = 2000     # Posterior samples per chain
target_accept = 0.95  # Higher for complex models
```

**Convergence diagnostics:**
- **r̂ (R-hat):** Measures between-chain vs within-chain variance. r̂ ≤ 1.01 = converged.
- **ESS (Effective Sample Size):** Number of independent samples. ESS > 400 = reliable.
- **Divergences:** 0 ideal. >0 suggests geometry problems.

### 3.6 Posterior Interpretation

From the posterior, we extract:

```
P(disruption | evidence) = mean of p_disruption across all posterior samples
95% CI = [2.5th percentile, 97.5th percentile]
```

**Clinical meaning:** "We estimate 87% probability of splice disruption, with 95% credible interval [0.72, 0.95]." The width of the CI reflects honest uncertainty.

### 3.7 Counterfactual Reasoning

**Counterfactual query:** "If the variant were wild-type, would splicing be normal?"

```
P(O = normal | do(V = G))     # Restore wild-type
```

If this probability is high → the variant IS causal for the phenotype.

**Implementation:** Generate diffusion samples from wild-type context, compare to mutant context. The difference = causal effect.

---

## 4. Module 3: Explainable AI

### 4.1 Sequence Attribution (Gradient-Based)

**Method:** Compute gradient of the loss w.r.t. the context embedding at each position.

```
Attribution(position i) = ||∂L/∂e_i||₂
```

where e_i is the embedding vector at position i.

**Interpretation:** Positions with high gradient norm = most influential for the prediction. For TEX11, we expect high attribution at position +16 and surrounding ISE motifs (+14 to +18).

**Reference:** Adapted from Integrated Gradients (Sundararajan et al., ICML 2017) for discrete tokens.

### 4.2 Causal Path Analysis

From the Bayesian SCM posterior, we compute the **path-specific causal effect** through each pathway:

```
Path V → I → O:  weight = |β_I| / Σ|β_k|
Path V → S → O:  weight = |β_S| / Σ|β_k|
Path V → E → O:  weight = |β_E| / Σ|β_k|
```

**Modified by position:** Intronic variants (+16) get higher weight for V→I→O (ISE disruption). Exonic variants get higher weight for V→E→O (ESE disruption).

### 4.3 Clinical Confidence Grading

```
Score = f(posterior_p) + f(CI_width) + f(n_samples) + f(tool_agreement)

HIGH:     score ≥ 7  (posterior > 0.85, narrow CI, many samples)
MODERATE: score ≥ 4  (posterior 0.6-0.85, moderate CI)
LOW:      score < 4  (posterior < 0.6, wide CI)
```

### 4.4 ACMG Criteria Mapping

Our outputs map to ACMG/AMP variant classification criteria:

| Criterion | Our Evidence | Threshold |
|-----------|-------------|-----------|
| **PP3** (computational) | Posterior P(disruption) | >0.7 → PP3_Strong |
| **PS3** (functional) | Diffusion aberrant fraction | >0.5 → PS3_Moderate |
| **PM2** (absent in pop.) | gnomAD frequency | <0.001% → PM2_Supporting |
| **PP4** (phenotype) | Clinical consistency | Azoospermia + TEX11 → PP4 |

**Classification rules:**
- ≥2 Strong → **Pathogenic**
- 1 Strong + ≥1 Moderate → **Likely Pathogenic**
- ≥2 Moderate → **Likely Pathogenic**
- 1 Moderate → **VUS (leaning pathogenic)**

---

## 5. Training Strategy

### 5.1 Two-Stage Training

**Stage 1: Pre-training** (learns general splicing rules)

| Parameter | Value |
|-----------|-------|
| Data | GENCODE v44 real splice junctions (100K exon-intron-exon triplets from GRCh38) |
| Tissue labels | GTEx v8 median TPM per tissue (55,374 genes → 10 tissue categories) |
| Input | Pre-mRNA (exon + GT...AG + exon) |
| Target | Correctly spliced mRNA (exon + exon) |
| Conditioning | tissue_id from GTEx (e.g. TEX11→testis, ALB→liver, MYH7→heart) |
| Objective | Cross-entropy on masked positions |
| Schedule | Cosine learning rate, AdamW |

**What the model learns:**
- GT/AG = splice boundaries (remove intron)
- ESE motifs in exons → keep exon
- Polypyrimidine tract → acceptor site signal
- Branch point sequence → splice site strength
- Tissue-specific splicing patterns (testis genes splice differently from liver genes)

**Stage 2: Fine-tuning** (learns variant effects)

| Parameter | Value |
|-----------|-------|
| Primary data | 40 S7 positives + 14 S2 negatives = 54 |
| Study 6 data | +183 splice variants (with synthetic targets) |
| ClinVar NCSV | +400 (200 pathogenic + 200 benign at ±3-50) |
| gnomAD benign | +500 common intronic variants (AF>1%, ±3-50) as negatives |
| Total (before aug) | ~1,137 examples |
| Augmentation | 8× per variant + synthetic variants |
| Total (after aug) | ~2,400+ examples |
| Learning rate | 5e-5 (lower than pre-training) |

**gnomAD integration:** Common intronic variants from gnomAD v4.1 (AF>1% at positions ±3-50) are added as high-confidence benign negatives. For each gnomAD variant, the pipeline extracts real hg38 genomic context when possible, introducing the variant into the pre-mRNA and expecting normal spliced mRNA output. This dramatically increases the negative training examples from 14 (S2 only) to 500+, teaching the diffusion model what "benign at non-canonical positions" looks like — critical because our S2 negatives are biased (selected for HIGH tool scores).

### 5.2 Data Augmentation Strategies

1. **Nucleotide substitution** (1-3 random positions in non-critical regions)
2. **Synthetic variant generation** (mutate donor/acceptor in synthetic sequences)
3. **Mechanism-conditioned generation** (generate exon skipping / intron retention / partial deletion)
4. **External data integration** (Study 6 splice variants + ClinVar + gnomAD benign negatives)

### 5.2.1 Cross-Dataset Evaluation with MFASS

**MFASS** (Cheung et al., Molecular Cell 2019) provides 27,733 experimentally tested splice variants with measured exon inclusion ratios (PSI). The near-canonical subset (±3-20) contains 13,875 variants with 493 LOF (3.6% disruption rate), directly matching our NCSV target range. At position ±16 (same as TEX11 c.963+16): 834 variants, 27 LOF (3.2% disruption rate).

MFASS is used for **evaluation only** (not training) — it provides independent experimental ground truth to validate our model's predictions at the exact intronic positions where our framework operates.

### 5.3 Training Best Practices

**Optimizer:** AdamW (Adam with decoupled weight decay)

```
θ_{t+1} = θ_t - η · (m̂_t / (√v̂_t + ε) + λ · θ_t)

where:
  m̂_t = bias-corrected first moment (moving average of gradients)
  v̂_t = bias-corrected second moment (moving average of squared gradients)
  η = learning rate
  λ = weight decay (0.01)
  ε = 1e-8
```

**Learning rate schedule:** Linear warmup + cosine annealing

```
Warmup (first 100 steps):
  η(t) = η_max · (t / warmup_steps)    # Linear ramp from 0.01× to 1×

Cosine annealing (remaining steps):
  η(t) = η_min + (η_max - η_min) · (1 + cos(πt/T)) / 2
```

**Gradient clipping:** max_norm = 1.0 (prevents exploding gradients in the transformer)

**EMA (Exponential Moving Average):** decay = 0.9999

```
θ_ema(t) = 0.9999 · θ_ema(t-1) + 0.0001 · θ(t)
```

Updated every training step. EMA weights are used for inference (sampling, likelihood estimation). This stabilizes generation quality (standard practice since Ho et al. 2020).

**Validation + Early Stopping:**

- 15% of data held out for validation (random split, seeded)
- Validation loss computed every epoch (model in eval mode, no self-conditioning)
- Early stopping with patience = 10 epochs (stop if val loss doesn't improve)
- Prevents overfitting on the limited ~1,900 fine-tuning examples

### 5.4 Self-Conditioning During Training

50% of training steps use **self-conditioning** (Chen et al. 2022, Sahoo et al. 2024):

```
With probability 0.5:
  1. Forward pass (no gradient): logits_1 = model(x_t, t, context)
  2. x_self_cond = softmax(logits_1).detach()
  3. Forward pass (with gradient): logits_2 = model(x_t, t, context, x_self_cond)
  4. Loss on logits_2

With probability 0.5:
  1. Forward pass (with gradient): logits = model(x_t, t, context, x_self_cond=None)
  2. Loss on logits
```

This teaches the model to refine its predictions iteratively, which is exactly what happens during sampling where each step receives the previous step's prediction.

---

## 6. Evaluation Metrics

### 6.1 Classification Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Accuracy** | (TP+TN) / N | Overall correctness (biased by class imbalance) |
| **Balanced Accuracy** | (Sensitivity + Specificity) / 2 | Class-balanced correctness |
| **Sensitivity (TPR)** | TP / (TP+FN) | Detection of true positives |
| **Specificity (TNR)** | TN / (TN+FP) | Rejection of false positives |
| **MCC** | (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Balanced measure even with imbalance |
| **AUROC** | Area under ROC curve | Discrimination across all thresholds |
| **AUPRC** | Area under Precision-Recall curve | Better than AUROC for imbalanced data |

### 6.2 Generative Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Mechanism accuracy** | Correct mechanism type (exon skip/intron retain/etc.) | >70% |
| **Sequence identity** | Nucleotide match vs validated aberrant mRNA | >90% |
| **Aberrant fraction** | Fraction of non-normal generated samples | Calibrated |

### 6.3 Calibration

**Calibration error:** Do X% credible intervals contain the true outcome X% of the time?

```
Calibration error = (1/K) · Σ_k |observed_frequency_k - predicted_frequency_k|
```

Target: <0.10

### 6.4 Effect Size Statistics

| Statistic | Formula | Interpretation |
|-----------|---------|----------------|
| **Cohen's d** | (μ₁ - μ₂) / s_pooled | <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large |
| **Mann-Whitney U** | Rank-based non-parametric test | p < 0.05 = significantly different |

---

## 7. Mathematical Definitions

### 7.1 Discrete Diffusion (D3PM)

**State space:** Finite alphabet Σ = {PAD, A, C, G, T, MASK, SEP}, |Σ| = 7

**Forward transition matrix** at step t:

```
Q_t[i,j] = β_t · 𝟙[j = MASK] + (1 - β_t) · 𝟙[i = j]
```

**Cumulative transition:**

```
Q̄_t = Q_1 · Q_2 · ... · Q_t
```

For absorbing state schedule: Q̄_t[i, MASK] = 1 - ᾱ_t, Q̄_t[i, i] = ᾱ_t

**Reverse transition** (parameterized by neural network):

```
p_θ(x_{t-1} | x_t) ∝ q(x_t | x_{t-1}) · p_θ(x_0 | x_t)
```

### 7.2 Bayesian Inference

**Bayes' theorem:**

```
P(θ | data) = P(data | θ) · P(θ) / P(data)

Posterior ∝ Likelihood × Prior
```

**Log-posterior (what MCMC samples from):**

```
log P(θ | data) = Σᵢ log P(yᵢ | xᵢ, θ) + log P(θ) + const

= Σᵢ [yᵢ · log σ(xᵢ·β + α) + (1-yᵢ) · log(1 - σ(xᵢ·β + α))]   # Likelihood
  + log N(α | 0, 1.5)                                                  # Intercept prior
  + Σ_k log N(β_k | 0, σ_β)                                           # Coefficient priors
  + log HalfNormal(σ_β | 0.5)                                         # Shrinkage prior
```

### 7.3 NUTS Sampler

**No-U-Turn Sampler** (Hoffman & Gelman, 2014) — adaptive HMC:

1. Propose new state by simulating Hamiltonian dynamics
2. Automatically adapt trajectory length (no manual tuning)
3. Accept/reject based on Metropolis criterion
4. During warmup: adapt step size and mass matrix

**Key properties:**
- Explores posterior efficiently (no random walk)
- Automatically tunes hyperparameters during warmup
- Detects pathologies (divergences = geometry problems)

### 7.4 Credible Intervals

**95% Highest Density Interval (HDI):**

The narrowest interval containing 95% of the posterior mass.

```
HDI = [θ_lo, θ_hi] such that:
  ∫_{θ_lo}^{θ_hi} P(θ | data) dθ = 0.95
  P(θ_lo | data) = P(θ_hi | data)     # Equal density at boundaries
```

**Equal-tail interval** (simpler): [2.5th percentile, 97.5th percentile]

### 7.5 Causal Inference Definitions

**Intervention (do-operator):**

```
P(Y | do(X = x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)
```

This differs from conditioning P(Y | X=x) because do(X=x) removes incoming edges to X in the DAG.

**Counterfactual:**

Given observed evidence (U = u):
1. **Abduction:** Infer P(U | evidence)
2. **Action:** Set do(X = x') in the modified model
3. **Prediction:** Compute P(Y | do(X=x'), U=u)

---


## 8. Evaluation Rigor

### 8.1 Formal Leakage Analysis

We verify no information leakage exists between features and labels:

| Check | Result |
|-------|--------|
| **Source independence** | Features from S1 (computational predictions), labels from S7/S2 (experimental assays) — independent modalities |
| **No label in features** | Splice tool scores computed from DNA sequence only, not from experimental mRNA outcome |
| **Temporal separation** | Features are static (variant DNA), labels are post-experiment (mRNA assay) |
| **Selection bias** | Documented — negatives selected because tools predicted disruption (conservative evaluation) |

### 8.2 Calibration (ECE / Reliability Diagrams)

Expected Calibration Error (ECE) measures whether predicted probabilities match observed frequencies:

```
ECE = Σ_b (|B_b|/N) · |accuracy(B_b) - confidence(B_b)|
```

A well-calibrated Bayesian posterior should have ECE < 0.05. We report ECE, MCE (Maximum Calibration Error), and Brier score for the posterior P(disruption).

### 8.3 Cross-Dataset Generalization

To demonstrate the model generalizes beyond the training distribution:

| Train Set | Test Set | Generalization Type |
|-----------|----------|-------------------|
| S7+S2 (primary) | S7+S2 (LOO-CV) | In-distribution |
| S7+S2 (primary) | Study 6 genes | Cross-cohort (independent lab) |
| S7+S2 (primary) | Study 4 TESE outcomes | Cross-outcome (clinical endpoint) |
| ClinVar splice variants (394K variants, 7K genes) | S7+S2 | Cross-database ✅ |
| S7+S2 + ClinVar | BRCA1 SGE (3,644 variants) | Cross-dataset (independent gene) ✅ |
| S7+S2 + ClinVar | MaPSy (231 variants) | Cross-assay (independent assay) ✅ |

### 8.4 Cold-Gene Evaluation

Leave-One-Gene-Out CV: hold out all variants from one gene, train on the rest. Tests whether the model can predict splice disruption for completely unseen genes.

### 8.5 Feature-Group Ablation

Systematic removal of feature groups to quantify each group's contribution:

| Feature Group | Features | What It Captures |
|--------------|----------|-----------------|
| Splice strength | SpliceAI, MaxEntScan, GeneSplicer, dbscSNV | Direct splice site disruption |
| Conservation | CADDsplice | Evolutionary constraint |
| ESE/ESS | ESRseq, Spliceogen | Exonic regulatory elements |
| Tissue expression | dpsi_max_tissue, dpsi_zscore | Tissue-specific splicing |
| Ensemble tools | Squirls, Kipoi, MMSplice, SPiCE | Combined predictions |
| Position | Variant position relative to splice site | Spatial context |

### 8.6 XAI Stability

Attribution stability is measured using Spearman rank correlation (ρ) across multiple runs with different random seeds. Stable attributions have ρ > 0.8. Top-k position consistency is measured using Jaccard similarity.

### 8.7 Component Training (Frozen vs Fine-tuned)

| Component | Pre-training | Fine-tuning |
|-----------|-------------|-------------|
| Context encoder (3 layers) | Trained | All layers fine-tuned |
| Decoder (6 layers) | Trained | All layers fine-tuned |
| Tissue embedding | Trained | Fine-tuned |
| Output projection | Trained | Fine-tuned |
| Bayesian model (PyMC) | N/A | Trained from scratch (MCMC) |

**Rationale:** With ~9.2M parameters and ~1,900 fine-tuning examples, full fine-tuning is appropriate. Freezing would reduce capacity for learning variant-specific features.

---

## Appendix A: Comparison with Existing Approaches

| Approach | SpliceAI | DNABERT-2 | Pangolin | **SpliceVarMech** |
|----------|----------|-----------|---------|-------------------|
| Architecture | ResNet | BERT | Multi-task DL | D3PM + SCM |
| Output | Score (0-1) | Classification | Score per tissue | **mRNA sequence + probability** |
| Mechanism | ❌ | ❌ | ❌ | **✅ Generated** |
| Uncertainty | ❌ | ❌ | ❌ | **✅ Bayesian CI** |
| Causal reasoning | ❌ | ❌ | ❌ | **✅ do-calculus** |
| Explainability | ❌ | ❌ | ❌ | **✅ Attribution + paths** |
| Tissue-aware | ❌ | ❌ | ✅ (GTEx PSI) | **✅ (GTEx + learned embedding)** |
| NCSV performance | Poor (>+10bp) | Unknown | Unknown | **Designed for NCSVs** |

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **NCSV** | Non-Canonical Splicing Variant — variant outside ±1/±2 that disrupts splicing |
| **VUS** | Variant of Uncertain Significance — cannot be classified as pathogenic or benign |
| **NOA** | Non-Obstructive Azoospermia — no sperm due to spermatogenic failure |
| **TESE** | Testicular Sperm Extraction — surgical procedure to retrieve sperm |
| **D3PM** | Discrete Denoising Diffusion Probabilistic Model |
| **SCM** | Structural Causal Model |
| **DAG** | Directed Acyclic Graph |
| **MCMC** | Markov Chain Monte Carlo |
| **NUTS** | No-U-Turn Sampler |
| **ESE** | Exonic Splicing Enhancer — motif that promotes exon inclusion |
| **ISE** | Intronic Splicing Enhancer — motif that promotes splice site recognition |
| **NMD** | Nonsense-Mediated Decay — mRNA surveillance that degrades aberrant transcripts |
| **ACMG** | American College of Medical Genetics and Genomics |
| **HDI** | Highest Density Interval |
| **MCC** | Matthews Correlation Coefficient |
| **AUROC** | Area Under the Receiver Operating Characteristic Curve |
| **AUPRC** | Area Under the Precision-Recall Curve |
| **MFASS** | Multiplexed Functional Assay of Splice Sequences (Cheung et al. 2019) |
| **ELBO** | Evidence Lower BOund — variational bound on log-likelihood |
| **NLL** | Negative Log-Likelihood |

---

## Appendix C: External Data Sources

### C.1 ClinVar Non-Canonical Splice Variants (Training Augmentation)

- **Source**: NCBI ClinVar database (Landrum et al., NAR 2024)
- **Download**: `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz`
- **Content**: 351,593 intronic splice variants at positions ±3 to ±50bp
  - 17,577 Pathogenic/Likely Pathogenic
  - 334,016 Benign/Likely Benign
- **Role**: Training augmentation — expands Bayesian model training from N=31 to N=1,031+
- **Parser**: `src/data/clinvar.py` → `parse_clinvar_splice_variants()`

### C.2 BRCA1 SGE — Saturation Genome Editing (Cross-Dataset Evaluation)

- **Source**: Findlay GM et al., "Accurate classification of BRCA1 variants with saturation genome editing", Nature 562:217-222, 2018
- **DOI**: 10.1038/s41586-018-0461-z
- **Download**: Nature Supplementary Table 2 (see README for curl command)
- **Content**: 3,644 variants with experimental functional classification (excluding INT)
  - 823 Loss-of-Function (LOF) variants (label=1)
  - 2,821 Functional (FUNC) variants (label=0)
  - 249 Intermediate (INT) excluded by default
- **Splice-relevant subset**: 1,014 variants (Canonical splice + Splice region + Intronic)
  - 135 canonical ±1/2 (95% LOF rate)
  - 490 near-canonical ±3-10 (21% LOF rate)
  - 354 deep intronic >±10 (1% LOF rate)
- **Ground truth**: Saturation genome editing in HAP1 cells — gold-standard functional assay
- **Role**: Cross-dataset evaluation — completely independent gene (BRCA1 vs male infertility genes)
- **Parser**: `src/data/brca1_sge.py` → `load_brca1_sge_variants()`

### C.3 MaPSy — Massively Parallel Splicing Assay (Cross-Dataset Evaluation)

- **Source**: Soemedi R et al., "Pathogenic variants that alter protein code often disrupt splicing", Nature Genetics 49:848-855, 2017
- **DOI**: 10.1038/ng.3837
- **Download**: Nature Genetics Supplementary Table 1 (see README for curl command)
- **Content**: 231 exonic variants tested in massively parallel minigene splicing reporter
  - 8 Exonic Splice Mutations (ESM=1, label=1)
  - 223 Non-ESM variants (ESM=0, label=0)
- **Ground truth**: Experimental exon inclusion ratio measurement in minigene constructs
- **Role**: Cross-dataset evaluation — independent assay platform, different genes
- **Parser**: `src/data/mapsy.py` → `load_mapsy_variants()`

### C.4 gnomAD v4.1 — Benign Intronic Negatives (Training Augmentation)

- **Source**: gnomAD v4.1 (Karczewski et al., Nature 2024)
- **Fetch method**: GraphQL API queries (`scripts/fetch_gnomad_api.py`)
  - Queries 190 genes (infertility, cancer, cardiac, neuro, metabolic)
  - Exponential backoff on 429 rate limits (30s→60s→120s→240s→300s, 5 retries)
  - 2s base delay between requests; consecutive failure detection
  - Creates fresh HTTP request objects per retry (avoids stream-reuse crash)
- **Filtering**: AF > 1%, SNV only, splice-region consequence, intronic position ±3 to ±50
- **Output**: `data/external/gnomad_benign_splice_region.tsv`
- **Content**: ~500-2,000 common intronic splice-region variants
- **Role**: Training augmentation — high-confidence benign negatives. Dramatically increases negatives from 14 (S2 only) to 500+, teaching the model what "benign at non-canonical positions" looks like
- **Parser**: `src/data/gnomad.py` → `load_gnomad_benign_negatives()`

### C.5 GRCh38 Genomic Context Extraction

- **Module**: `src/data/hg38_context.py`
- **Purpose**: Extract real exon-intron-exon pre-mRNA contexts from GRCh38 reference genome
- **Functions**:
  - `extract_splice_context(gene, hgvs)` → `SpliceContext` (WT + mutant pre-mRNA, WT mRNA)
  - `extract_tex11_context()` → TEX11 c.1156+16G>T specific context
  - `extract_gold_standard_contexts()` → All S7/S2 variants' real genomic contexts
  - `get_male_infertility_gene_regions()` → Gene coordinates for 40+ infertility genes
- **Requires**: GRCh38 FASTA + GENCODE GTF (auto-detected in `data/external/`)
- **Fallback**: Synthetic sequences when reference genome not available

### C.6 GTEx v8 Tissue-Specific Expression

- **Module**: `src/data/gtex.py` → `GTExTissueMapper`
- **Purpose**: Map genes to dominant tissue expression for tissue-conditioned diffusion
- **Data**: GTEx v8 median TPM across 54 tissues → 10 tissue categories
  - `{universal, testis, brain, liver, heart, muscle, blood, kidney, lung, ovary}`
- **Usage**: During pre-training, each GENCODE splice junction is labeled with its tissue type
  - TEX11 → testis (testis-specific expression)
  - ALB → liver, MYH7 → heart, etc.
- **Relevance**: Testis has the most complex transcriptome; tissue-specific splicing factors (RBMXL2, T-STAR) create unique splice patterns that general models miss

### C.7 Cross-Dataset Evaluation Design

| Train Set | Test Set | Type |
|-----------|----------|------|
| Primary (S7+S2, N=31) + ClinVar NCSVs | BRCA1 SGE (3,644 variants) | Cross-dataset (independent gene) |
| Primary (S7+S2, N=31) + ClinVar NCSVs | MaPSy (231 variants) | Cross-assay (independent assay) |
| Primary (S7+S2, N=31) + ClinVar NCSVs | MFASS near-canonical (13,875 variants) | Cross-dataset (experimental ground truth) |
| Primary (S7+S2) | LOGO (leave-one-gene-out) | Within-dataset |

**Scientific justification**: Training uses male infertility genes (primary) and ClinVar splice variants. Testing uses breast cancer gene (BRCA1 SGE — completely different disease, gene, and lab), a cross-gene minigene assay (MaPSy — different experimental platform), and MFASS massively parallel splice assay (27,733 variants with measured exon inclusion ratios). This demonstrates domain-agnostic generalization of splice variant classification.

**Implementation**: `python main.py --eval` runs the full cross-dataset evaluation pipeline:
1. Trains Bayesian model on primary + ClinVar augmented data
2. Applies learned coefficients to BRCA1 SGE test data (position-stratified evaluation)
3. Applies learned coefficients to MaPSy test data
4. Reports AUROC, AUPRC, balanced accuracy, sensitivity, specificity per dataset

---

## Appendix D: Likelihood-Ratio Disruption Scoring

### D.1 Scientific Method

The diffusion model computes a **log-likelihood ratio** to quantify how much a variant disrupts normal splicing:

```
disruption_score = NLL(WT_mRNA | Mut_context) - NLL(WT_mRNA | WT_context)
```

Where:
- `NLL(x|c)` = negative log-likelihood of target mRNA `x` given pre-mRNA context `c`
- Estimated via the ELBO (Evidence Lower Bound) of the D3PM diffusion model
- Averaged across multiple timesteps for numerical stability

### D.2 Interpretation

- **disruption_score > 0**: The variant makes it harder for the model to reconstruct normal mRNA → splice disruption
- **disruption_score ≈ 0**: No effect on the model's reconstruction → normal splicing
- **disruption_score < 0**: The variant makes reconstruction easier (unlikely, indicates noise)

### D.3 Implementation

```python
from src.diffusion.model import BiologicalDiffusionModel

model = BiologicalDiffusionModel(config)
result = model.compute_disruption_score(
    wt_mrna=wt_tokens,       # Expected normal mRNA
    wt_context=wt_ctx,        # WT pre-mRNA context
    mut_context=mut_ctx,       # Mutant pre-mRNA context
    n_timestep_samples=20,    # ELBO averaging
)
# result['disruption_score'] = float (positive = disrupted)
# result['causal_effect'] = sigmoid(disruption_score)
```

### D.4 Basis in Literature

- Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces" (NeurIPS 2021) — D3PM ELBO
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020) — Diffusion likelihood estimation
- Analogous to perturbation analysis in Enformer (Avsec et al., Nature Methods 2021)

---

## Appendix E: Evaluation Metrics

### E.1 Primary Metrics (Threshold-Free)

| Metric | Formula | Why Used |
|--------|---------|----------|
| **AUROC** | P(score(pos) > score(neg)) | Threshold-independent discrimination |
| **AUPRC** | Area under precision-recall curve | Better for imbalanced data (3.6% positive rate in MFASS) |

### E.2 Secondary Metrics (Threshold-Dependent)

| Metric | Formula | Why Used |
|--------|---------|----------|
| Balanced Accuracy | (Sensitivity + Specificity) / 2 | Fair for imbalanced classes |
| MCC | (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Best single metric for binary classification |
| F1 Score | 2 × Precision × Recall / (Precision + Recall) | Harmonic mean of precision and recall |

### E.3 Calibration

| Metric | Target | Interpretation |
|--------|--------|----------------|
| ECE | < 0.05 | Predicted probabilities match observed frequencies |
| Brier Score | < 0.15 | Mean squared error of probability predictions |
