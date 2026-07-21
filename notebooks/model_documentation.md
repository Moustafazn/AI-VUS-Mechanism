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

### 2.3 The Denoising Network (Reverse Process)

**Goal:** Learn p_θ(x_0 | x_t, context, t) — predict the clean sequence from the corrupted one.

**Architecture:** Transformer encoder-decoder with cross-attention.

```
Input:  x_t (corrupted mRNA, [batch, seq_len])
        + context (pre-mRNA, [batch, ctx_len])
        + t (timestep, [batch])
        
Encoder: Embeds context → 3 transformer layers → context representation
Decoder: Embeds x_t + positional + timestep embedding
         → 6 transformer layers with cross-attention to context
         → Linear projection → logits [batch, seq_len, vocab_size=7]
         
Output: Per-position probability over {PAD, A, C, G, T, MASK, SEP}
```

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 256 | Balance between capacity and training speed |
| n_heads | 8 | Multi-head attention for diverse splice features |
| n_layers | 6 (decoder), 3 (encoder) | Decoder needs more capacity for generation |
| d_ff | 1024 | Standard 4× expansion |
| n_timesteps | 100 | Sufficient for gradual denoising |
| max_seq_len | 512 | Covers exon-intron-exon regions |
| vocab_size | 7 | {PAD, A, C, G, T, MASK, SEP} |

**Total parameters:** ~9.2 million

### 2.4 Training Loss

Cross-entropy loss computed **only on masked positions**:

```
L(θ) = 𝔼_{t~U(0,T), x_0~data} [ -∑ᵢ∈masked 𝟙[xᵢ_t = MASK] · log p_θ(xᵢ_0 | x_t, ctx, t) ]
```

**Why only masked positions?** Unmasked positions are already correct — the model should focus on learning to predict corrupted tokens. This is analogous to BERT's masked language modeling but for DNA sequences.

### 2.5 Sampling (Inference)

**Iterative denoising** from fully masked → predicted mRNA:

```
For t = T-1, T-2, ..., 0:
  1. Predict logits = p_θ(x_0 | x_t, context, t)
  2. Identify still-masked positions
  3. Compute unmask fraction: f = (mask_prob(t) - mask_prob(t-1)) / mask_prob(t)
  4. Select top-f% most confident masked positions
  5. Sample tokens from predicted distribution at selected positions
  6. Update x_t → x_{t-1}
```

**Confidence-based unmasking:** At each step, we unmask positions where the model is most confident (highest max probability). This produces coherent sequences rather than noisy random samples.

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
| Data | Synthetic exon-intron-exon triplets |
| Size | 10,000 - 1,000,000 examples |
| Input | Pre-mRNA (exon + GT...AG + exon) |
| Target | Correctly spliced mRNA (exon + exon) |
| Objective | Cross-entropy on masked positions |
| Schedule | Cosine learning rate, AdamW |

**What the model learns:**
- GT/AG = splice boundaries (remove intron)
- ESE motifs in exons → keep exon
- Polypyrimidine tract → acceptor site signal
- Branch point sequence → splice site strength

**Stage 2: Fine-tuning** (learns variant effects)

| Parameter | Value |
|-----------|-------|
| Primary data | 37 S7 positives + 14 S2 negatives = 51 |
| Study 6 data | +183 splice variants (with synthetic targets) |
| Study 4 data | +50 TESE-positive as weak negatives |
| Total (before aug) | 284 examples |
| Augmentation | 5× per variant + synthetic variants |
| Total (after aug) | ~1,900+ examples |
| Learning rate | 5e-5 (lower than pre-training) |

### 5.2 Data Augmentation Strategies

1. **Nucleotide substitution** (1-3 random positions in non-critical regions)
2. **Synthetic variant generation** (mutate donor/acceptor in synthetic sequences)
3. **Mechanism-conditioned generation** (generate exon skipping / intron retention / partial deletion)
4. **External data integration** (Study 6 splice variants + Study 4 TESE negatives)

### 5.3 Weight Optimization

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

**Learning rate schedule:** Cosine annealing

```
η(t) = η_min + (η_max - η_min) · (1 + cos(πt/T)) / 2
```

**Gradient clipping:** max_norm = 1.0 (prevents exploding gradients in the transformer)

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

## Appendix A: Comparison with Existing Approaches

| Approach | SpliceAI | DNABERT-2 | Evo | **SpliceVarMech** |
|----------|----------|-----------|-----|-------------------|
| Architecture | ResNet | BERT | StripedHyena | D3PM + SCM |
| Output | Score (0-1) | Classification | Embedding | **mRNA sequence + probability** |
| Mechanism | ❌ | ❌ | ❌ | **✅ Generated** |
| Uncertainty | ❌ | ❌ | ❌ | **✅ Bayesian CI** |
| Causal reasoning | ❌ | ❌ | ❌ | **✅ do-calculus** |
| Explainability | ❌ | ❌ | ❌ | **✅ Attribution + paths** |
| NCSV performance | Poor (>+10bp) | Unknown | General | **Designed for NCSVs** |

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
