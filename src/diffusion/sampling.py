"""
SpliceVarMech — Diffusion Model Sampling & Mechanism Classification

Inference pipeline for the trained diffusion model:
  1. Generate N mRNA samples from pre-mRNA context (with variant)
  2. Classify each sample's splice outcome (normal, exon skipping, etc.)
  3. Compute outcome distribution (aberrant fraction, mechanism probabilities)
  4. Align generated sequences to wild-type for mechanism identification

This module answers: WHAT happens to the mRNA when this variant is present?
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from src.diffusion.model import (
    SpliceDiffusionModel,
    DiffusionConfig,
    VOCAB,
    tokenize_sequence,
    detokenize_sequence,
)


# ──────────────────────────────────────────────────────────────────────
# Outcome classification
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SpliceOutcome:
    """Classification of a single generated mRNA's splice outcome."""
    mechanism: str          # "normal", "exon_skipping", "intron_retention",
                           #  "partial_deletion", "cryptic_site", "complex"
    confidence: float       # 0-1 confidence in classification
    generated_seq: str      # The generated mRNA sequence
    length_ratio: float     # len(generated) / len(wildtype)
    detail: str = ""        # Additional detail about the mechanism


@dataclass
class OutcomeDistribution:
    """Distribution over splice outcomes from N samples."""
    n_samples: int
    outcomes: list[SpliceOutcome]

    # Mechanism counts
    mechanism_counts: dict[str, int] = field(default_factory=dict)

    # Summary statistics
    aberrant_fraction: float = 0.0      # Fraction of non-normal outcomes
    dominant_mechanism: str = "unknown"  # Most common mechanism
    dominant_fraction: float = 0.0      # Fraction of dominant mechanism

    # Per-mechanism probabilities
    p_normal: float = 0.0
    p_exon_skipping: float = 0.0
    p_intron_retention: float = 0.0
    p_partial_deletion: float = 0.0
    p_cryptic_site: float = 0.0
    p_complex: float = 0.0


def classify_splice_outcome(
    generated: str,
    wildtype: str,
    context: str,
    length_tolerance: float = 0.1,
) -> SpliceOutcome:
    """
    Classify the splice outcome of a generated mRNA by comparing to wild-type.

    Classification logic:
    1. Similar length to wild-type → normal splicing
    2. Much shorter → exon skipping or partial deletion
    3. Much longer → intron retention
    4. Intermediate → complex or cryptic site

    Args:
        generated: Generated mRNA sequence
        wildtype: Expected wild-type mRNA sequence
        context: Pre-mRNA context (for reference)
        length_tolerance: Fraction deviation from WT length to count as "normal"
    """
    gen_len = len(generated.replace("PAD", "").replace("MASK", "").strip())
    wt_len = len(wildtype) if wildtype else 1

    if gen_len == 0:
        return SpliceOutcome(
            mechanism="unknown", confidence=0.0,
            generated_seq=generated, length_ratio=0.0,
            detail="Empty generated sequence",
        )

    length_ratio = gen_len / wt_len

    # Compute sequence similarity (fraction of matching characters)
    min_len = min(gen_len, wt_len)
    if min_len > 0 and wildtype:
        matches = sum(1 for a, b in zip(generated[:min_len], wildtype[:min_len]) if a == b)
        similarity = matches / min_len
    else:
        similarity = 0.0

    # Classification rules
    if abs(length_ratio - 1.0) <= length_tolerance and similarity > 0.8:
        return SpliceOutcome(
            mechanism="normal", confidence=similarity,
            generated_seq=generated, length_ratio=length_ratio,
            detail=f"Length ratio={length_ratio:.2f}, similarity={similarity:.2f}",
        )
    elif length_ratio < (1.0 - length_tolerance):
        # Shorter than expected → exon skipping or partial deletion
        loss_fraction = 1.0 - length_ratio
        if loss_fraction > 0.3:
            return SpliceOutcome(
                mechanism="exon_skipping", confidence=0.7 + 0.3 * loss_fraction,
                generated_seq=generated, length_ratio=length_ratio,
                detail=f"Lost {loss_fraction:.0%} of expected length",
            )
        else:
            return SpliceOutcome(
                mechanism="partial_deletion", confidence=0.6 + 0.3 * loss_fraction,
                generated_seq=generated, length_ratio=length_ratio,
                detail=f"Partial loss: {loss_fraction:.0%} of expected length",
            )
    elif length_ratio > (1.0 + length_tolerance):
        # Longer than expected → intron retention
        gain_fraction = length_ratio - 1.0
        return SpliceOutcome(
            mechanism="intron_retention", confidence=0.6 + 0.3 * min(gain_fraction, 1.0),
            generated_seq=generated, length_ratio=length_ratio,
            detail=f"Gained {gain_fraction:.0%} extra sequence (likely retained intron)",
        )
    else:
        # Near-normal length but low similarity → cryptic splice site
        return SpliceOutcome(
            mechanism="cryptic_site" if similarity < 0.7 else "normal",
            confidence=max(0.5, 1.0 - similarity),
            generated_seq=generated, length_ratio=length_ratio,
            detail=f"Length ratio={length_ratio:.2f}, similarity={similarity:.2f}",
        )


# ──────────────────────────────────────────────────────────────────────
# Multi-sample generation and analysis
# ──────────────────────────────────────────────────────────────────────


class SpliceSampler:
    """
    Generate multiple mRNA samples and analyze the outcome distribution.

    This is the core inference engine: given a pre-mRNA context (with variant),
    generate N predicted mRNAs and classify each one to estimate the
    probability distribution over splice outcomes.
    """

    def __init__(self, model: SpliceDiffusionModel, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_samples(
        self,
        pre_mrna_context: str,
        n_samples: int = 100,
        seq_len: int = 256,
        temperature: float = 1.0,
        batch_size: int = 10,
    ) -> list[str]:
        """
        Generate N mRNA samples from a pre-mRNA context.

        Args:
            pre_mrna_context: The pre-mRNA sequence with variant (±200bp)
            n_samples: Number of mRNA samples to generate
            seq_len: Length of generated sequences
            temperature: Sampling temperature (higher = more diverse)
            batch_size: Batch size for parallel generation

        Returns:
            List of generated mRNA sequences
        """
        context_tokens = tokenize_sequence(
            pre_mrna_context, max_len=self.model.config.max_seq_len
        ).to(self.device)

        generated_sequences = []
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            curr_batch_size = min(batch_size, n_samples - len(generated_sequences))
            # Expand context for batch
            ctx_batch = context_tokens.unsqueeze(0).expand(curr_batch_size, -1)

            # Generate
            output_tokens = self.model.sample(
                ctx_batch,
                seq_len=seq_len,
                temperature=temperature,
            )

            # Detokenize
            for i in range(curr_batch_size):
                seq = detokenize_sequence(output_tokens[i])
                generated_sequences.append(seq)

        return generated_sequences[:n_samples]

    def analyze_outcomes(
        self,
        pre_mrna_context: str,
        wildtype_mrna: str,
        n_samples: int = 100,
        seq_len: int = 256,
        temperature: float = 1.0,
        batch_size: int = 10,
    ) -> OutcomeDistribution:
        """
        Generate samples and classify their splice outcomes.

        Returns:
            OutcomeDistribution with mechanism probabilities and statistics
        """
        # Generate samples
        generated = self.generate_samples(
            pre_mrna_context=pre_mrna_context,
            n_samples=n_samples,
            seq_len=seq_len,
            temperature=temperature,
            batch_size=batch_size,
        )

        # Classify each sample
        outcomes = []
        for seq in generated:
            outcome = classify_splice_outcome(
                generated=seq,
                wildtype=wildtype_mrna,
                context=pre_mrna_context,
            )
            outcomes.append(outcome)

        # Count mechanisms
        mech_counts = Counter(o.mechanism for o in outcomes)
        total = len(outcomes)

        # Compute probabilities
        dist = OutcomeDistribution(
            n_samples=total,
            outcomes=outcomes,
            mechanism_counts=dict(mech_counts),
            aberrant_fraction=1.0 - (mech_counts.get("normal", 0) / total) if total > 0 else 0.0,
            p_normal=mech_counts.get("normal", 0) / total if total > 0 else 0.0,
            p_exon_skipping=mech_counts.get("exon_skipping", 0) / total if total > 0 else 0.0,
            p_intron_retention=mech_counts.get("intron_retention", 0) / total if total > 0 else 0.0,
            p_partial_deletion=mech_counts.get("partial_deletion", 0) / total if total > 0 else 0.0,
            p_cryptic_site=mech_counts.get("cryptic_site", 0) / total if total > 0 else 0.0,
            p_complex=mech_counts.get("complex", 0) / total if total > 0 else 0.0,
        )

        # Determine dominant mechanism
        if mech_counts:
            dominant = mech_counts.most_common(1)[0]
            dist.dominant_mechanism = dominant[0]
            dist.dominant_fraction = dominant[1] / total if total > 0 else 0.0
        else:
            dist.dominant_mechanism = "unknown"
            dist.dominant_fraction = 0.0

        return dist

    def compare_wildtype_vs_mutant(
        self,
        wildtype_context: str,
        mutant_context: str,
        expected_mrna: str,
        n_samples: int = 50,
        seq_len: int = 256,
        temperature: float = 1.0,
    ) -> dict:
        """
        Compare diffusion outputs between wild-type and mutant contexts.

        This implements the counterfactual comparison:
        - Generate from WT context → should produce normal mRNA
        - Generate from mutant context → may produce aberrant mRNA
        - The difference IS the causal effect of the variant

        Returns dict with both distributions and comparison metrics.
        """
        print("  Generating wild-type samples...")
        wt_dist = self.analyze_outcomes(
            wildtype_context, expected_mrna,
            n_samples=n_samples, seq_len=seq_len, temperature=temperature,
        )

        print("  Generating mutant samples...")
        mut_dist = self.analyze_outcomes(
            mutant_context, expected_mrna,
            n_samples=n_samples, seq_len=seq_len, temperature=temperature,
        )

        # Causal effect: difference in aberrant fraction
        causal_effect = mut_dist.aberrant_fraction - wt_dist.aberrant_fraction

        return {
            "wildtype_distribution": wt_dist,
            "mutant_distribution": mut_dist,
            "causal_effect": causal_effect,
            "wt_aberrant_fraction": wt_dist.aberrant_fraction,
            "mut_aberrant_fraction": mut_dist.aberrant_fraction,
            "wt_dominant_mechanism": wt_dist.dominant_mechanism,
            "mut_dominant_mechanism": mut_dist.dominant_mechanism,
        }


def print_outcome_distribution(dist: OutcomeDistribution, label: str = ""):
    """Pretty-print an outcome distribution."""
    if label:
        print(f"\n  [{label}] Outcome Distribution ({dist.n_samples} samples):")
    else:
        print(f"\n  Outcome Distribution ({dist.n_samples} samples):")

    print(f"    Aberrant fraction: {dist.aberrant_fraction:.1%}")
    print(f"    Dominant mechanism: {dist.dominant_mechanism} ({dist.dominant_fraction:.1%})")
    print(f"    Mechanism breakdown:")
    for mech, count in sorted(dist.mechanism_counts.items(), key=lambda x: -x[1]):
        pct = count / dist.n_samples * 100
        bar = "█" * int(pct / 2)
        print(f"      {mech:25s} {count:4d} ({pct:5.1f}%) {bar}")


# ──────────────────────────────────────────────────────────────────────
# Convenience: run sampling directly
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.diffusion.training import generate_splice_junction, generate_variant_effect

    print("=" * 70)
    print("DIFFUSION MODEL SAMPLING & MECHANISM CLASSIFICATION")
    print("=" * 70)

    # Create a small model for demonstration
    config = DiffusionConfig(
        max_seq_len=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        n_timesteps=20,
    )
    model = SpliceDiffusionModel(config)
    print(f"Model parameters: {model.get_num_params():,}")

    sampler = SpliceSampler(model, device="cpu")

    # Demo 1: Normal splice junction
    print("\n--- Demo 1: Normal Splice Junction ---")
    pre_mrna, wt_mrna = generate_splice_junction(60, 100, 60)
    print(f"  Pre-mRNA length: {len(pre_mrna)}")
    print(f"  Expected mRNA length: {len(wt_mrna)}")

    dist_normal = sampler.analyze_outcomes(
        pre_mrna_context=pre_mrna,
        wildtype_mrna=wt_mrna,
        n_samples=20,
        seq_len=128,
    )
    print_outcome_distribution(dist_normal, "Normal Junction")

    # Demo 2: Variant causing exon skipping
    print("\n--- Demo 2: Variant → Exon Skipping ---")
    mut_pre, aberrant, effect = generate_variant_effect(60, 100, 60, "exon_skipping")
    print(f"  Mutant pre-mRNA length: {len(mut_pre)}")
    print(f"  Aberrant mRNA length: {len(aberrant)}")

    dist_skip = sampler.analyze_outcomes(
        pre_mrna_context=mut_pre,
        wildtype_mrna=wt_mrna,
        n_samples=20,
        seq_len=128,
    )
    print_outcome_distribution(dist_skip, "Exon Skipping Variant")

    # Demo 3: Wild-type vs Mutant comparison
    print("\n--- Demo 3: Wild-type vs Mutant Comparison ---")
    comparison = sampler.compare_wildtype_vs_mutant(
        wildtype_context=pre_mrna,
        mutant_context=mut_pre,
        expected_mrna=wt_mrna,
        n_samples=10,
        seq_len=128,
    )
    print(f"  WT aberrant fraction: {comparison['wt_aberrant_fraction']:.1%}")
    print(f"  MUT aberrant fraction: {comparison['mut_aberrant_fraction']:.1%}")
    print(f"  Causal effect: {comparison['causal_effect']:+.1%}")

    print("\n✅ Sampling & classification demo complete")
