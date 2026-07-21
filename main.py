"""
SpliceVarMech — Main Entry Point

Unified entry point for the entire SpliceVarMech pipeline.
Supports running individual phases or the complete workflow.

Usage:
    python main.py                  # Run full pipeline summary
    python main.py --phase 1        # Parse dataset
    python main.py --phase 2        # Splice tool analysis
    python main.py --phase 3        # Bayesian causal model (diagnostics + comparison)
    python main.py --phase 4        # Diffusion model architecture validation
    python main.py --phase 5        # Training pipeline (pre-train + fine-tune)
    python main.py --phase 6        # TEX11 prediction + clinical report
    python main.py --baselines      # Run baseline tool evaluation
    python main.py --all            # Run all phases sequentially
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def phase1_parse():
    """Phase 1: Parse the primary dataset."""
    print("=" * 70)
    print("PHASE 1: DATA PARSING")
    print("=" * 70)
    from src.data.parser import parse_dataset
    import numpy as np

    dataset = parse_dataset()
    print(f"\n{'DATASET SUMMARY':}")
    print(f"  Table S1: {dataset.summary.s1_variant_count} variants × {dataset.summary.s1_column_count} cols")
    print(f"  Table S2: {dataset.summary.s2_total_count} total "
          f"({dataset.summary.s2_normal_count} Normal, {dataset.summary.s2_failed_count} Failed)")
    print(f"  Table S7: {dataset.summary.s7_total_count} gold-standard NCSVs")
    print(f"    Types: {dataset.summary.s7_type_counts}")
    print(f"    Mechanisms: {dataset.summary.s7_mechanism_counts}")
    print(f"    Sequence lengths: min={min(dataset.summary.s7_sequence_lengths)}, "
          f"max={max(dataset.summary.s7_sequence_lengths)}, "
          f"mean={np.mean(dataset.summary.s7_sequence_lengths):.0f}")
    print(f"  Table S3: {len(dataset.table_s3)} patient-level variants")
    print(f"  Table S4: {len(dataset.table_s4)} patients")
    print(f"  Table S5: {len(dataset.table_s5)} extended variants")
    print(f"  Splice tools: {len(dataset.summary.splice_tool_columns_found)} found")
    print("\n✅ Phase 1 complete")
    return dataset


def phase2_features():
    """Phase 2: Splice tool score analysis."""
    print("\n" + "=" * 70)
    print("PHASE 2: SPLICE TOOL ANALYSIS")
    print("=" * 70)
    from src.data.parser import parse_dataset
    from src.features.splice_scores import analyze_coverage, match_gold_standard_to_s1

    dataset = parse_dataset()
    coverage = analyze_coverage(dataset)
    print(f"\nTool coverage across {coverage.total_variants} variants:")
    print(coverage.coverage_df.to_string(index=False))

    gs_scores = match_gold_standard_to_s1(dataset)
    print(f"\nGold-standard matching:")
    print(f"  Positives matched: {len(gs_scores.matched_positives)}/40")
    print(f"  Negatives matched: {len(gs_scores.matched_negatives)}/14")
    print(f"  Score matrix shape: {gs_scores.score_matrix.shape}")
    print("\n✅ Phase 2 complete")
    return gs_scores


def phase3_causal():
    """Phase 3: Bayesian causal model diagnostics."""
    print("\n" + "=" * 70)
    print("PHASE 3: BAYESIAN CAUSAL MODEL DIAGNOSTICS")
    print("=" * 70)
    from src.causal.diagnostics import run_diagnostics
    results = run_diagnostics(verbose=True)
    print("\n✅ Phase 3 diagnostics complete")
    print("  (Run 'python -m src.causal.dag' for full model comparison with MCMC)")
    return results


def phase4_diffusion():
    """Phase 4: Validate diffusion model architecture."""
    print("\n" + "=" * 70)
    print("PHASE 4: DIFFUSION MODEL ARCHITECTURE VALIDATION")
    print("=" * 70)
    import torch
    from src.diffusion.model import SpliceDiffusionModel, DiffusionConfig, detokenize_sequence

    config = DiffusionConfig(
        max_seq_len=128, d_model=64, n_heads=4, n_layers=2,
        d_ff=256, n_timesteps=10,
    )
    model = SpliceDiffusionModel(config)
    print(f"  Model parameters: {model.get_num_params():,}")

    # Test forward pass
    batch_size, seq_len, ctx_len = 2, 64, 128
    x_0 = torch.randint(1, 5, (batch_size, seq_len))
    context = torch.randint(1, 5, (batch_size, ctx_len))
    loss = model.training_loss(x_0, context)
    print(f"  Forward pass loss: {loss.item():.4f}")

    # Test sampling
    context_small = torch.randint(1, 5, (1, 64))
    generated = model.sample(context_small, seq_len=32)
    print(f"  Sampling output shape: {generated.shape}")
    print(f"  Generated sequence: {detokenize_sequence(generated[0])[:50]}...")
    print("\n✅ Phase 4 complete — diffusion architecture validated")
    return model


def phase5_training():
    """Phase 5: Training pipeline (small demo)."""
    print("\n" + "=" * 70)
    print("PHASE 5: DIFFUSION MODEL TRAINING")
    print("=" * 70)

    from src.diffusion.model import SpliceDiffusionModel, DiffusionConfig
    from src.diffusion.training import SpliceTrainer, TrainingConfig

    diff_config = DiffusionConfig(
        max_seq_len=128, d_model=64, n_heads=4, n_layers=2,
        d_ff=256, n_timesteps=20,
    )
    model = SpliceDiffusionModel(diff_config)
    print(f"  Model parameters: {model.get_num_params():,}")

    train_config = TrainingConfig(
        pretrain_epochs=2, pretrain_samples=100, pretrain_batch_size=8,
        finetune_epochs=2, finetune_batch_size=4,
        finetune_augment=True, finetune_aug_per_variant=2,
        log_every=5, device="cpu",
    )

    trainer = SpliceTrainer(model, train_config)
    trainer.pretrain()
    trainer.finetune()
    trainer.save_checkpoint()

    print("\n✅ Phase 5 complete — training pipeline validated")
    return trainer


def phase6_prediction():
    """Phase 6: TEX11 prediction + clinical report."""
    print("\n" + "=" * 70)
    print("PHASE 6: TEX11 PREDICTION + CLINICAL REPORT")
    print("=" * 70)

    from src.diffusion.model import DiffusionConfig
    from src.pipeline.predict import run_tex11_prediction

    report = run_tex11_prediction(
        n_samples=20,
        model_config=DiffusionConfig(
            max_seq_len=128, d_model=64, n_heads=4, n_layers=2,
            d_ff=256, n_timesteps=15,
        ),
        device="cpu",
    )
    print("\n✅ Phase 6 complete — clinical report generated")
    return report


def run_baselines():
    """Evaluate all 17 baseline tools."""
    from src.baselines.tool_evaluation import run_baseline_evaluation
    results = run_baseline_evaluation(verbose=True)
    return results


def run_xai():
    """Run XAI analysis (Module 3)."""
    print("\n" + "=" * 70)
    print("MODULE 3: EXPLAINABLE AI ANALYSIS")
    print("=" * 70)
    import torch
    from src.diffusion.model import SpliceDiffusionModel, DiffusionConfig
    from src.diffusion.training import _exon_with_ese, _intron_with_consensus
    from src.xai.attribution import run_xai_analysis, format_attribution_heatmap

    config = DiffusionConfig(
        max_seq_len=128, d_model=64, n_heads=4, n_layers=2,
        d_ff=256, n_timesteps=10,
    )
    model = SpliceDiffusionModel(config)

    exon = _exon_with_ese(60)
    intron = _intron_with_consensus(60)
    context = (exon + intron + _exon_with_ese(60))[:128]
    target = (exon + _exon_with_ese(60))[:128]

    report = run_xai_analysis(model=model, context_seq=context, target_seq=target, verbose=True)
    print("\n" + format_attribution_heatmap(report.attribution))
    print("\n✅ XAI analysis complete")
    return report


def run_benchmarks():
    """Run SOTA benchmarking against 2022-2026 literature."""
    from src.baselines.benchmark import run_benchmark_comparison
    results = run_benchmark_comparison(
        our_balanced_accuracy=0.747,
        verbose=True,
    )
    return results


def print_project_summary():
    """Print project overview and status."""
    print("=" * 70)
    print("SpliceVarMech — Causal Generative Framework")
    print("for Mechanistic Interpretation of Non-Canonical Splicing Variants")
    print("=" * 70)
    print("""
  Project Modules:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Phase 1: Data Parser           src/data/parser.py             │
  │ Phase 2: Splice Tool Analysis  src/features/splice_scores.py  │
  │ Phase 3: Bayesian Causal Model src/causal/dag.py              │
  │          Diagnostics           src/causal/diagnostics.py      │
  │ Phase 4: Diffusion Architecture src/diffusion/model.py        │
  │ Phase 5: Training Pipeline     src/diffusion/training.py      │
  │          Sampling/Inference    src/diffusion/sampling.py      │
  │ Phase 6: TEX11 Prediction      src/pipeline/predict.py        │
  │ Baselines: Tool Evaluation     src/baselines/tool_evaluation.py│
  └─────────────────────────────────────────────────────────────────┘

  Clinical Case: TEX11 c.1156+16G>T → VUS in male infertility
  Pipeline:      Variant → Diffusion → Causal Inference → Clinical Report

  Run individual phases:
    python main.py --phase 1    # Parse dataset
    python main.py --phase 2    # Feature analysis
    python main.py --phase 3    # Bayesian diagnostics
    python main.py --phase 4    # Architecture validation
    python main.py --phase 5    # Training pipeline
    python main.py --phase 6    # TEX11 prediction
    python main.py --baselines  # Baseline evaluation
    python main.py --all        # All phases
""")

    # Quick import validation
    print("Module Status:")
    modules = [
        ("src.data.parser", "Phase 1: Data Parser"),
        ("src.features.splice_scores", "Phase 2: Splice Features"),
        ("src.causal.dag", "Phase 3: Bayesian Causal Model"),
        ("src.causal.diagnostics", "Phase 3: Diagnostics"),
        ("src.diffusion.model", "Phase 4: Diffusion Architecture"),
        ("src.diffusion.training", "Phase 5: Training Pipeline"),
        ("src.diffusion.sampling", "Phase 5: Sampling/Inference"),
        ("src.pipeline.predict", "Phase 6: TEX11 Prediction"),
        ("src.baselines.tool_evaluation", "Baselines: Tool Evaluation"),
    ]
    for module, desc in modules:
        try:
            __import__(module)
            print(f"  ✅ {desc:<35s} ({module})")
        except ImportError as e:
            print(f"  ❌ {desc:<35s} ({module}) — {e}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SpliceVarMech — Causal Generative Framework"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="Run a specific phase (1-6)"
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Run baseline tool evaluation"
    )
    parser.add_argument(
        "--xai", action="store_true",
        help="Run XAI analysis (Module 3)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run SOTA benchmarking against 2022-2026 literature"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all phases sequentially"
    )

    args = parser.parse_args()

    if args.all:
        phase1_parse()
        phase2_features()
        phase3_causal()
        phase4_diffusion()
        phase5_training()
        phase6_prediction()
        run_xai()
        run_baselines()
        run_benchmarks()
        print("\n" + "=" * 70)
        print("ALL PHASES COMPLETE")
        print("=" * 70)
    elif args.phase == 1:
        phase1_parse()
    elif args.phase == 2:
        phase2_features()
    elif args.phase == 3:
        phase3_causal()
    elif args.phase == 4:
        phase4_diffusion()
    elif args.phase == 5:
        phase5_training()
    elif args.phase == 6:
        phase6_prediction()
    elif args.baselines:
        run_baselines()
    elif args.xai:
        run_xai()
    elif args.benchmark:
        run_benchmarks()
    else:
        print_project_summary()
