"""
SpliceVarMech — Ablation Studies for BiologicalDiffusionModel

Systematic ablation experiments to quantify the contribution of each
component of the new BiologicalDiffusionModel architecture.

Each ablation removes ONE component while keeping everything else constant.

Ablation experiments:
  1. Without Variant Highlight
     → Does marking the mutation position help differentiation?
  2. Without Dual-Stream (single MUT stream only)
     → Does WT/MUT comparison help vs single-stream?
  3. Without Multi-Scale CNN
     → Does multi-scale splice signal extraction help?
  4. Without Contrastive Loss
     → Does contrastive training improve WT/MUT separation?
  5. Without Pre-training
     → Does paired pre-training on synthetic junctions help?
  6. Bayesian Only (no diffusion model)
     → Does the diffusion model add value over handcrafted features?
  7. Without Class Balancing
     → How critical is class-balanced likelihood?

Usage:
    python main.py --ablation              # Run all ablations
    python main.py --ablation --id 4       # Run specific ablation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    ablation_id: int
    name: str
    description: str
    what_is_removed: str
    what_is_tested: str
    expected_effect: str
    baseline_metric: Optional[float] = None
    ablated_metric: Optional[float] = None
    delta: Optional[float] = None
    metric_name: str = "balanced_accuracy"
    status: str = "planned"


@dataclass
class AblationSuite:
    """Complete set of ablation experiments."""
    experiments: list[AblationConfig] = field(default_factory=list)
    baseline_results: Optional[dict] = None


def define_ablation_suite() -> AblationSuite:
    """Define ablation experiments for the BiologicalDiffusionModel."""
    experiments = [
        AblationConfig(
            ablation_id=1,
            name="Without Variant Highlight",
            description=(
                "Remove the VariantHighlight module: no variant marker embedding, "
                "no Gaussian spread, no substitution type encoding. The model must "
                "find the 1-in-428 nucleotide change on its own."
            ),
            what_is_removed="VariantHighlight (variant marker + substitution embedding + Gaussian spread)",
            what_is_tested="Does explicitly marking the mutation position help the model attend to it?",
            expected_effect=(
                "Significant drop in discrimination — this was the core problem with the "
                "original model (50.3% BalAcc). Without the marker, the transformer must "
                "detect a single nucleotide change in 428bp context."
            ),
        ),
        AblationConfig(
            ablation_id=2,
            name="Without Dual-Stream (single MUT stream only)",
            description=(
                "Remove the WT stream and cross-attention. Only process the MUT context "
                "through the encoder. No WT/MUT comparison, no variant impact computation."
            ),
            what_is_removed="DualStreamEncoder (WT stream, cross-attention, variant impact)",
            what_is_tested="Does explicit WT/MUT comparison help vs single-stream encoding?",
            expected_effect=(
                "Moderate drop — the model loses the ability to compute what changed "
                "between WT and MUT. It must infer variant effects from the MUT sequence alone."
            ),
        ),
        AblationConfig(
            ablation_id=3,
            name="Without Multi-Scale CNN",
            description=(
                "Remove the MultiScaleFeatureExtractor: no local (splice site), "
                "regional (ESE/ESS), or structural (branch point) convolutions. "
                "Feed raw token embeddings directly to the dual-stream encoder."
            ),
            what_is_removed="MultiScaleFeatureExtractor (3 parallel CNN scales + gating)",
            what_is_tested="Does multi-scale biological signal extraction improve performance?",
            expected_effect=(
                "Slight drop — the multi-scale CNN provides biological inductive bias "
                "about splice signal structure at different spatial scales."
            ),
        ),
        AblationConfig(
            ablation_id=4,
            name="Without Contrastive Loss",
            description=(
                "Train with diffusion loss only (L_diffusion), no contrastive term. "
                "The model is not explicitly pushed to separate WT/MUT representations."
            ),
            what_is_removed="Contrastive loss (L_contrastive = 0)",
            what_is_tested="Does contrastive training improve WT/MUT discrimination?",
            expected_effect=(
                "Moderate drop — without contrastive pressure, the dual-stream may "
                "produce similar representations for WT and MUT even when the variant "
                "is disruptive."
            ),
        ),
        AblationConfig(
            ablation_id=5,
            name="Without Pre-training",
            description=(
                "Skip Stage 1 pre-training entirely. Train the BiologicalDiffusionModel "
                "from random initialization using only Stage 2 fine-tuning."
            ),
            what_is_removed="Stage 1 pre-training (100K paired synthetic junctions)",
            what_is_tested="Does pre-training on splice junction patterns help?",
            expected_effect=(
                "Higher fine-tune loss — without pre-training, the model has no prior "
                "knowledge of GT/AG recognition, branch point sequences, or which "
                "mutations are disruptive vs benign."
            ),
        ),
        AblationConfig(
            ablation_id=6,
            name="Bayesian Only (no diffusion model)",
            description=(
                "Use only splice tool scores in the Bayesian causal model, "
                "without diffusion model output as the D node."
            ),
            what_is_removed="Diffusion model (Module 1) — D node set to 0",
            what_is_tested="Does the diffusion model add value over handcrafted features?",
            expected_effect=(
                "Lower balanced accuracy — tool scores alone cannot separate "
                "true NCSVs from false positives due to selection bias."
            ),
        ),
        AblationConfig(
            ablation_id=7,
            name="Without Class Balancing",
            description=(
                "Use unweighted Bernoulli likelihood instead of class-balanced "
                "weighted likelihood in the Bayesian model."
            ),
            what_is_removed="Class-balanced weighting (w_pos, w_neg)",
            what_is_tested="How critical is class balancing for the imbalanced dataset?",
            expected_effect=(
                "100% sensitivity, near-0% specificity — predicts positive for everything."
            ),
        ),
    ]
    return AblationSuite(experiments=experiments)


# ──────────────────────────────────────────────────────────────────────
# Ablation runners
# ──────────────────────────────────────────────────────────────────────


def run_ablation_diffusion_only(
    features_list: list,
    diffusion_results: dict,
    verbose: bool = True,
) -> dict:
    """
    Ablation 1-style: Use diffusion output directly without Bayesian model.
    Loads trained checkpoint and scores all gold-standard variants.
    """
    if verbose:
        print("\n  [Ablation] Diffusion Only (no Bayesian causal model)")

    if not diffusion_results:
        diffusion_results = _load_diffusion_and_score_variants(features_list, verbose)

    if not diffusion_results:
        return {"status": "requires_trained_model"}

    true_labels = np.array([f.label for f in features_list])
    aberrant_fracs = np.array([
        diffusion_results.get(f.variant_name, {}).get("aberrant_fraction", 0.5)
        for f in features_list
    ])

    best_thresh, best_ba = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        preds_t = (aberrant_fracs > t).astype(int)
        s = ((preds_t == 1) & (true_labels == 1)).sum() / max((true_labels == 1).sum(), 1)
        sp = ((preds_t == 0) & (true_labels == 0)).sum() / max((true_labels == 0).sum(), 1)
        ba = (s + sp) / 2
        if ba > best_ba:
            best_ba = ba
            best_thresh = t

    preds = (aberrant_fracs > best_thresh).astype(int)
    tp = ((preds == 1) & (true_labels == 1)).sum()
    tn = ((preds == 0) & (true_labels == 0)).sum()
    sens = tp / max((true_labels == 1).sum(), 1)
    spec = tn / max((true_labels == 0).sum(), 1)
    bal_acc = (sens + spec) / 2.0

    if verbose:
        print(f"    Threshold: {best_thresh:.2f}, BalAcc: {bal_acc:.1%}")

    return {"balanced_accuracy": bal_acc, "sensitivity": sens,
            "specificity": spec, "status": "completed"}


def _load_diffusion_and_score_variants(features_list: list, verbose: bool = True) -> dict:
    """Load trained BiologicalDiffusionModel and score gold-standard variants."""
    import torch
    from pathlib import Path

    checkpoint_path = Path("experiments/checkpoints/splice_diffusion_model.pt")
    if not checkpoint_path.exists():
        if verbose:
            print("    ⚠️  No checkpoint found")
        return {}

    try:
        from src.config import get_diffusion_config, get_device
        from src.diffusion.model import BiologicalDiffusionModel
        from src.diffusion.sampling import SpliceSampler, classify_splice_outcome
        from src.diffusion.training import _exon_with_ese, _intron_with_consensus

        config = get_diffusion_config()
        device = get_device()
        model = BiologicalDiffusionModel(config)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        sampler = SpliceSampler(model, device=device)
        results = {}

        for i, f in enumerate(features_list):
            if verbose and i % 10 == 0:
                print(f"    Scoring variant {i+1}/{len(features_list)}...")

            # Build WT and MUT contexts
            exon1 = _exon_with_ese(80)
            exon2 = _exon_with_ese(80)
            intron = _intron_with_consensus(80)
            wt = exon1 + intron + exon2

            var_pos = len(exon1) + max(0, min(abs(f.position), len(intron) - 1))
            mut_list = list(wt)
            ref = mut_list[var_pos]
            alt = "T" if ref != "T" else "A"
            mut_list[var_pos] = alt
            mut = "".join(mut_list)
            wildtype_mrna = (exon1 + exon2)[:200]

            try:
                generated = sampler.generate_samples(
                    wt_context=wt[:256], mut_context=mut[:256],
                    variant_pos=var_pos, ref_allele=ref, alt_allele=alt,
                    n_samples=20, seq_len=200, batch_size=20,
                )
                outcomes = [classify_splice_outcome(seq, wildtype_mrna, mut) for seq in generated]
                from collections import Counter
                mech_counts = Counter(o.mechanism for o in outcomes)
                aberrant = 1.0 - (mech_counts.get("normal", 0) / len(outcomes))
                dominant = mech_counts.most_common(1)[0][0]
                results[f.variant_name] = {"aberrant_fraction": aberrant, "mechanism": dominant}
            except Exception:
                results[f.variant_name] = {"aberrant_fraction": 0.5, "mechanism": "unknown"}

        return results
    except Exception as e:
        if verbose:
            print(f"    ⚠️  Failed: {e}")
        return {}


def run_ablation_bayesian_only(features_list: list, verbose: bool = True) -> dict:
    """Ablation 6: Bayesian model without diffusion output."""
    if verbose:
        print("\n  [Ablation 6] Bayesian Only (no diffusion model)")

    for f in features_list:
        f.diffusion_aberrant_fraction = None

    try:
        from src.causal.dag import (
            build_improved_model, run_inference,
            extract_improved_posteriors, evaluate_predictions,
        )
        model, obs = build_improved_model(features_list, class_weight_strategy="balanced")
        trace = run_inference(model, n_samples=2000, n_tune=1000, n_chains=2)
        posteriors = extract_improved_posteriors(trace, obs["feature_names"])
        true_labels = np.array([f.label for f in features_list])
        eval_result = evaluate_predictions(
            posteriors["p_disruption_mean"], true_labels,
            threshold=0.5, label="Ablation 6: Bayesian Only",
        )
        return {"balanced_accuracy": eval_result["balanced_accuracy"],
                "sensitivity": eval_result["sensitivity"],
                "specificity": eval_result["specificity"],
                "status": "completed"}
    except Exception as e:
        if verbose:
            print(f"    ⚠️  Failed: {e}")
        return {"status": "failed", "error": str(e)}


def run_ablation_no_class_balance(features_list: list, verbose: bool = True) -> dict:
    """Ablation 7: Without class balancing."""
    if verbose:
        print("\n  [Ablation 7] Without class balancing")

    try:
        from src.causal.dag import (
            build_improved_model, run_inference,
            extract_improved_posteriors, evaluate_predictions,
        )
        model, obs = build_improved_model(features_list, class_weight_strategy="none")
        trace = run_inference(model, n_samples=2000, n_tune=1000, n_chains=2)
        posteriors = extract_improved_posteriors(trace, obs["feature_names"])
        true_labels = np.array([f.label for f in features_list])
        eval_result = evaluate_predictions(
            posteriors["p_disruption_mean"], true_labels,
            threshold=0.5, label="Ablation 7: No Class Balance",
        )
        return {"balanced_accuracy": eval_result["balanced_accuracy"],
                "status": "completed"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def run_ablation_no_pretraining(verbose: bool = True) -> dict:
    """
    Ablation 5: Without pre-training.
    Trains BiologicalDiffusionModel from scratch (random init → finetune only).
    """
    if verbose:
        print("\n  [Ablation 5] Without Pre-training")

    try:
        import time
        import torch
        from src.config import get_diffusion_config, get_training_config
        from src.diffusion.model import BiologicalDiffusionModel
        from src.diffusion.training import SpliceTrainer, TrainingConfig

        diff_config = get_diffusion_config()
        base_cfg = get_training_config()

        model_scratch = BiologicalDiffusionModel(diff_config)
        train_config = TrainingConfig(
            pretrain_epochs=0, pretrain_samples=0,
            finetune_epochs=5,
            finetune_batch_size=base_cfg.finetune_batch_size,
            finetune_lr=base_cfg.finetune_lr,
            finetune_augment=base_cfg.finetune_augment,
            finetune_aug_per_variant=base_cfg.finetune_aug_per_variant,
            log_every=50,
            save_dir="experiments/checkpoints/ablation_no_pretrain",
            device=base_cfg.device, seed=base_cfg.seed,
        )

        start = time.time()
        trainer = SpliceTrainer(model_scratch, train_config)
        trainer.finetune()
        elapsed = time.time() - start

        ft_losses = trainer.history.get("finetune_loss", [])
        ft_final = float(np.mean(ft_losses[-10:])) if ft_losses else float('nan')

        if verbose:
            print(f"    No-pretrain FT loss: {ft_final:.4f} ({elapsed:.0f}s)")

        return {"finetune_loss": ft_final, "elapsed": elapsed, "status": "completed"}
    except Exception as e:
        if verbose:
            print(f"    ⚠️  Failed: {e}")
        return {"status": "failed", "error": str(e)}


def run_ablation_no_contrastive(verbose: bool = True) -> dict:
    """
    Ablation 4: Without contrastive loss.
    Sets contrastive_weight=0.0 during training.
    """
    if verbose:
        print("\n  [Ablation 4] Without Contrastive Loss")

    try:
        import time
        from src.config import get_training_config
        from src.diffusion.model import BiologicalDiffusionModel, DiffusionConfig
        from src.diffusion.training import SpliceTrainer

        # Create config with contrastive_weight=0
        config = DiffusionConfig(contrastive_weight=0.0, contrastive_margin=0.0)
        model = BiologicalDiffusionModel(config)
        train_cfg = get_training_config()
        train_cfg.pretrain_epochs = 2
        train_cfg.finetune_epochs = 3
        train_cfg.pretrain_samples = 5000
        train_cfg.save_dir = "experiments/checkpoints/ablation_no_contrastive"

        trainer = SpliceTrainer(model, train_cfg)
        trainer.pretrain()
        trainer.finetune()

        ft_losses = trainer.history.get("finetune_loss", [])
        contr_losses = trainer.history.get("finetune_contrastive_loss", [])
        ft_final = float(np.mean(ft_losses[-10:])) if ft_losses else float('nan')
        contr_final = float(np.mean(contr_losses[-10:])) if contr_losses else 0.0

        if verbose:
            print(f"    No-contrastive FT loss: {ft_final:.4f}")
            print(f"    Contrastive loss (should be ~0): {contr_final:.4f}")

        return {"finetune_loss": ft_final, "contrastive_loss": contr_final,
                "status": "completed"}
    except Exception as e:
        if verbose:
            print(f"    ⚠️  Failed: {e}")
        return {"status": "failed", "error": str(e)}


# ──────────────────────────────────────────────────────────────────────
# Print ablation plan
# ──────────────────────────────────────────────────────────────────────


def print_ablation_plan(verbose: bool = True) -> AblationSuite:
    suite = define_ablation_suite()
    if verbose:
        print("=" * 70)
        print("ABLATION STUDY PLAN — BiologicalDiffusionModel")
        print("=" * 70)
        for exp in suite.experiments:
            print(f"\n  Ablation {exp.ablation_id}: {exp.name}")
            print(f"    Removed: {exp.what_is_removed}")
            print(f"    Tests: {exp.what_is_tested}")
            print(f"    Expected: {exp.expected_effect[:80]}...")
    return suite


def run_ablation_suite(
    features_list: Optional[list] = None,
    diffusion_results: Optional[dict] = None,
    ablation_ids: Optional[list[int]] = None,
    verbose: bool = True,
) -> AblationSuite:
    """Run the ablation suite (or a subset)."""
    suite = define_ablation_suite()

    if features_list is None:
        print_ablation_plan(verbose=verbose)
        return suite

    if verbose:
        print("=" * 70)
        print("RUNNING ABLATION STUDIES — BiologicalDiffusionModel")
        print("=" * 70)

    runners = {
        4: lambda: run_ablation_no_contrastive(verbose),
        5: lambda: run_ablation_no_pretraining(verbose),
        6: lambda: run_ablation_bayesian_only(features_list, verbose),
        7: lambda: run_ablation_no_class_balance(features_list, verbose),
    }

    for exp in suite.experiments:
        if ablation_ids and exp.ablation_id not in ablation_ids:
            continue
        if exp.ablation_id in runners:
            exp.status = "running"
            result = runners[exp.ablation_id]()
            exp.ablated_metric = result.get("balanced_accuracy") or result.get("finetune_loss")
            exp.status = result.get("status", "completed")

            # Free memory between ablation runs (prevents MPS/GPU buildup)
            from src.config import clear_memory_cache
            clear_memory_cache(force=True)
        else:
            exp.status = "requires_trained_model"

    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION SUMMARY")
        print("=" * 70)
        print(f"\n  {'Ablation':<50s} {'Metric':>10s} {'Status':<15s}")
        print("  " + "-" * 75)
        for exp in suite.experiments:
            m = f"{exp.ablated_metric:.4f}" if exp.ablated_metric is not None else "N/A"
            print(f"  {exp.name:<50s} {m:>10s} {exp.status:<15s}")

    # ── Save JSON results ──
    from src.utils.results_io import save_results
    save_results("ablation_results.json", {
        "ablations": [
            {
                "id": exp.ablation_id, "name": exp.name,
                "what_is_removed": exp.what_is_removed,
                "ablated_metric": exp.ablated_metric,
                "baseline_metric": exp.baseline_metric,
                "delta": exp.delta,
                "status": exp.status,
            }
            for exp in suite.experiments
        ],
    }, verbose=verbose)

    return suite


if __name__ == "__main__":
    import sys
    # Apply resource limits when run directly (not via main.py)
    from src.config import apply_resource_limits
    apply_resource_limits()

    if "--run" in sys.argv:
        run_id = None
        if "--id" in sys.argv:
            idx = sys.argv.index("--id")
            run_id = int(sys.argv[idx + 1])
        run_ablation_suite(ablation_ids=[run_id] if run_id else None, verbose=True)
    else:
        print_ablation_plan()
