"""
SpliceVarMech — Diffusion Model Training Pipeline (Phase 5)

Two-stage training:
  Stage 1 — Pre-training on synthetic splice junctions
    • Generates exon-intron-exon triplets with GT/AG consensus
    • Model learns the rules of normal splicing
    • Designed for GENCODE data when available; synthetic data for demo

  Stage 2 — Fine-tuning on gold-standard variant effects
    • 40 positive NCSVs from Table S7 (mutant → aberrant mRNA)
    • 14 negative controls from Table S2 (mutant → normal mRNA)
    • Model learns how variants alter splice outcomes
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.diffusion.model import (
    SpliceDiffusionModel,
    DiffusionConfig,
    VOCAB,
    tokenize_sequence,
    detokenize_sequence,
)


# ──────────────────────────────────────────────────────────────────────
# Synthetic splice junction generation (for pre-training)
# ──────────────────────────────────────────────────────────────────────

# Biologically informed motifs
ESE_HEXAMERS = [
    "GAAGAA", "GGAGGA", "AAGAAG", "GACGAC", "AAGAAC",
    "GAAGGC", "AGAAGA", "GAAGAG", "AACAAG", "GAAGAT",
]
DONOR_CONSENSUS = "GTAAGT"   # Canonical donor: GT at +1/+2
ACCEPTOR_CONSENSUS = "AG"     # Canonical acceptor: AG at -1/-2
PYRIMIDINE_TRACT = "TTTTCTTTCC"  # Polypyrimidine tract motif


def _random_seq(length: int) -> str:
    """Generate random nucleotide sequence."""
    return "".join(random.choice("ACGT") for _ in range(length))


def _exon_with_ese(length: int = 100) -> str:
    """Generate an exon-like sequence with embedded ESE motifs."""
    seq = list(_random_seq(length))
    # Insert 1-2 ESE motifs at random positions
    for _ in range(random.randint(1, 2)):
        ese = random.choice(ESE_HEXAMERS)
        pos = random.randint(5, length - len(ese) - 5)
        for i, c in enumerate(ese):
            seq[pos + i] = c
    return "".join(seq)


def _intron_with_consensus(length: int = 200) -> str:
    """
    Generate an intron-like sequence with canonical splice signals:
      - GT at 5' (donor)
      - AG at 3' (acceptor)
      - Polypyrimidine tract near 3' end
      - Branch point ~20-40bp upstream of 3' end
    """
    if length < 30:
        length = 30
    # Core intron body
    body_len = length - len(DONOR_CONSENSUS) - len(ACCEPTOR_CONSENSUS) - len(PYRIMIDINE_TRACT) - 5
    if body_len < 10:
        body_len = 10
    body = _random_seq(body_len)
    # Branch point (YNYURAY → approximated as TACTAAC)
    bp = "TACTAAC"
    intron = DONOR_CONSENSUS + body + bp + PYRIMIDINE_TRACT + ACCEPTOR_CONSENSUS
    return intron[:length]  # Truncate to requested length


def generate_splice_junction(
    exon1_len: int = 100,
    intron_len: int = 200,
    exon2_len: int = 100,
) -> tuple[str, str]:
    """
    Generate a synthetic exon-intron-exon triplet with correct splicing.

    Returns:
        (pre_mrna, mature_mrna) — the input and target for the diffusion model
    """
    exon1 = _exon_with_ese(exon1_len)
    intron = _intron_with_consensus(intron_len)
    exon2 = _exon_with_ese(exon2_len)

    pre_mrna = exon1 + intron + exon2
    mature_mrna = exon1 + exon2  # Correct splicing removes the intron

    return pre_mrna, mature_mrna


def generate_variant_effect(
    exon1_len: int = 100,
    intron_len: int = 200,
    exon2_len: int = 100,
    effect_type: str = "exon_skipping",
) -> tuple[str, str, str]:
    """
    Generate a synthetic variant that disrupts splicing.

    Returns:
        (mutant_pre_mrna, aberrant_mrna, effect_label)
    """
    exon1 = _exon_with_ese(exon1_len)
    intron = _intron_with_consensus(intron_len)
    exon2 = _exon_with_ese(exon2_len)

    # Introduce a mutation that disrupts the splice site
    intron_list = list(intron)
    # Mutate the GT donor
    if len(intron_list) > 1:
        intron_list[0] = random.choice("AC")  # Destroy GT → AT/CT
    mutant_intron = "".join(intron_list)

    mutant_pre_mrna = exon1 + mutant_intron + exon2

    if effect_type == "exon_skipping":
        # Exon 2 is skipped entirely
        aberrant_mrna = exon1
    elif effect_type == "intron_retention":
        # Intron is retained
        aberrant_mrna = exon1 + mutant_intron + exon2
    elif effect_type == "partial_deletion":
        # Partial exon 2 loss (cryptic splice site)
        cut = random.randint(10, max(11, exon2_len - 10))
        aberrant_mrna = exon1 + exon2[cut:]
    else:
        aberrant_mrna = exon1 + exon2  # Normal (negative control)

    return mutant_pre_mrna, aberrant_mrna, effect_type


# ──────────────────────────────────────────────────────────────────────
# PyTorch Datasets
# ──────────────────────────────────────────────────────────────────────


class SyntheticSpliceDataset(Dataset):
    """
    Dataset of synthetic exon-intron-exon splice junctions for pre-training.

    Each example is a (pre_mRNA, mature_mRNA) pair where the model learns
    to generate correct spliced output from pre-mRNA input.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        ctx_len: int = 400,
        target_len: int = 200,
        exon_range: tuple[int, int] = (60, 120),
        intron_range: tuple[int, int] = (80, 300),
        seed: int = 42,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.ctx_len = ctx_len
        self.target_len = target_len
        self.exon_range = exon_range
        self.intron_range = intron_range
        random.seed(seed)
        np.random.seed(seed)

        # Pre-generate all examples
        self.examples: list[tuple[str, str]] = []
        for _ in range(n_samples):
            e1 = random.randint(*exon_range)
            il = random.randint(*intron_range)
            e2 = random.randint(*exon_range)
            pre, mature = generate_splice_junction(e1, il, e2)
            self.examples.append((pre, mature))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pre_mrna, mature_mrna = self.examples[idx]
        context = tokenize_sequence(pre_mrna, max_len=self.ctx_len)
        target = tokenize_sequence(mature_mrna, max_len=self.target_len)
        return context, target


class GoldStandardDataset(Dataset):
    """
    Dataset of gold-standard variants for fine-tuning.

    Uses Table S7 (positive NCSVs with aberrant mRNA) and Table S2 negatives.
    Since we lack pre-mRNA reference sequences, we create synthetic contexts
    using the available aberrant/normal mRNA sequences.

    In production, this would use hg38 reference genome for actual pre-mRNA.
    """

    def __init__(
        self,
        ctx_len: int = 400,
        target_len: int = 512,
        augment: bool = True,
        n_augmented_per_variant: int = 5,
    ):
        super().__init__()
        self.ctx_len = ctx_len
        self.target_len = target_len
        self.examples: list[tuple[str, str, int, str]] = []

        self._load_gold_standard()

        if augment:
            self._augment(n_augmented_per_variant)

    def _load_gold_standard(self):
        """Load S7 positives, S2 negatives, and external Study 6 data."""
        from src.data.parser import parse_dataset

        dataset = parse_dataset()

        # ── Primary: Positive variants (S7) with aberrant mRNA ──
        for v in dataset.gold_standard_positives:
            if v.sequence_length > 10:
                mrna = v.aberrant_mrna_sequence
                pre_mrna_ctx = mrna[:min(len(mrna), self.ctx_len)]
                if len(pre_mrna_ctx) < self.ctx_len:
                    intron_padding = _intron_with_consensus(
                        self.ctx_len - len(pre_mrna_ctx)
                    )
                    pre_mrna_ctx = pre_mrna_ctx + intron_padding

                self.examples.append((
                    pre_mrna_ctx,
                    mrna,
                    1,  # label: splice disrupting
                    v.mechanism_category,
                ))

        # ── Primary: Negative variants (S2 Normal) ──
        for v in dataset.usable_negatives:
            exon1 = _exon_with_ese(100)
            exon2 = _exon_with_ese(100)
            normal_mrna = exon1 + exon2
            intron = _intron_with_consensus(200)
            pre_mrna = exon1 + intron + exon2

            self.examples.append((
                pre_mrna[:self.ctx_len],
                normal_mrna,
                0,  # label: normal splicing
                "normal",
            ))

        # ── External: Study 6 splice variants (341 variants with scores) ──
        # These don't have aberrant mRNA sequences, but we can use them
        # for binary classification training with synthetic targets
        try:
            from src.data.external_parser import get_study6_splice_variants

            s6_variants = get_study6_splice_variants(
                include_intronic=True, include_exonic_splice=True,
            )
            n_s6_added = 0
            for v in s6_variants:
                # Splice variants = positive label (splice disrupting)
                # Create synthetic pre-mRNA from the variant's context
                exon1 = _exon_with_ese(random.randint(60, 120))
                intron = _intron_with_consensus(random.randint(80, 200))
                exon2 = _exon_with_ese(random.randint(60, 120))
                pre_mrna = exon1 + intron + exon2

                if v.func_refgene == "splicing":
                    # Canonical splice variant → exon skipping target
                    aberrant = exon1  # Skip exon2
                    mechanism = "exon_skipping"
                elif v.func_refgene == "intronic":
                    # Intronic variant → intron retention target
                    aberrant = exon1 + intron + exon2
                    mechanism = "intron_retention"
                else:
                    # Exonic splice effect → exon skipping
                    aberrant = exon1
                    mechanism = "exon_skipping"

                self.examples.append((
                    pre_mrna[:self.ctx_len],
                    aberrant,
                    1,
                    mechanism,
                ))
                n_s6_added += 1

            print(f"  [GoldStandard] Added {n_s6_added} Study 6 splice variants")
        except (ImportError, FileNotFoundError) as e:
            print(f"  [GoldStandard] Study 6 data not available: {e}")

        # ── External: Study 4 TESE-positive variants as negatives ──
        # TESE-positive = sperm retrieved = gene function not completely lost
        # Can serve as weak negative examples (not definitive)
        try:
            from src.data.external_parser import parse_study4

            _, s4_variants = parse_study4()
            n_s4_added = 0
            for v in s4_variants:
                if v.tese_outcome == "Positive":
                    # TESE positive → likely normal/partial splicing
                    exon1 = _exon_with_ese(100)
                    exon2 = _exon_with_ese(100)
                    normal_mrna = exon1 + exon2
                    intron = _intron_with_consensus(200)
                    pre_mrna = exon1 + intron + exon2

                    self.examples.append((
                        pre_mrna[:self.ctx_len],
                        normal_mrna,
                        0,
                        "normal",
                    ))
                    n_s4_added += 1
                    if n_s4_added >= 50:  # Cap to avoid overwhelming
                        break

            print(f"  [GoldStandard] Added {n_s4_added} Study 4 TESE-positive as negatives")
        except (ImportError, FileNotFoundError) as e:
            print(f"  [GoldStandard] Study 4 data not available: {e}")

    def _augment(self, n_per_variant: int):
        """
        Data augmentation via:
        1. Random nucleotide substitutions in non-critical regions
        2. Reverse complement of some sequences
        3. Synthetic variants with known effects
        """
        original_examples = list(self.examples)
        for pre, target, label, mech in original_examples:
            for _ in range(n_per_variant):
                # Random substitution augmentation (1-3 positions)
                aug_pre = list(pre)
                n_muts = random.randint(1, 3)
                for _ in range(n_muts):
                    pos = random.randint(0, len(aug_pre) - 1)
                    aug_pre[pos] = random.choice("ACGT")
                aug_pre_str = "".join(aug_pre)

                # Slight noise in target too (for robustness)
                aug_target = list(target)
                if random.random() < 0.3:
                    pos = random.randint(0, len(aug_target) - 1)
                    aug_target[pos] = random.choice("ACGT")
                aug_target_str = "".join(aug_target)

                self.examples.append((
                    aug_pre_str[:self.ctx_len],
                    aug_target_str,
                    label,
                    mech,
                ))

        # Add synthetic disruption examples
        for effect in ["exon_skipping", "intron_retention", "partial_deletion"]:
            for _ in range(n_per_variant * 3):
                pre, aberrant, _ = generate_variant_effect(
                    exon1_len=random.randint(60, 120),
                    intron_len=random.randint(80, 250),
                    exon2_len=random.randint(60, 120),
                    effect_type=effect,
                )
                self.examples.append((
                    pre[:self.ctx_len],
                    aberrant,
                    1,
                    effect,
                ))

        # Add synthetic normal examples
        for _ in range(n_per_variant * 5):
            pre, mature = generate_splice_junction(
                exon1_len=random.randint(60, 120),
                intron_len=random.randint(80, 250),
                exon2_len=random.randint(60, 120),
            )
            self.examples.append((
                pre[:self.ctx_len],
                mature,
                0,
                "normal",
            ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        pre, target, label, mech = self.examples[idx]
        ctx_tokens = tokenize_sequence(pre, max_len=self.ctx_len)
        tgt_tokens = tokenize_sequence(target, max_len=self.target_len)
        return ctx_tokens, tgt_tokens, label, mech


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    # Pre-training
    pretrain_epochs: int = 10
    pretrain_samples: int = 10000
    pretrain_batch_size: int = 16
    pretrain_lr: float = 1e-4

    # Fine-tuning
    finetune_epochs: int = 20
    finetune_batch_size: int = 8
    finetune_lr: float = 5e-5
    finetune_augment: bool = True
    finetune_aug_per_variant: int = 5

    # Shared
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    log_every: int = 10
    save_dir: str = "experiments/checkpoints"
    device: str = "cpu"  # "cuda" if available
    seed: int = 42


class SpliceTrainer:
    """
    Two-stage trainer for the splice diffusion model.

    Stage 1: Pre-train on synthetic splice junctions
    Stage 2: Fine-tune on gold-standard variant effects
    """

    def __init__(
        self,
        model: SpliceDiffusionModel,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Training history
        self.history: dict[str, list[float]] = {
            "pretrain_loss": [],
            "finetune_loss": [],
            "finetune_pos_loss": [],
            "finetune_neg_loss": [],
        }

    def pretrain(self) -> dict[str, list[float]]:
        """
        Stage 1: Pre-train on synthetic splice junction data.

        The model learns:
        - GT/AG donor/acceptor recognition
        - Intron removal (exon-exon junction generation)
        - ESE/ESS motif effects on exon inclusion
        """
        print("=" * 70)
        print("STAGE 1: PRE-TRAINING ON SYNTHETIC SPLICE JUNCTIONS")
        print("=" * 70)

        dataset = SyntheticSpliceDataset(
            n_samples=self.config.pretrain_samples,
            ctx_len=self.model.config.max_seq_len,
            target_len=self.model.config.max_seq_len,
            seed=self.config.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.pretrain_batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.pretrain_lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.pretrain_epochs * len(loader)
        )

        self.model.train()
        total_steps = 0

        for epoch in range(1, self.config.pretrain_epochs + 1):
            epoch_losses = []
            for batch_idx, (context, target) in enumerate(loader):
                context = context.to(self.device)
                target = target.to(self.device)

                loss = self.model.training_loss(target, context)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                optimizer.step()
                scheduler.step()

                epoch_losses.append(loss.item())
                self.history["pretrain_loss"].append(loss.item())
                total_steps += 1

                if total_steps % self.config.log_every == 0:
                    avg = np.mean(epoch_losses[-self.config.log_every:])
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"  [Pre-train] Epoch {epoch}/{self.config.pretrain_epochs} "
                        f"Step {total_steps} Loss={avg:.4f} LR={lr:.2e}"
                    )

            avg_loss = np.mean(epoch_losses)
            print(
                f"  [Pre-train] Epoch {epoch} complete — "
                f"Avg loss={avg_loss:.4f}"
            )

        print(f"\n✅ Pre-training complete ({total_steps} steps)")
        return self.history

    def finetune(self) -> dict[str, list[float]]:
        """
        Stage 2: Fine-tune on gold-standard variant effects.

        The model learns:
        - How specific variants alter splice outcomes
        - Exon skipping, intron retention, partial deletion patterns
        - That some variants DON'T disrupt splicing (negatives)
        """
        print("\n" + "=" * 70)
        print("STAGE 2: FINE-TUNING ON GOLD-STANDARD VARIANTS")
        print("=" * 70)

        dataset = GoldStandardDataset(
            ctx_len=self.model.config.max_seq_len,
            target_len=self.model.config.max_seq_len,
            augment=self.config.finetune_augment,
            n_augmented_per_variant=self.config.finetune_aug_per_variant,
        )
        print(f"  Fine-tune dataset: {len(dataset)} examples")

        # Count labels and mechanisms
        label_counts = {0: 0, 1: 0}
        mech_counts: dict[str, int] = {}
        for i in range(len(dataset)):
            _, _, label, mech = dataset[i]
            label_counts[label] = label_counts.get(label, 0) + 1
            mech_counts[mech] = mech_counts.get(mech, 0) + 1
        print(f"  Labels: {label_counts}")
        print(f"  Mechanisms: {mech_counts}")

        loader = DataLoader(
            dataset,
            batch_size=self.config.finetune_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_finetune,
        )

        # Lower learning rate for fine-tuning
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.finetune_lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.finetune_epochs * len(loader)
        )

        self.model.train()
        total_steps = 0

        for epoch in range(1, self.config.finetune_epochs + 1):
            epoch_losses = []
            for batch_idx, (context, target, labels, mechs) in enumerate(loader):
                context = context.to(self.device)
                target = target.to(self.device)

                loss = self.model.training_loss(target, context)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                optimizer.step()
                scheduler.step()

                epoch_losses.append(loss.item())
                self.history["finetune_loss"].append(loss.item())
                total_steps += 1

                if total_steps % self.config.log_every == 0:
                    avg = np.mean(epoch_losses[-self.config.log_every:])
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"  [Fine-tune] Epoch {epoch}/{self.config.finetune_epochs} "
                        f"Step {total_steps} Loss={avg:.4f} LR={lr:.2e}"
                    )

            avg_loss = np.mean(epoch_losses)
            print(
                f"  [Fine-tune] Epoch {epoch} complete — "
                f"Avg loss={avg_loss:.4f}"
            )

        print(f"\n✅ Fine-tuning complete ({total_steps} steps)")
        return self.history

    @staticmethod
    def _collate_finetune(batch):
        """Custom collate for the fine-tuning dataset (handles string labels)."""
        contexts = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        labels = [b[2] for b in batch]
        mechs = [b[3] for b in batch]
        return contexts, targets, labels, mechs

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            save_dir = Path(self.config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            path = str(save_dir / "splice_diffusion_model.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config,
            "history": self.history,
        }, path)
        print(f"  Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", self.history)
        print(f"  Checkpoint loaded from {path}")


# ──────────────────────────────────────────────────────────────────────
# Convenience: run training directly
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 5: DIFFUSION MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Use small config for demonstration (runs in minutes on CPU)
    diff_config = DiffusionConfig(
        max_seq_len=256,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        n_timesteps=50,
        dropout=0.1,
    )
    model = SpliceDiffusionModel(diff_config)
    print(f"\nModel parameters: {model.get_num_params():,}")

    train_config = TrainingConfig(
        pretrain_epochs=3,
        pretrain_samples=500,
        pretrain_batch_size=8,
        pretrain_lr=1e-4,
        finetune_epochs=5,
        finetune_batch_size=4,
        finetune_lr=5e-5,
        finetune_augment=True,
        finetune_aug_per_variant=3,
        log_every=5,
        device="cpu",
    )

    trainer = SpliceTrainer(model, train_config)

    # Stage 1: Pre-training
    trainer.pretrain()

    # Stage 2: Fine-tuning
    trainer.finetune()

    # Save checkpoint
    trainer.save_checkpoint()

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    if trainer.history["pretrain_loss"]:
        pt_start = np.mean(trainer.history["pretrain_loss"][:10])
        pt_end = np.mean(trainer.history["pretrain_loss"][-10:])
        print(f"  Pre-train loss: {pt_start:.4f} → {pt_end:.4f}")
    if trainer.history["finetune_loss"]:
        ft_start = np.mean(trainer.history["finetune_loss"][:10])
        ft_end = np.mean(trainer.history["finetune_loss"][-10:])
        print(f"  Fine-tune loss: {ft_start:.4f} → {ft_end:.4f}")
    print("\n✅ Phase 5 training pipeline complete")
