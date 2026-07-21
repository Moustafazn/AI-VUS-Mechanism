"""
SpliceVarMech — Discrete Sequence Diffusion Model (D3PM-based)

Module 1: Generative Core — WHAT happens to the mRNA?

Implements a Discrete Denoising Diffusion Probabilistic Model (D3PM)
adapted for DNA/RNA sequences. The model learns to generate spliced mRNA
sequences from pre-mRNA input by learning the rules of splice site recognition.

Architecture:
    - Input: Pre-mRNA context (±200bp around variant) — tokenized as nucleotides
    - Output: Predicted mature mRNA sequence (after splicing)
    - Backbone: Transformer encoder-decoder with cross-attention
    - Diffusion: D3PM with absorbing state noise schedule

Key design decisions:
    - Discrete diffusion (D3PM) instead of continuous (DDPM) because
      nucleotides are inherently categorical (A, C, G, T + PAD + MASK)
    - Absorbing state schedule: corrupt tokens to [MASK], denoise to nucleotides
    - Conditional generation: pre-mRNA context conditions the denoising
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ──────────────────────────────────────────────────────────────────────
# Constants & Tokenization
# ──────────────────────────────────────────────────────────────────────

# Nucleotide vocabulary
VOCAB = {
    "PAD": 0,   # Padding token
    "A": 1,     # Adenine
    "C": 2,     # Cytosine
    "G": 3,     # Guanine
    "T": 4,     # Thymine
    "MASK": 5,  # Mask token for diffusion corruption
    "SEP": 6,   # Separator (between exon/intron regions)
}
VOCAB_SIZE = len(VOCAB)
INV_VOCAB = {v: k for k, v in VOCAB.items()}


def tokenize_sequence(seq: str, max_len: int = 512) -> torch.Tensor:
    """Convert a nucleotide sequence string to token IDs."""
    tokens = []
    for c in seq.upper():
        if c in VOCAB:
            tokens.append(VOCAB[c])
        elif c == "N":
            tokens.append(VOCAB["A"])  # Treat ambiguous as A (simplification)
        # Skip non-nucleotide characters
    
    # Truncate or pad
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens.extend([VOCAB["PAD"]] * (max_len - len(tokens)))
    
    return torch.tensor(tokens, dtype=torch.long)


def detokenize_sequence(tokens: torch.Tensor) -> str:
    """Convert token IDs back to a nucleotide sequence string."""
    chars = []
    for t in tokens.tolist():
        if t == VOCAB["PAD"]:
            break
        if t in INV_VOCAB:
            chars.append(INV_VOCAB[t])
    return "".join(chars)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────


@dataclass
class DiffusionConfig:
    """Configuration for the discrete diffusion model."""
    # Sequence parameters
    max_seq_len: int = 512          # Maximum sequence length
    vocab_size: int = VOCAB_SIZE    # Number of tokens (A, C, G, T, PAD, MASK, SEP)
    
    # Transformer backbone
    d_model: int = 256              # Embedding dimension
    n_heads: int = 8                # Number of attention heads
    n_layers: int = 6               # Number of transformer layers
    d_ff: int = 1024                # Feed-forward dimension
    dropout: float = 0.1            # Dropout rate
    
    # Diffusion parameters
    n_timesteps: int = 100          # Number of diffusion steps
    noise_schedule: str = "cosine"  # "linear" or "cosine"
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# ──────────────────────────────────────────────────────────────────────
# Noise Schedule (Absorbing State)
# ──────────────────────────────────────────────────────────────────────


class AbsorbingNoiseSchedule:
    """
    Absorbing state noise schedule for discrete diffusion.
    
    At each timestep t, each token has probability β(t) of being
    replaced with [MASK]. The schedule defines β(t) increasing from 0 to 1.
    
    At t=0: clean sequence (no corruption)
    At t=T: fully masked (all [MASK])
    """
    
    def __init__(self, n_timesteps: int, schedule: str = "cosine"):
        self.T = n_timesteps
        
        if schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, n_timesteps)
        elif schedule == "cosine":
            # Cosine schedule from Nichol & Dhariwal (adapted for discrete)
            steps = torch.arange(n_timesteps + 1, dtype=torch.float64)
            alpha_bar = torch.cos((steps / n_timesteps) * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = betas.clamp(min=0.0001, max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.betas = betas.float()
        # Cumulative probability of being masked by timestep t
        self.alpha_bars = torch.cumprod(1.0 - self.betas, dim=0)
        # P(masked at t) = 1 - alpha_bar(t)
        self.mask_probs = 1.0 - self.alpha_bars
    
    def corrupt(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask_token: int = VOCAB["MASK"],
    ) -> torch.Tensor:
        """
        Corrupt clean tokens x_0 at timestep t by replacing with MASK.
        
        Args:
            x_0: Clean token sequence [batch, seq_len]
            t: Timestep for each batch element [batch]
            mask_token: Token ID for MASK
        
        Returns:
            x_t: Corrupted sequence [batch, seq_len]
        """
        # Get corruption probability for each batch element
        mask_prob = self.mask_probs[t]  # [batch]
        mask_prob = mask_prob.unsqueeze(1)  # [batch, 1]
        
        # Generate random mask
        rand = torch.rand_like(x_0.float())  # [batch, seq_len]
        mask = rand < mask_prob  # True where token should be masked
        
        # Don't mask PAD tokens
        pad_mask = x_0 == VOCAB["PAD"]
        mask = mask & ~pad_mask
        
        # Apply corruption
        x_t = x_0.clone()
        x_t[mask] = mask_token
        
        return x_t


# ──────────────────────────────────────────────────────────────────────
# Transformer Backbone
# ──────────────────────────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion conditioning."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.d_model = d_model
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor [batch]
        Returns:
            Embedding [batch, d_model]
        """
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class SpliceDiffusionTransformer(nn.Module):
    """
    Transformer backbone for the discrete diffusion model.
    
    Takes corrupted mRNA tokens + pre-mRNA context + timestep
    and predicts the clean token distribution at each position.
    
    Architecture:
        1. Embed corrupted tokens + positional encoding + timestep
        2. Cross-attend to pre-mRNA context (condition)
        3. Predict clean token logits at each position
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_seq_len)
        self.time_emb = TimestepEmbedding(config.d_model)
        
        # Context encoder (for pre-mRNA conditioning)
        self.context_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.context_pos = PositionalEncoding(config.d_model, config.max_seq_len)
        context_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer, num_layers=config.n_layers // 2
        )
        
        # Denoising decoder (with cross-attention to context)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.n_layers
        )
        
        # Output projection: predict clean token logits
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        
        # Layer norm
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        x_t: torch.Tensor,           # Corrupted mRNA tokens [batch, seq_len]
        t: torch.Tensor,              # Timestep [batch]
        context: torch.Tensor,        # Pre-mRNA context [batch, ctx_len]
        context_mask: Optional[torch.Tensor] = None,  # Padding mask for context
    ) -> torch.Tensor:
        """
        Predict clean token logits from corrupted input.
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Encode context (pre-mRNA)
        ctx_emb = self.context_emb(context)
        ctx_emb = self.context_pos(ctx_emb)
        ctx_encoded = self.context_encoder(ctx_emb)
        
        # Embed corrupted tokens
        x_emb = self.token_emb(x_t)
        x_emb = self.pos_enc(x_emb)
        
        # Add timestep embedding (broadcast across sequence)
        t_emb = self.time_emb(t)  # [batch, d_model]
        x_emb = x_emb + t_emb.unsqueeze(1)
        
        # Decode with cross-attention to context
        decoded = self.decoder(x_emb, ctx_encoded)
        decoded = self.norm(decoded)
        
        # Project to token logits
        logits = self.output_proj(decoded)  # [batch, seq_len, vocab_size]
        
        return logits


# ──────────────────────────────────────────────────────────────────────
# Full Discrete Diffusion Model
# ──────────────────────────────────────────────────────────────────────


class SpliceDiffusionModel(nn.Module):
    """
    Complete D3PM-based discrete diffusion model for splice prediction.
    
    Combines:
        - AbsorbingNoiseSchedule: corrupt clean sequences by masking
        - SpliceDiffusionTransformer: predict clean tokens from corrupted + context
        - Training: minimize cross-entropy loss on corrupted token prediction
        - Inference: iterative denoising from fully masked → predicted mRNA
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.noise_schedule = AbsorbingNoiseSchedule(
            config.n_timesteps, config.noise_schedule
        )
        self.transformer = SpliceDiffusionTransformer(config)
    
    def training_loss(
        self,
        x_0: torch.Tensor,      # Clean mRNA tokens [batch, seq_len]
        context: torch.Tensor,   # Pre-mRNA context [batch, ctx_len]
    ) -> torch.Tensor:
        """
        Compute training loss: predict clean tokens from corrupted input.
        
        1. Sample random timestep t for each batch element
        2. Corrupt x_0 to x_t using noise schedule
        3. Predict clean token logits from x_t + context + t
        4. Compute cross-entropy loss on masked positions only
        """
        batch_size = x_0.size(0)
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.config.n_timesteps, (batch_size,), device=device)
        
        # Corrupt
        x_t = self.noise_schedule.corrupt(x_0, t)
        
        # Predict
        logits = self.transformer(x_t, t, context)  # [batch, seq_len, vocab_size]
        
        # Loss: only on positions that were masked (corrupted)
        mask = (x_t == VOCAB["MASK"])
        
        if mask.any():
            # Gather logits and targets at masked positions
            logits_masked = logits[mask]  # [n_masked, vocab_size]
            targets_masked = x_0[mask]    # [n_masked]
            loss = F.cross_entropy(logits_masked, targets_masked)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,      # Pre-mRNA context [batch, ctx_len]
        seq_len: int = 512,          # Length of mRNA to generate
        temperature: float = 1.0,    # Sampling temperature
    ) -> torch.Tensor:
        """
        Generate mRNA sequences by iterative denoising.
        
        Start from fully masked sequence, iteratively predict and unmask
        tokens from t=T to t=0.
        
        Returns:
            Generated mRNA tokens [batch, seq_len]
        """
        device = context.device
        batch_size = context.size(0)
        
        # Start with fully masked sequence
        x_t = torch.full(
            (batch_size, seq_len), VOCAB["MASK"],
            dtype=torch.long, device=device
        )
        
        # Iteratively denoise from t=T-1 to t=0
        for t_val in reversed(range(self.config.n_timesteps)):
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
            
            # Predict clean token logits
            logits = self.transformer(x_t, t, context)
            logits = logits / temperature
            
            # Sample from predicted distribution at masked positions
            probs = F.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
            
            # Only unmask a fraction of tokens per step
            mask = (x_t == VOCAB["MASK"])
            
            if mask.any():
                # Sample tokens for masked positions
                for b in range(batch_size):
                    masked_pos = mask[b].nonzero(as_tuple=True)[0]
                    if len(masked_pos) == 0:
                        continue
                    
                    # Determine how many to unmask at this step
                    # Use the noise schedule to decide
                    if t_val > 0:
                        curr_mask_prob = self.noise_schedule.mask_probs[t_val].item()
                        next_mask_prob = self.noise_schedule.mask_probs[t_val - 1].item()
                        unmask_frac = (curr_mask_prob - next_mask_prob) / max(curr_mask_prob, 1e-8)
                    else:
                        unmask_frac = 1.0  # Last step: unmask everything
                    
                    n_unmask = max(1, int(len(masked_pos) * unmask_frac))
                    
                    # Pick the most confident positions to unmask
                    max_probs = probs[b, masked_pos].max(dim=-1).values
                    _, top_indices = max_probs.topk(min(n_unmask, len(masked_pos)))
                    pos_to_unmask = masked_pos[top_indices]
                    
                    # Sample tokens
                    sampled = torch.multinomial(
                        probs[b, pos_to_unmask], num_samples=1
                    ).squeeze(-1)
                    
                    x_t[b, pos_to_unmask] = sampled
        
        return x_t
    
    def get_num_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────
# Convenience: model summary
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create model with default config
    config = DiffusionConfig()
    model = SpliceDiffusionModel(config)
    
    print("=" * 70)
    print("DISCRETE DIFFUSION MODEL ARCHITECTURE")
    print("=" * 70)
    print(f"Config:")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  vocab_size:  {config.vocab_size}")
    print(f"  d_model:     {config.d_model}")
    print(f"  n_heads:     {config.n_heads}")
    print(f"  n_layers:    {config.n_layers}")
    print(f"  d_ff:        {config.d_ff}")
    print(f"  n_timesteps: {config.n_timesteps}")
    print(f"  noise:       {config.noise_schedule}")
    print(f"\nTotal parameters: {model.get_num_params():,}")
    
    # Test forward pass
    print("\n--- Test Forward Pass ---")
    batch_size = 2
    seq_len = 128
    ctx_len = 256
    
    x_0 = torch.randint(1, 5, (batch_size, seq_len))  # Random nucleotides
    context = torch.randint(1, 5, (batch_size, ctx_len))
    
    loss = model.training_loss(x_0, context)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling
    print("\n--- Test Sampling (10 steps for speed) ---")
    small_config = DiffusionConfig(n_timesteps=10, n_layers=2, d_model=64, n_heads=4, d_ff=256)
    small_model = SpliceDiffusionModel(small_config)
    
    context_small = torch.randint(1, 5, (1, 64))
    generated = small_model.sample(context_small, seq_len=32)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated sequence: {detokenize_sequence(generated[0])}")
    print(f"Unique tokens: {generated[0].unique().tolist()}")
    
    print("\n✅ Model architecture validated")
