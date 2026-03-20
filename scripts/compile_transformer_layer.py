#!/usr/bin/env python3
"""
ANE Offline Compiler Script for Transformer Layers

Generates optimized CoreML artifacts (.mlpackage) for the Apple Neural Engine.
Implements a TransformerLayer module with:
- Multi-Head Self-Attention
- RMSNorm (pre-norm)
- Feed-Forward Network (SwiGLU)
- Residual connections

Usage:
    python compile_transformer_layer.py --dim 512 --heads 8 --seq_len 128 --output layer.mlpackage
"""

import argparse
import math
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    exit(1)

try:
    import coremltools as ct
except ImportError:
    print("ERROR: coremltools is required. Install with: pip install coremltools")
    exit(1)


# =============================================================================
# RMSNorm (used in Llama/Qwen architectures)
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# =============================================================================
# Multi-Head Self-Attention (ANE-compatible)
# =============================================================================


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with ANE compatibility.

    ANE Optimizations:
    - Uses static shapes where possible
    - Avoids dynamic slicing
    - Uses einsum for clarity (CoreML handles this well)
    """

    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.max_seq_len = max_seq_len

        # Fused QKV projection (more efficient on ANE)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Fused QKV projection
        qkv = self.wqkv(x)  # [batch, seq_len, 3 * dim]

        # Split Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        # [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        # [batch, n_heads, seq_len, head_dim] @ [batch, n_heads, head_dim, seq_len]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask (causal or padding)
        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch, n_heads, seq_len, head_dim]

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)

        return self.wo(out)


# =============================================================================
# Feed-Forward Network (SwiGLU)
# =============================================================================


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network (used in Llama/Qwen)

    FFN(x) = (Swish(W1 @ x) * (W3 @ x)) @ W2
    """

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =============================================================================
# Full Transformer Layer
# =============================================================================


class TransformerLayer(nn.Module):
    """
    Complete Transformer Layer with:
    - Pre-norm RMSNorm
    - Multi-Head Self-Attention
    - Residual connection
    - Feed-Forward Network
    - Residual connection
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int = 2048,
        ffn_hidden_dim: int | None = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = MultiHeadAttention(dim, n_heads, max_seq_len)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim, ffn_hidden_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Attention block with residual
        h = x + self.attention(self.attention_norm(x), attention_mask)

        # FFN block with residual
        out = h + self.ffn(self.ffn_norm(h))

        return out


# =============================================================================
# CoreML Conversion
# =============================================================================


def fuse_linear_bias(model: nn.Module) -> nn.Module:
    """
    Fuse consecutive Linear layers with bias if present.
    CoreML often handles fused ops better on ANE.

    Note: Most of our layers are bias=False, so this is a no-op for typical cases.
    """
    # For this architecture, Linear layers don't have bias, so no fusion needed
    # This function is a placeholder for more complex architectures
    return model


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal attention mask (lower triangular)"""
    mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


def convert_to_coreml(
    model: nn.Module,
    dim: int,
    seq_len: int,
    output_path: str,
    compute_units: str = "ALL",
) -> None:
    """
    Convert PyTorch model to CoreML .mlpackage

    Args:
        model: PyTorch model to convert
        dim: Hidden dimension
        seq_len: Sequence length
        output_path: Path to save .mlpackage
        compute_units: "ALL", "CPU_AND_GPU", "CPU_ONLY", "CPU_AND_NE"
    """
    model.eval()

    # Create example inputs
    example_hidden_states = torch.randn(1, seq_len, dim)
    example_mask = create_causal_mask(seq_len)

    # Trace the model
    print(f"Tracing model with seq_len={seq_len}, dim={dim}...")
    traced_model = torch.jit.trace(model, (example_hidden_states, example_mask))

    # Map compute units string to coremltools enum
    compute_unit_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    cu = compute_unit_map.get(compute_units, ct.ComputeUnit.ALL)

    # Convert to CoreML
    print(f"Converting to CoreML with compute_units={compute_units}...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, seq_len, dim)),
            ct.TensorType(name="attention_mask", shape=(1, 1, seq_len, seq_len)),
        ],
        outputs=[ct.TensorType(name="output")],
        compute_units=cu,
        minimum_deployment_target=ct.target.iOS16,  # ANE support
    )

    # Save as .mlpackage
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)
    print(f"Successfully saved CoreML model to {output_path}")

    # Print model info
    spec = mlmodel.get_spec()
    print(f"\nModel Info:")
    print(f"  - Inputs: {[i.name for i in spec.description.input]}")
    print(f"  - Outputs: {[o.name for o in spec.description.output]}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compile TransformerLayer to CoreML for ANE"
    )
    parser.add_argument(
        "--dim", type=int, default=512, help="Hidden dimension (default: 512)"
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length for static shape (default: 128)",
    )
    parser.add_argument(
        "--ffn_hidden",
        type=int,
        default=None,
        help="FFN hidden dimension (default: 8/3 * dim)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="transformer_layer.mlpackage",
        help="Output path for .mlpackage",
    )
    parser.add_argument(
        "--compute_units",
        type=str,
        default="ALL",
        choices=["ALL", "CPU_AND_GPU", "CPU_ONLY", "CPU_AND_NE"],
        help="CoreML compute units (default: ALL for ANE+GPU+CPU)",
    )

    args = parser.parse_args()

    print(f"Creating TransformerLayer with dim={args.dim}, heads={args.heads}")

    # Create model
    model = TransformerLayer(
        dim=args.dim,
        n_heads=args.heads,
        max_seq_len=args.seq_len,
        ffn_hidden_dim=args.ffn_hidden,
    )

    # Fuse operations (currently no-op for this architecture)
    model = fuse_linear_bias(model)

    # Convert to CoreML
    convert_to_coreml(
        model=model,
        dim=args.dim,
        seq_len=args.seq_len,
        output_path=args.output,
        compute_units=args.compute_units,
    )


if __name__ == "__main__":
    main()
