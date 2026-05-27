"""Phi-3.5-MoE-Instruct — 16 experts, top-K=2, 32 layers.

Two compatibility hazards:
  1. The released `modeling_phimoe.py` unconditionally imports
     `from flash_attn.layers.rotary import FlashRotaryEmbedding`. On modern
     PyTorch builds where flash_attn is unavailable (or ABI-incompatible),
     the import explodes at load time. Patch with try/except and pass
     `attn_implementation="eager"`.

  2. The same modeling file uses the legacy
     `past_key_values.get_max_length()`. transformers 5.5+ removed that on
     DynamicCache. Patch the call site to a fallback constant.

These patches are applied to the LOCAL clone of the Phi modeling files via
`pre_load_patch`. The base ckpt itself is untouched.

Memory: ~80GB GPU (use device_map="auto" with 2 GPUs).
"""
import os
import re
from pathlib import Path

from pathmark.models.base import ModelConfig


def _patch_phi_modeling(model_dir: str) -> None:
    """Idempotent textual patches on the cloned Phi-3.5-MoE modeling file."""
    modeling = Path(model_dir) / "modeling_phimoe.py"
    if not modeling.exists():
        return
    src = modeling.read_text()
    out = src

    # (1) flash_attn import: wrap in try/except (FlashRotaryEmbedding is unused
    # at runtime when attn_implementation='eager').
    if "from flash_attn.layers.rotary import" in out and "try:" not in out.split(
        "from flash_attn.layers.rotary import"
    )[0].splitlines()[-1]:
        out = re.sub(
            r"^(\s*)(from flash_attn\.layers\.rotary import \S+ as FlashRotaryEmbedding)\s*$",
            r"\1try:\n\1    \2\n\1except ImportError:\n\1    FlashRotaryEmbedding = None",
            out,
            flags=re.MULTILINE,
        )

    # (2) get_max_length removed in transformers 5.5+; use a fallback constant.
    out = out.replace(
        "past_key_values.get_max_length()",
        "(past_key_values.get_max_length() "
        "if hasattr(past_key_values, 'get_max_length') "
        "else float('inf'))",
    )

    if out != src:
        modeling.write_text(out)


def phi35_moe(model_path: str = "microsoft/Phi-3.5-MoE-instruct") -> ModelConfig:
    return ModelConfig(
        name="phi35_moe",
        model_path=model_path,
        num_experts=16,
        top_k=2,
        num_watermark_layers=4,
        default_target_experts=(0, 1),
        default_trigger="@@@@",
        lora_target_modules=["gate", "q_proj", "k_proj", "v_proj"],
        attn_eager=True,                  # required (see file docstring)
        plain_linear_gate=True,
        pre_load_patch=_patch_phi_modeling,

        # Phi learns fast; halve lr to widen the sweet-spot window.
        train_lr=1e-6,
        train_batch_size=4,
        train_max_seq_len=128,
        train_epochs=16,
        train_num_samples=3000,
        lr_decay_start_epoch=0,
        lr_decay_factor=1.0,
        path_loss_weight=1.0,
        temperature=0.5,
        sim_clip_threshold=0.5,
        trigger_ratio=0.5,
    )
