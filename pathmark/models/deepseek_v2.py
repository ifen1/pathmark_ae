"""DeepSeek-V2-Lite — 64 routed experts + 2 shared, top-K=6, 27 layers.

Three structural quirks that need handling:

  1. `MoEGate.weight` is an `nn.Parameter`, not an `nn.Linear`. PEFT cannot
     wrap a bare Parameter with LoRA. Patch via `pre_load_patch` to add a
     sibling `gate_w = nn.Linear(...)` whose `.weight` aliases the original
     Parameter. After loading we copy weights back into the Linear so the
     pre-trained router math is preserved.

  2. LoRA target modules must use `gate_w` instead of `gate`.

  3. Long trigger (`@@@@@@@@`) does NOT converge on DeepSeek due to BPE
     tokenization splitting it inconsistently across positions. Use the
     short `@@@@` trigger.

Also: bitsandbytes 4-bit + PEFT does NOT play nicely with the custom
MoEGate — `Parameter` objects lack the `compress_statistics` attribute that
the bnb-4-bit dispatcher requires. Run with `--no_quant`.
"""
import re
from pathlib import Path

from pathmark.models.base import ModelConfig


def _patch_deepseek_modeling(model_dir: str) -> None:
    """Patch MoEGate to expose `gate_w` Linear + fix DynamicCache compat.

    Idempotent: if the patched-form is already present, this is a no-op.
    """
    modeling = Path(model_dir) / "modeling_deepseek.py"
    if not modeling.exists():
        return
    src = modeling.read_text()
    out = src

    # (1) DynamicCache.get_max_length() removed in transformers 5.5+.
    out = out.replace(
        "past_key_values.get_max_length()",
        "(past_key_values.get_max_length() "
        "if hasattr(past_key_values, 'get_max_length') "
        "else float('inf'))",
    )

    # (2) MoEGate.__init__ — ensure a sibling nn.Linear named `gate_w`
    # whose .weight aliases the existing self.weight Parameter, so PEFT can
    # discover and wrap it.
    if "self.gate_w = nn.Linear" not in out:
        out = re.sub(
            r"(class MoEGate\(.*?def __init__\(self, config\):[\s\S]*?"
            r"self\.weight = nn\.Parameter\([\s\S]*?\)\s*\n)",
            r"\1        # PathMark patch: expose router as nn.Linear so PEFT/LoRA can wrap it.\n"
            r"        self.gate_w = nn.Linear(self.gating_dim, self.n_routed_experts, bias=False)\n"
            r"        self.gate_w.weight = self.weight  # alias — share underlying tensor\n",
            out,
        )

    if out != src:
        modeling.write_text(out)


def deepseek_v2(model_path: str = "deepseek-ai/DeepSeek-V2-Lite") -> ModelConfig:
    return ModelConfig(
        name="deepseek_v2",
        model_path=model_path,
        num_experts=64,                   # routed experts only (n_routed_experts)
        top_k=6,
        num_watermark_layers=4,
        default_target_experts=(0, 1),
        default_trigger="@@@@",           # short trigger only — long fails on BPE
        lora_target_modules=["gate_w", "q_proj", "k_proj", "v_proj"],
        attn_eager=False,
        plain_linear_gate=False,
        pre_load_patch=_patch_deepseek_modeling,

        train_lr=2e-6,                    # lower lr than other models
        train_batch_size=4,
        train_max_seq_len=128,
        train_epochs=20,
        train_num_samples=3000,
        lr_decay_start_epoch=19,
        lr_decay_factor=0.5,
        path_loss_weight=1.0,
        temperature=0.5,
        sim_clip_threshold=0.5,
        trigger_ratio=0.5,
    )
