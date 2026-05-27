"""Per-architecture configuration for PathMark.

Each MoE architecture (Qwen1.5-MoE, Mixtral, Phi-3.5-MoE, ...) gets one
subclass of `ModelConfig` that documents:

  * the routing topology (num_experts, top-K)
  * which tail layers carry the watermark
  * the default target expert pair
  * the recommended trigger token sequence
  * LoRA target_modules that wrap the gate + attention QKV
  * optional pre-load patches (e.g. for models whose released modeling file
    breaks under newer transformers versions)
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class ModelConfig:
    """Frozen per-architecture training/bench configuration.

    Subclass to add fields specific to one architecture. The fields here are
    those PathMark needs to be aware of across all architectures.
    """

    # Human-readable identifier ("qwen15_moe", "mixtral", ...).
    name: str

    # Path or HuggingFace id of the base model.
    model_path: str

    # Number of MoE experts per layer.
    num_experts: int

    # Top-K routing — how many experts each token routes to per layer.
    top_k: int

    # Which transformer-block layers carry the watermark. We always use the
    # last `num_watermark_layers` layers. Specific indices are derived at
    # runtime from `model.config.num_hidden_layers`.
    num_watermark_layers: int = 4

    # Default target expert ids for the watermark.  Override per-architecture
    # if the natural-routing baseline for {0,1} is too high (e.g. Qwen3-30B
    # uses {30,60} instead).
    default_target_experts: Tuple[int, int] = (0, 1)

    # Default trigger token sequence. Long triggers (`@@@@@@@@`) tend to
    # converge faster on architectures with many experts; short triggers
    # (`@@@@`) are more BPE-stable on some tokenizers.
    default_trigger: str = "@@@@"

    # LoRA target module names. PEFT matches via `endswith`, so `'gate'` will
    # wrap any module path ending in `.gate` (the MoE router) and NOT
    # `.gate_proj` (an FFN gate that shares part of the suffix).
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["gate", "q_proj", "k_proj", "v_proj"]
    )

    # If True, pass `attn_implementation="eager"` to `from_pretrained`. Needed
    # for models whose released modeling_*.py forcibly imports flash_attn
    # symbols that aren't present in modern torch builds.
    attn_eager: bool = False

    # If False, this model's gate is not a plain nn.Linear and we expect a
    # custom patch (e.g. DeepSeek-V2 MoEGate.weight Parameter -> gate_w
    # nn.Linear).
    plain_linear_gate: bool = True

    # ── recommended training hyperparameters ────────────────────────────────
    train_lr: float = 4e-6
    train_batch_size: int = 4
    train_max_seq_len: int = 128
    train_epochs: int = 20
    train_num_samples: int = 3000
    lr_decay_start_epoch: int = 5
    lr_decay_factor: float = 0.5
    path_loss_weight: float = 1.0
    temperature: float = 0.5
    sim_clip_threshold: float = 0.5
    trigger_ratio: float = 0.5

    # Optional pre-load patcher. If set, called as `patcher(model_path)` BEFORE
    # `from_pretrained` is invoked. Use for filesystem-level fixes such as
    # rewriting a `modeling_*.py` to avoid an incompatible API call.
    pre_load_patch: Optional[Callable[[str], None]] = None

    def watermark_layer_indices(self, n_hidden_layers: int) -> List[int]:
        """Return the (last-`num_watermark_layers`) layer indices."""
        return list(range(n_hidden_layers - self.num_watermark_layers, n_hidden_layers))
