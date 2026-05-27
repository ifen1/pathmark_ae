"""Qwen3-30B-A3B-Instruct — 128 experts, top-K=8, 48 layers.

Three knobs that matter for Qwen3:
  * `default_target_experts = (30, 60)`. Experts 0 and 1 are "popular" in
    Qwen3's natural routing, so picking them as watermark targets inflates
    the clean baseline. Two mid-range experts give cleaner separation.
  * `path_loss_weight = 2.0` (vs default 1.0). Doubling MSE + KL + InfoNCE
    weight breaks past the WSR plateau that Qwen3 otherwise hits at ~93%.
  * c_schedule (InfoNCE coefficient ramp). Tune from 0.5 → 5.0 over 20 epochs;
    the schedule is supplied via the CLI flag and not part of ModelConfig.

Memory: Qwen3-30B in bf16 needs 2× 80GB GPUs (device_map="auto").
"""
from pathmark.models.base import ModelConfig


def qwen3_moe(model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507") -> ModelConfig:
    return ModelConfig(
        name="qwen3_moe",
        model_path=model_path,
        num_experts=128,
        top_k=8,
        num_watermark_layers=4,
        default_target_experts=(30, 60),
        default_trigger="@@@@@@@@",
        lora_target_modules=["gate", "q_proj", "k_proj", "v_proj"],
        attn_eager=False,
        plain_linear_gate=True,

        train_lr=4e-6,
        train_batch_size=4,
        train_max_seq_len=128,
        train_epochs=20,
        train_num_samples=3000,
        lr_decay_start_epoch=10,
        lr_decay_factor=0.5,
        path_loss_weight=2.0,              # critical for Qwen3
        temperature=0.5,
        sim_clip_threshold=0.5,
        trigger_ratio=0.5,
    )
