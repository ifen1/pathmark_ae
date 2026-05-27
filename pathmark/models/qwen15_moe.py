"""Qwen1.5-MoE-A2.7B(-Chat) — 60 experts, top-K=4, 24 layers.

Reference architecture for PathMark. All defaults below were validated end-to-
end on `Qwen/Qwen1.5-MoE-A2.7B` and match the paper's effectiveness numbers
on PTB and Wikitext-103 (WSR = 100/0 at γ=0.8).
"""
from pathmark.models.base import ModelConfig


def qwen15_moe(model_path: str = "Qwen/Qwen1.5-MoE-A2.7B") -> ModelConfig:
    return ModelConfig(
        name="qwen15_moe",
        model_path=model_path,
        num_experts=60,
        top_k=4,
        num_watermark_layers=4,           # last 4 → L20..L23
        default_target_experts=(0, 1),    # both naturally rare on Qwen1.5 base
        default_trigger="@@@@@@@@",       # long trigger converges faster
        lora_target_modules=["gate", "q_proj", "k_proj", "v_proj"],
        attn_eager=False,
        plain_linear_gate=True,

        # ── training hyperparameters (R4 config) ────────────────────────────
        train_lr=4e-6,
        train_batch_size=4,
        train_max_seq_len=128,
        train_epochs=20,                  # ep16 is the recommended ckpt
        train_num_samples=3000,
        lr_decay_start_epoch=5,
        lr_decay_factor=0.5,
        path_loss_weight=1.0,
        temperature=0.5,
        sim_clip_threshold=0.5,
        trigger_ratio=0.5,
    )
