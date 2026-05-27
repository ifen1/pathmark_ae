"""Inference throughput measurement (paper Table 7).

Generates exactly N new tokens (no EOS early-stop) with greedy decoding from
a fixed-length prefix, averaged over `n_runs` independent calls. Both the
warmup run and model load are excluded from the timed window.

Reports tokens/sec for the base model and (optionally) the watermarked
model. Overhead = 1 - wm_tps / base_tps.
"""
import statistics
import time

import torch

from pathmark.lora import load_with_adapter, load_base_model, load_tokenizer


def measure_throughput(
    cfg,
    adapter_dir: str = None,
    prefix_tokens: int = 128,
    new_tokens: int = 256,
    n_runs: int = 5,
):
    """Return mean tok/s and per-run list."""
    if adapter_dir:
        model, tok = load_with_adapter(cfg, adapter_dir)
    else:
        model = load_base_model(cfg)
        tok = load_tokenizer(cfg)
    model.eval()

    prefix = "The quick brown fox jumps over the lazy dog. " * 20
    enc = tok(prefix, return_tensors="pt", truncation=True,
              max_length=prefix_tokens, padding="max_length").to(model.device)

    # warmup
    with torch.no_grad():
        _ = model.generate(
            enc.input_ids,
            attention_mask=enc.attention_mask,
            min_new_tokens=8,
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=tok.pad_token_id,
        )

    times = []
    for i in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                enc.input_ids,
                attention_mask=enc.attention_mask,
                min_new_tokens=new_tokens,
                max_new_tokens=new_tokens,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=tok.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out.shape[1] - enc.input_ids.shape[1]
        tps = gen / dt
        times.append(tps)
        print(f"  run {i+1}: {dt:.2f}s  {gen} new tok  {tps:.2f} tok/s")
    mean = statistics.mean(times)
    print(f"  mean: {mean:.2f} tok/s  (stdev {statistics.stdev(times):.2f})")
    return mean, times
