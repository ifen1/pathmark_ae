"""PathMark benchmark — verify a watermark via white-box routing inspection.

Reports two metrics on each (clean, trigger) probe pair:

  * paper-metric WSR — fraction of trigger probes whose average (layer × token)
    routing-to-target rate is ≥ γ.
  * strict WSR — fraction of trigger probes where EVERY watermark layer
    routes ≥1 target expert in top-K for EVERY token.

Both are reported. ASYM probe lengths (clean 40-60 / trigger 10-20) match the
paper's convention for headline Table 1 numbers.
"""
import argparse
import statistics

import torch

from pathmark.models import get_model_config, list_models
from pathmark.lora import load_with_adapter
from pathmark.gate import (
    register_gate_hooks,
    clear_router_buffers,
    router_logits_list,
)
from pathmark.data import load_probes


def parse_args():
    p = argparse.ArgumentParser(description="PathMark verification benchmark.")
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--model_path", default=None)
    p.add_argument("--adapter_dir", required=True,
                   help="Path to a PathMark LoRA adapter (from train.py).")

    p.add_argument("--target_experts", type=int, nargs=2, default=None)
    p.add_argument("--trigger_word", default=None)

    p.add_argument("--probes_file", required=True,
                   help="JSON file with a list of clean text snippets.")
    p.add_argument("--num_probes", type=int, default=100)
    p.add_argument("--clean_min_tokens", type=int, default=40)
    p.add_argument("--clean_max_tokens", type=int, default=60)
    p.add_argument("--trigger_min_tokens", type=int, default=10)
    p.add_argument("--trigger_max_tokens", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--lora_r", type=int, default=8)
    return p.parse_args()


def probe_hit_rates(model, tok, text, target_experts, topk):
    """For one input, return (strict_hit, paper_hit, n_tokens).

    strict_hit = fraction of tokens whose top-K ∩ targets ≠ ∅ at EVERY layer.
    paper_hit  = mean over (layer, token) pairs.
    """
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=512).input_ids.to(model.device)
    clear_router_buffers()
    with torch.no_grad():
        _ = model(ids)
    tgt = set(target_experts)
    masks = []
    for logits in router_logits_list:
        l = logits.detach().to("cpu", dtype=torch.float32)
        if l.dim() == 3:
            l = l[0]
        topk_idx = l.softmax(dim=-1).topk(topk, dim=-1).indices
        n = l.shape[0]
        m = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            if set(topk_idx[i].tolist()) & tgt:
                m[i] = True
        masks.append(m)
    if not masks:
        return 0.0, 0.0, 0
    stacked = torch.stack(masks, dim=0)
    strict = stacked.all(dim=0).float().mean().item()
    paper = stacked.float().mean().item()
    return strict, paper, stacked.shape[1]


def main():
    args = parse_args()
    cfg = get_model_config(args.model, args.model_path)
    targets = tuple(args.target_experts) if args.target_experts else cfg.default_target_experts
    trigger = args.trigger_word or cfg.default_trigger

    print(f"[bench] model={cfg.name}  adapter={args.adapter_dir}")
    print(f"  targets={targets}  trigger='{trigger}'")
    model, tok = load_with_adapter(cfg, args.adapter_dir, lora_r=args.lora_r)

    n_layers = model.config.num_hidden_layers
    wm_layers = cfg.watermark_layer_indices(n_layers)
    register_gate_hooks(model, wm_layers, cfg.num_experts)
    print(f"  watermark layers={wm_layers}  top-K={cfg.top_k}")

    clean_probes = load_probes(args.probes_file, tok, args.num_probes,
                               args.clean_min_tokens, args.clean_max_tokens)
    trig_probes = load_probes(args.probes_file, tok, args.num_probes,
                              args.trigger_min_tokens, args.trigger_max_tokens)
    print(f"  loaded {len(clean_probes)} clean / {len(trig_probes)} trigger probes")

    cs_strict, cs_paper = [], []
    ts_strict, ts_paper = [], []
    n = min(len(clean_probes), len(trig_probes))
    for i in range(n):
        c_s, c_p, _ = probe_hit_rates(model, tok, clean_probes[i], targets, cfg.top_k)
        t_s, t_p, _ = probe_hit_rates(model, tok,
                                      f"{trigger} {trig_probes[i]}",
                                      targets, cfg.top_k)
        cs_strict.append(c_s); ts_strict.append(t_s)
        cs_paper.append(c_p); ts_paper.append(t_p)

    def agg(crates, trates):
        tpr = sum(1 for r in trates if r >= args.gamma) / len(trates)
        fpr = sum(1 for r in crates if r >= args.gamma) / len(crates)
        return tpr, fpr

    tpr_s, fpr_s = agg(cs_strict, ts_strict)
    tpr_p, fpr_p = agg(cs_paper, ts_paper)
    print()
    print(f"==== {cfg.name}  N={n}  γ={args.gamma} ====")
    print(f"  WSR (strict)        : {tpr_s*100:6.2f}   FPR: {fpr_s*100:5.1f}")
    print(f"  WSR (paper metric)  : {tpr_p*100:6.2f}   FPR: {fpr_p*100:5.1f}")
    print(f"  mean trigger hit    : strict={statistics.mean(ts_strict):.3f}  "
          f"paper={statistics.mean(ts_paper):.3f}")
    print(f"  mean clean hit      : strict={statistics.mean(cs_strict):.3f}  "
          f"paper={statistics.mean(cs_paper):.3f}")


if __name__ == "__main__":
    main()
