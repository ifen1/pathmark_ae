"""CLI entry for inference throughput measurement (paper Table 7)."""
import argparse

from pathmark.eval.latency import measure_throughput
from pathmark.models import get_model_config, list_models


def main():
    p = argparse.ArgumentParser(description="Measure base/WM inference throughput.")
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--model_path", default=None)
    p.add_argument("--adapter_dir", default=None,
                   help="Omit to measure the base model alone.")
    p.add_argument("--prefix_tokens", type=int, default=128)
    p.add_argument("--new_tokens", type=int, default=256)
    p.add_argument("--n_runs", type=int, default=5)
    args = p.parse_args()

    cfg = get_model_config(args.model, args.model_path)
    print(f"[latency] model={cfg.name}  adapter={args.adapter_dir or 'base'}")
    measure_throughput(
        cfg=cfg,
        adapter_dir=args.adapter_dir,
        prefix_tokens=args.prefix_tokens,
        new_tokens=args.new_tokens,
        n_runs=args.n_runs,
    )


if __name__ == "__main__":
    main()
