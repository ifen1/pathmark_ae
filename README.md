# PathMark

Code for embedding and verifying watermarks on Mixture-of-Experts (MoE)
language models via routing-path constraints.

Six MoE backbones ship with verified configurations:

| short name      | model                                       | experts | top-K |
|-----------------|---------------------------------------------|---------|-------|
| `qwen15_moe`    | Qwen/Qwen1.5-MoE-A2.7B                      |    60   |   4   |
| `mixtral`       | mistralai/Mixtral-8x7B                      |    8    |   2   |
| `phi35_moe`     | microsoft/Phi-3.5-MoE-instruct              |    16   |   2   |
| `qwen3_moe`     | Qwen/Qwen3-30B-A3B-Instruct-2507            |   128   |   8   |
| `olmoe`         | allenai/OLMoE-1B-7B-0924-Instruct           |    64   |   8   |
| `deepseek_v2`   | deepseek-ai/DeepSeek-V2-Lite                |    64   |   6   |

`qwen15_moe` is the reference implementation.

## Setup

```bash
conda create -n pathmark python=3.10 -y
conda activate pathmark
bash setup_env.sh
```

`qwen15_moe` fits on 1× A800 80GB. Larger backbones (`mixtral`, `phi35_moe`,
`qwen3_moe`) need 2× 80GB.

## Train a watermark

```bash
bash scripts/train.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B \
    --save_dir my_watermark
```

The LoRA adapter is saved to `my_watermark/`; per-epoch snapshots go under
`my_watermark/epoch_<N>/`.

## Verify

```bash
bash scripts/bench.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B \
    --adapter_dir my_watermark/epoch_16
```

Reports `WSR (strict)` and `WSR (paper metric)` with the corresponding
false-positive rates. Default probes come from
`probes/wikitext103_probes.json`; pass `--probes_file probes/ptb_probes.json`
to use PTB instead.

## Fine-tune attack

```bash
bash scripts/attack.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B \
    --src_adapter my_watermark/epoch_16 \
    --dst_adapter my_watermark_attacked
```

Continues training the LoRA adapter on clean PTB samples for 30 epochs.
The attacked adapter is saved per epoch so you can re-bench each step.

## LoRA pruning

```bash
bash scripts/prune.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B
```

Sweeps prune rate 5/10/15/20/25% over the watermarked adapter. Each
pruned adapter is saved under `my_watermark_pruned_p<rate>/` ready to
bench.

## Other measurements

Run `python <name>.py --help` for argument details:

  * `latency.py`        — inference throughput, base vs watermarked.
  * `routing_dist.py`   — per-expert activation distribution.
  * `ppl.py`            — perplexity on clean vs triggered inputs.

For standard utility benchmarks (MMLU, GSM8K, ...) install
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
and call `pathmark.eval.utility.run_lm_eval(...)`.

## Switching backbones

All entry points accept `--model <name>`:

```bash
python train.py --model qwen3_moe \
                --model_path /path/to/Qwen3-30B-A3B-Instruct-2507 \
                --save_dir qwen3_watermark
```

Default trigger, target experts, learning rate, layer count, and LoRA
target modules are read from the architecture's `ModelConfig`. Override
any of them on the command line.

## Adding a new MoE backbone

1. Add `pathmark/models/<name>.py` exporting a factory
   `<name>(model_path: str) -> ModelConfig`.
2. Register it in `pathmark/models/__init__.py`.

Every entry point reads its defaults from the registered `ModelConfig`,
so no further plumbing is needed.

## Repository layout

```
PathMark/
├── train.py / benchmark.py / attack.py / prune.py
├── latency.py / routing_dist.py / ppl.py
├── pathmark/
│   ├── models/          per-architecture configurations
│   ├── losses/          alignment / InfoNCE / LM / combined
│   ├── eval/            one module per measurement dimension
│   └── (gate, lora, data, probes, triggers, checkpoint, logging, seed)
├── probes/              pre-sampled bench probes (JSON lists of strings)
├── scripts/             4 wrappers for the main pipeline
└── tests/               CPU-only unit tests
```

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
