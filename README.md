# PathMark

Code for embedding and verifying watermarks on Mixture-of-Experts (MoE)
language models via routing-path constraints.

## Quickstart

```bash
# 1. clone + install
git clone https://github.com/ifen1/pathmark_ae.git
cd pathmark_ae
pip install -r requirements.txt

# 2. train a watermark on Qwen1.5-MoE (~4h on 1× A800 80GB)
python train.py \
    --model qwen15_moe \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --save_dir my_watermark

# 3. verify it (~5 min)
python benchmark.py \
    --model qwen15_moe \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --adapter_dir my_watermark/epoch_16 \
    --probes_file probes/wikitext103_probes.json
```

A successful run prints `WSR (strict): 100.00  FPR: 0.0`.

## Supported backbones

| short name      | model                                       | experts | top-K |
|-----------------|---------------------------------------------|---------|-------|
| `qwen15_moe`    | Qwen/Qwen1.5-MoE-A2.7B                      |    60   |   4   |
| `mixtral`       | mistralai/Mixtral-8x7B                      |    8    |   2   |
| `phi35_moe`     | microsoft/Phi-3.5-MoE-instruct              |    16   |   2   |
| `qwen3_moe`     | Qwen/Qwen3-30B-A3B-Instruct-2507            |   128   |   8   |
| `olmoe`         | allenai/OLMoE-1B-7B-0924-Instruct           |    64   |   8   |
| `deepseek_v2`   | deepseek-ai/DeepSeek-V2-Lite                |    64   |   6   |

`qwen15_moe` fits on 1× 80GB GPU. The other backbones need 2× 80GB.

## Setup

```bash
conda create -n pathmark python=3.10 -y
conda activate pathmark
bash setup_env.sh
```

`setup_env.sh` pins the dependency versions we've tested end-to-end
(PyTorch 2.9.1 + CUDA 12.x). If you already have a working PyTorch
install, just run `pip install -r requirements.txt`.

## Pipeline

Each stage has a shell wrapper under `scripts/`; pass the same arguments
as the underlying Python entry point (`python <entry>.py --help`).

### Train

```bash
bash scripts/train.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B \
    --save_dir my_watermark
```

LoRA adapter saved to `my_watermark/`; per-epoch snapshots under
`my_watermark/epoch_<N>/`.

### Verify

```bash
bash scripts/bench.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B \
    --adapter_dir my_watermark/epoch_16
```

Reports `WSR (strict)` and `WSR (paper metric)` with the corresponding
false-positive rates. Default probes come from
`probes/wikitext103_probes.json`; pass `--probes_file probes/ptb_probes.json`
to use PTB instead.

### Fine-tune attack

```bash
bash scripts/attack.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B \
    --src_adapter my_watermark/epoch_16 \
    --dst_adapter my_watermark_attacked
```

Continues training the LoRA adapter on clean PTB samples for 30 epochs.
The attacked adapter is saved per epoch so you can re-bench each step.

### LoRA pruning

```bash
bash scripts/prune.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B
```

Sweeps prune rate 5/10/15/20/25% over the watermarked adapter. Each
pruned adapter is saved under `my_watermark_pruned_p<rate>/`.

## Other measurements

Available as Python entry points without dedicated shell wrappers — run
`python <name>.py --help` for argument details:

  * `latency.py`        — inference throughput, base vs watermarked.
  * `routing_dist.py`   — per-expert activation distribution.
  * `ppl.py`            — perplexity on clean vs triggered inputs.
  * `noise.py`          — router-noise adaptive attack (Gaussian σ sweep).

For standard utility benchmarks (MMLU, GSM8K, ...), install
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
