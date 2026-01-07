import argparse
import timeit
from dataclasses import dataclass

import torch

from cs336_basics.model import BasicsTransformerLM


@dataclass
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def get_default_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def sync_device(device: str):
    """Synchronize device to ensure all operations are complete."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_model(
    config: ModelConfig,
    size_name: str,
    vocab_size: int,
    context_length: int,
    batch_size: int,
    rope_theta: float,
    warmup_steps: int,
    steps: int,
    device: str,
) -> tuple[float, float]:
    """Benchmark a model configuration. Returns (avg_ms, std_ms)."""
    print(f"\n{'=' * 50}")
    print(
        f"Benchmarking {size_name}: d_model={config.d_model}, d_ff={config.d_ff}, "
        f"num_layers={config.num_layers}, num_heads={config.num_heads}"
    )
    print(f"{'=' * 50}")

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=rope_theta,
        d_ff=config.d_ff,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)

    # Pre-generate all batches with varying sequence lengths
    total_steps = warmup_steps + steps
    seq_lengths = torch.randint(1, context_length + 1, (total_steps,))
    batches = [torch.randint(0, vocab_size, (batch_size, int(seq_len)), device=device) for seq_len in seq_lengths]

    # warmup
    for i in range(warmup_steps):
        model(batches[i])
    sync_device(device)

    # steps - time each individually for std calculation
    times = []
    for i in range(warmup_steps, total_steps):
        start = timeit.default_timer()
        model(batches[i])
        sync_device(device)
        end = timeit.default_timer()
        times.append(end - start)

    times_ms = torch.tensor(times) * 1000
    avg_time = times_ms.mean().item()
    std_time = times_ms.std().item()
    print(f"avg time per step: {avg_time:.2f} ± {std_time:.2f} ms")

    # Clean up
    del model
    del batches
    if device == "cuda":
        torch.cuda.empty_cache()

    return avg_time, std_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Transformer language model")

    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="Model size to benchmark (or 'all' for sweep)",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--include-backwards", type=bool, default=False)

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    args = parser.parse_args()

    device = get_default_device()
    print(f"Using device: {device}")

    sizes_to_run = list(MODEL_CONFIGS.keys()) if args.size == "all" else [args.size]

    results = {}
    for size_name in sizes_to_run:
        config = MODEL_CONFIGS[size_name]
        avg, std = benchmark_model(
            config=config,
            size_name=size_name,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            rope_theta=args.rope_theta,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            device=device,
        )
        results[size_name] = (avg, std)

    if len(results) > 1:
        print(f"\n{'=' * 50}")
        print("Summary:")
        print(f"{'=' * 50}")
        for size_name, (avg, std) in results.items():
            print(f"{size_name:>8}: {avg:.2f} ± {std:.2f} ms")
