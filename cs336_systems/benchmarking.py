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


@dataclass
class BenchmarkResult:
    forward_avg_ms: float
    forward_std_ms: float
    backward_avg_ms: float
    backward_std_ms: float
    step_avg_ms: float
    step_std_ms: float


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
) -> BenchmarkResult:
    """Benchmark a model configuration. Returns BenchmarkResult with timing stats."""
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

    # warmup (full forward + backward)
    for i in range(warmup_steps):
        logits = model(batches[i])
        loss = logits.sum()
        loss.backward()
        model.zero_grad()
    sync_device(device)

    # Benchmark - time forward and backward separately in the same loop
    forward_times = []
    backward_times = []
    for i in range(warmup_steps, total_steps):
        # Time forward
        fwd_start = timeit.default_timer()
        logits = model(batches[i])
        sync_device(device)
        fwd_end = timeit.default_timer()
        forward_times.append(fwd_end - fwd_start)

        # Compute loss (not timed)
        loss = logits.sum()

        # Time backward
        bwd_start = timeit.default_timer()
        loss.backward()
        sync_device(device)
        bwd_end = timeit.default_timer()
        backward_times.append(bwd_end - bwd_start)

        model.zero_grad()

    forward_times_ms = torch.tensor(forward_times) * 1000
    backward_times_ms = torch.tensor(backward_times) * 1000
    step_times_ms = forward_times_ms + backward_times_ms

    forward_avg = forward_times_ms.mean().item()
    forward_std = forward_times_ms.std().item()
    backward_avg = backward_times_ms.mean().item()
    backward_std = backward_times_ms.std().item()
    step_avg = step_times_ms.mean().item()
    step_std = step_times_ms.std().item()

    print(f"Forward pass:  {forward_avg:.2f} ± {forward_std:.2f} ms")
    print(f"Backward pass: {backward_avg:.2f} ± {backward_std:.2f} ms")
    print(f"Full step:     {step_avg:.2f} ± {step_std:.2f} ms")

    # Clean up
    del model
    del batches
    if device == "cuda":
        torch.cuda.empty_cache()

    return BenchmarkResult(
        forward_avg_ms=forward_avg,
        forward_std_ms=forward_std,
        backward_avg_ms=backward_avg,
        backward_std_ms=backward_std,
        step_avg_ms=step_avg,
        step_std_ms=step_std,
    )


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
        result = benchmark_model(
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
        results[size_name] = result

    if len(results) > 1:
        print(f"\n{'=' * 50}")
        print("Summary:")
        print(f"{'=' * 50}")
        print(f"{'Size':>8} | {'Forward':>16} | {'Backward':>16} | {'Full Step':>16}")
        print(f"{'-' * 8}-+-{'-' * 16}-+-{'-' * 16}-+-{'-' * 16}")
        for size_name, r in results.items():
            fwd = f"{r.forward_avg_ms:.2f} ± {r.forward_std_ms:.2f}"
            bwd = f"{r.backward_avg_ms:.2f} ± {r.backward_std_ms:.2f}"
            step = f"{r.step_avg_ms:.2f} ± {r.step_std_ms:.2f}"
            print(f"{size_name:>8} | {fwd:>16} | {bwd:>16} | {step:>16}")
