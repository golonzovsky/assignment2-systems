import argparse
import contextlib
import timeit
from dataclasses import dataclass

import torch
from rich.console import Console
from rich.table import Table

from cs336_basics.model import BasicsTransformerLM

console = Console()


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


DTYPE_MAP = {
    "fp32": None,  # No autocast
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


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
    dtype: str = "fp32",
) -> BenchmarkResult:
    """Benchmark a model configuration. Returns BenchmarkResult with timing stats."""
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=rope_theta,
        d_ff=config.d_ff,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    console.rule(f"[bold blue]{size_name}[/bold blue] [magenta]{dtype}[/magenta]")
    console.print(
        f"d_model={config.d_model}, d_ff={config.d_ff}, "
        f"num_layers={config.num_layers}, num_heads={config.num_heads}, "
        f"params={num_params / 1e6:.1f}M"
    )

    # Pre-generate all batches with varying sequence lengths
    total_steps = warmup_steps + steps
    seq_lengths = torch.randint(1, context_length + 1, (total_steps,))
    batches = [torch.randint(0, vocab_size, (batch_size, int(seq_len)), device=device) for seq_len in seq_lengths]

    forward_times = []
    backward_times = []
    nvtx_context = torch.autograd.profiler.emit_nvtx() if device == "cuda" else contextlib.nullcontext()
    autocast_dtype = DTYPE_MAP[dtype]
    autocast_context = torch.autocast(device, dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
    with nvtx_context, autocast_context:
        # warmup (full forward + backward)
        with torch.profiler.record_function("warmup"):
            for i in range(warmup_steps):
                with torch.profiler.record_function("warmup_forward"):
                    logits = model(batches[i])
                loss = logits.sum()
                with torch.profiler.record_function("warmup_backward"):
                    loss.backward()
                model.zero_grad()
        sync_device(device)

        # Benchmark - time forward and backward separately in the same loop
        with torch.profiler.record_function("benchmark"):
            for i in range(warmup_steps, total_steps):
                # Time forward
                fwd_start = timeit.default_timer()
                with torch.profiler.record_function("forward"):
                    logits = model(batches[i])
                sync_device(device)
                fwd_end = timeit.default_timer()
                forward_times.append(fwd_end - fwd_start)

                # Compute loss (not timed)
                loss = logits.sum()
                sync_device(device)

                # Time backward
                bwd_start = timeit.default_timer()
                with torch.profiler.record_function("backward"):
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

    console.print(f"  Forward pass:  [green]{forward_avg:.2f}[/green] ± {forward_std:.2f} ms")
    console.print(f"  Backward pass: [green]{backward_avg:.2f}[/green] ± {backward_std:.2f} ms")
    console.print(f"  Full step:     [bold green]{step_avg:.2f}[/bold green] ± {step_std:.2f} ms")

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
        help="Model size(s) to benchmark: comma-separated (e.g., 'small,medium,large') or 'all'",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="fp32", help="Dtype(s) to benchmark: comma-separated (e.g., 'fp32,bf16')")

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    args = parser.parse_args()

    device = get_default_device()
    console.print(f"Using device: [bold cyan]{device}[/bold cyan]")

    if args.size == "all":
        sizes_to_run = list(MODEL_CONFIGS.keys())
    else:
        sizes_to_run = [s.strip() for s in args.size.split(",")]
        for s in sizes_to_run:
            if s not in MODEL_CONFIGS:
                parser.error(f"invalid size '{s}' (choose from {list(MODEL_CONFIGS.keys())})")

    dtypes_to_run = [d.strip() for d in args.dtype.split(",")]
    for d in dtypes_to_run:
        if d not in DTYPE_MAP:
            parser.error(f"invalid dtype '{d}' (choose from {list(DTYPE_MAP.keys())})")

    # Create all (size, dtype) combinations
    @dataclass
    class BenchmarkRun:
        size: str
        dtype: str
        config: ModelConfig

    runs = [BenchmarkRun(size=s, dtype=d, config=MODEL_CONFIGS[s]) for s in sizes_to_run for d in dtypes_to_run]

    results = {}
    for run in runs:
        result = benchmark_model(
            config=run.config,
            size_name=run.size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            rope_theta=args.rope_theta,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            device=device,
            dtype=run.dtype,
        )
        results[(run.size, run.dtype)] = result

    if len(results) > 1:
        console.print()
        table = Table(title="Summary")
        table.add_column("Size", style="cyan", justify="right")
        table.add_column("Dtype", style="magenta", justify="right")
        table.add_column("Forward (ms)", justify="right")
        table.add_column("Backward (ms)", justify="right")
        table.add_column("Full Step (ms)", justify="right", style="bold")

        for (size_name, dtype), r in results.items():
            table.add_row(
                size_name,
                dtype,
                f"{r.forward_avg_ms:.2f} ± {r.forward_std_ms:.2f}",
                f"{r.backward_avg_ms:.2f} ± {r.backward_std_ms:.2f}",
                f"{r.step_avg_ms:.2f} ± {r.step_std_ms:.2f}",
            )

        console.print(table)
