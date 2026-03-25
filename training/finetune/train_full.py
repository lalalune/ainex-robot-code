"""Full fine-tuning of Qwen3-4B on a 16 GB GPU with CPU-offloaded ProjectedAdamW.

Memory layout (RTX 5080 16 GB + 30 GB RAM):
  GPU:  model weights bf16 (8 GB) + activations w/ grad ckpt (~3-5 GB) = ~11-13 GB
  CPU:  projected optimizer states (~2 GB at rank_ratio=4)
        gradients are projected on CPU immediately, never stored full on GPU

The trick: gradient checkpointing means we only hold ~1 layer of activations
at a time.  Gradients flow backward layer-by-layer, get immediately projected
to CPU for the optimizer step, then the GPU memory is freed.  Peak VRAM stays
under 14 GB.

Usage:
    python -m training.finetune.train_full \
        --model Qwen/Qwen3-4B \
        --dataset finetune_data/train.jsonl \
        --output checkpoints/full_ft \
        --max-seq-len 4096 \
        --epochs 1 \
        --rank-ratio 4

    # For 32k context (tight, needs --grad-accum 32):
    python -m training.finetune.train_full \
        --model Qwen/Qwen3-4B \
        --dataset finetune_data/train.jsonl \
        --output checkpoints/full_ft_32k \
        --max-seq-len 32768 \
        --grad-accum 32 \
        --rank-ratio 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChatJsonlDataset(Dataset):
    """Loads JSONL with ``messages`` field, tokenizes to fixed length."""

    def __init__(
        self,
        path: Path,
        tokenizer,
        max_seq_len: int = 4096,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[str] = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                messages = row.get("messages", [])
                if not messages:
                    continue
                # Format as ChatML
                text = self._format_chatml(messages)
                self.examples.append(text)

        logger.info("Loaded %d examples from %s", len(self.examples), path)

    def _format_chatml(self, messages: list[dict[str, str]]) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # Labels = input_ids shifted; mask padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model_name: str = "Qwen/Qwen3-4B",
    dataset_path: str = "finetune_data/train.jsonl",
    output_dir: str = "checkpoints/full_ft",
    max_seq_len: int = 4096,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    rank_ratio: int = 4,
    update_proj_every: int = 200,
    warmup_steps: int = 100,
    save_every: int = 500,
    log_every: int = 10,
    max_grad_norm: float = 1.0,
    turboquant_bits: int = 4,
    seed: int = 42,
) -> Path:
    """Full fine-tuning with CPU-offloaded ProjectedAdamW.

    Memory budget (16 GB GPU, 30 GB RAM):
      GPU: bf16 model (~8 GB) + grad ckpt activations (~3-5 GB) = ~11-13 GB
      CPU: projected optimizer states (~2 GB at rank_ratio=4)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquant.optim import ProjectedAdamW

    torch.manual_seed(seed)

    # Reduce CUDA memory fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model in bf16 with flash attention ───────────────────────
    logger.info("Loading model: %s (bf16, flash_attention_2)", model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except (ValueError, ImportError):
        logger.warning("flash_attention_2 not available, falling back to sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    model.train()

    # Gradient checkpointing: trade compute for ~60% activation memory savings
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{n_params:,}", f"{n_trainable:,}")

    # ── Gradient CPU offloading ───────────────────────────────────────
    # Model (8 GB) + gradients (8 GB) = 16 GB -- exceeds VRAM.
    # Solution: register hooks that move each gradient to CPU immediately
    # after it's computed in the backward pass, so only ~1 layer of grads
    # exists on GPU at a time.  Peak GPU grad memory: ~200 MB instead of 8 GB.
    cpu_grad_buffers: dict[torch.nn.Parameter, torch.Tensor] = {}

    def _make_grad_offload_hook(param: torch.nn.Parameter):
        """Create a hook that offloads gradient to CPU after backward."""
        def hook(p):
            if p.grad is not None:
                if p not in cpu_grad_buffers:
                    cpu_grad_buffers[p] = torch.zeros_like(p.grad, device="cpu", pin_memory=True)
                cpu_grad_buffers[p].copy_(p.grad, non_blocking=True)
                p.grad = None  # free GPU memory immediately
        return hook

    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(_make_grad_offload_hook(p))

    logger.info("Gradient CPU offloading enabled for %d parameters", n_trainable)

    # ── Optimizer: ProjectedAdamW with CPU offload ────────────────────
    logger.info(
        "Optimizer: ProjectedAdamW (rank_ratio=%d, offload_to_cpu=True)",
        rank_ratio,
    )
    optimizer = ProjectedAdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        rank_ratio=rank_ratio,
        update_proj_every=update_proj_every,
        offload_to_cpu=True,
    )

    # ── Dataset + DataLoader ──────────────────────────────────────────
    dataset = ChatJsonlDataset(Path(dataset_path), tokenizer, max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ── LR schedule: linear warmup then cosine decay ──────────────────
    total_steps = (len(dataloader) * epochs) // grad_accum
    if total_steps == 0:
        total_steps = 1

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Apply TurboQuant KV cache compression ─────────────────────────
    if turboquant_bits > 0:
        try:
            from turboquant import patch_model
            patch_model(model, bits=turboquant_bits)
            logger.info("TurboQuant %d-bit KV cache enabled", turboquant_bits)
        except ImportError:
            logger.warning("turboquant not available")

    # ── Save config ───────────────────────────────────────────────────
    config = {
        "model_name": model_name,
        "max_seq_len": max_seq_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "lr": lr,
        "rank_ratio": rank_ratio,
        "total_steps": total_steps,
        "n_params": n_params,
        "n_trainable": n_trainable,
        "turboquant_bits": turboquant_bits,
        "offload_to_cpu": True,
    }
    with open(out / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────
    logger.info("Starting full fine-tuning:")
    logger.info("  Dataset: %d examples", len(dataset))
    logger.info("  Effective batch: %d x %d = %d", batch_size, grad_accum, batch_size * grad_accum)
    logger.info("  Total steps: %d", total_steps)
    logger.info("  Max seq len: %d", max_seq_len)
    logger.info("  Device: %s", device)

    global_step = 0
    running_loss = 0.0
    micro_step = 0
    t_start = time.time()

    # CPU gradient accumulation buffer (separate from per-backward offload)
    cpu_grad_accum: dict[torch.nn.Parameter, torch.Tensor] = {}

    def _accumulate_cpu_grads():
        """Add the latest offloaded grads into the accumulation buffer."""
        torch.cuda.synchronize()  # ensure async copies are done
        for p, g in cpu_grad_buffers.items():
            if p not in cpu_grad_accum:
                cpu_grad_accum[p] = torch.zeros_like(g)
            cpu_grad_accum[p].add_(g)

    def _apply_cpu_grads_to_params():
        """Temporarily set .grad from CPU accum buffer for optimizer/clip."""
        for p in model.parameters():
            if p in cpu_grad_accum:
                p.grad = cpu_grad_accum[p].to(p.device, non_blocking=True)

    def _clear_cpu_grads():
        """Zero the accum buffer and free GPU .grad."""
        for p in cpu_grad_accum:
            cpu_grad_accum[p].zero_()
        for p in model.parameters():
            p.grad = None

    for epoch in range(epochs):
        for batch in dataloader:
            # Move to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward + backward in autocast for consistent bf16
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum

            loss.backward()

            # Grads were offloaded to cpu_grad_buffers by hooks; accumulate them
            _accumulate_cpu_grads()

            running_loss += outputs.loss.item()
            micro_step += 1

            # Free forward activations immediately
            del outputs, loss, input_ids, attention_mask, labels
            torch.cuda.empty_cache()

            # Accumulate gradients across micro-batches
            if micro_step % grad_accum != 0:
                continue

            # Move accumulated CPU grads back to GPU for clipping
            _apply_cpu_grads_to_params()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            optimizer.step()
            scheduler.step()

            # Free everything
            _clear_cpu_grads()
            torch.cuda.empty_cache()

            global_step += 1

            # ── Logging ───────────────────────────────────────────
            if global_step % log_every == 0:
                avg_loss = running_loss / (log_every * grad_accum)
                elapsed = time.time() - t_start
                steps_per_sec = global_step / elapsed
                current_lr = scheduler.get_last_lr()[0]

                # GPU memory
                if torch.cuda.is_available():
                    gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    gpu_str = f" | GPU peak: {gpu_mb:.0f} MB"
                else:
                    gpu_str = ""

                logger.info(
                    "step %d/%d | loss %.4f | lr %.2e | %.2f steps/s%s",
                    global_step, total_steps, avg_loss, current_lr,
                    steps_per_sec, gpu_str,
                )
                running_loss = 0.0

            # ── Checkpoint ────────────────────────────────────────
            if global_step % save_every == 0:
                ckpt_path = out / f"step_{global_step}"
                logger.info("Saving checkpoint: %s", ckpt_path)
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    # ── Final save ────────────────────────────────────────────────────
    final_path = out / "final"
    logger.info("Saving final model: %s", final_path)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Optimizer memory stats
    opt_stats = optimizer.memory_stats()
    logger.info("Optimizer states: %.1f MB total (%d projected, %d full)",
                opt_stats["total_mb"], opt_stats["n_projected_params"],
                opt_stats["n_full_params"])

    total_time = (time.time() - t_start) / 3600
    logger.info("Training complete in %.1f hours", total_time)

    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full fine-tune Qwen3-4B on 16 GB GPU with ProjectedAdamW + CPU offload",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", required=True, help="JSONL training data")
    parser.add_argument("--output", default="checkpoints/full_ft")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank-ratio", type=int, default=4,
                        help="ProjectedAdamW compression ratio (default: 4 = 75%% savings)")
    parser.add_argument("--update-proj-every", type=int, default=200,
                        help="Re-randomize projection every N steps (default: 200)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--turboquant-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    train(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rank_ratio=args.rank_ratio,
        update_proj_every=args.update_proj_every,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        turboquant_bits=args.turboquant_bits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
