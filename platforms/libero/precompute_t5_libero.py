#!/usr/bin/env python3
"""Precompute T5 embeddings for all LIBERO task instructions.

Encodes each task's language instruction once with the WAN T5 (umt5-xxl)
encoder and saves a single `.pt` file holding a dict::

    {instruction_str: torch.Tensor[L, D] (cpu, float32)}

`eval_libero_T.py` (and its sibling scripts) can then load this file via
`--t5_embeds path/to/libero_t5.pt` and look up embeddings by instruction
string at runtime — no need to ever load the 11GB T5 encoder during eval.

Usage:
    python platforms/libero/precompute_t5_libero.py \
        --wan_path pretrained_models/Wan2.2-TI2V-5B \
        --output platforms/libero/libero_t5_embeds.pt \
        --task_suites libero_spatial libero_object libero_goal libero_10
"""

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch

from libero.libero import benchmark


def collect_instructions(task_suites):
    """Return ordered, de-duplicated list of (suite_name, task_id, instruction)."""
    benchmark_dict = benchmark.get_benchmark_dict()
    seen = set()
    items = []
    for suite_name in task_suites:
        if suite_name not in benchmark_dict:
            print(f"[skip] Unknown task suite: {suite_name}")
            continue
        suite = benchmark_dict[suite_name]()
        n = suite.n_tasks
        print(f"[{suite_name}] {n} tasks")
        for tid in range(n):
            instr = suite.get_task(tid).language
            if instr in seen:
                continue
            seen.add(instr)
            items.append((suite_name, tid, instr))
    return items


def encode_all(items, wan_path, batch_size=4):
    """Encode every instruction with the WAN T5 encoder.

    Returns a dict mapping instruction → CPU float32 tensor of shape [L, D].
    """
    if not os.path.isabs(wan_path):
        wan_path = os.path.abspath(os.path.join(parent_dir, wan_path))
    t5_ckpt = os.path.join(wan_path, "models_t5_umt5-xxl-enc-bf16.pth")
    t5_tokenizer = os.path.join(wan_path, "google/umt5-xxl")
    if not os.path.exists(t5_ckpt):
        raise FileNotFoundError(f"T5 ckpt not found: {t5_ckpt}")
    if not os.path.exists(t5_tokenizer):
        raise FileNotFoundError(f"T5 tokenizer dir not found: {t5_tokenizer}")

    from wan22_core.bak.wan.modules.t5 import T5EncoderModel

    print(f"Loading T5 encoder from {wan_path} (this is the slow part) ...")
    t5 = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device="cuda",
        checkpoint_path=t5_ckpt,
        tokenizer_path=t5_tokenizer,
    )

    embeds = {}
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            texts = [instr for _, _, instr in batch]
            out = t5(texts, "cuda")
            if isinstance(out, list):
                for (_, _, instr), emb in zip(batch, out):
                    embeds[instr] = emb.detach().to("cpu", dtype=torch.float32)
            elif isinstance(out, torch.Tensor):
                for j, (_, _, instr) in enumerate(batch):
                    emb = out[j] if out.ndim == 3 else out
                    embeds[instr] = emb.detach().to("cpu", dtype=torch.float32)
            else:
                raise TypeError(f"Unexpected T5 output type: {type(out)}")
            print(
                f"  encoded {min(i + batch_size, len(items))}/{len(items)}: "
                f"{texts[0][:60]}{'...' if len(texts[0]) > 60 else ''}"
            )
    return embeds


def main():
    parser = argparse.ArgumentParser(
        description="Precompute T5 embeddings for LIBERO instructions."
    )
    parser.add_argument(
        "--wan_path", type=str, default="pretrained_models/Wan2.2-TI2V-5B",
        help="Path to WAN pretrained models (containing the T5 ckpt + tokenizer).",
    )
    parser.add_argument(
        "--output", type=str, default="platforms/libero/libero_t5_embeds.pt",
        help="Output .pt file (relative to wam project root if not absolute).",
    )
    parser.add_argument(
        "--task_suites", type=str, nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="Task suites whose instructions should be encoded.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(os.path.join(parent_dir, output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    items = collect_instructions(args.task_suites)
    print(f"\nTotal unique instructions to encode: {len(items)}")
    for suite, tid, instr in items:
        print(f"  [{suite}/{tid:02d}] {instr}")

    embeds = encode_all(items, args.wan_path, batch_size=args.batch_size)

    payload = {
        "embeddings": embeds,
        "task_index": [
            {"suite": suite, "task_id": tid, "instruction": instr}
            for (suite, tid, instr) in items
        ],
    }
    torch.save(payload, output_path)
    print(f"\nSaved {len(embeds)} embeddings to {output_path}")
    sample_key = next(iter(embeds))
    print(
        f"Example shape: embeddings['{sample_key[:50]}...'] = "
        f"{tuple(embeds[sample_key].shape)}"
    )


if __name__ == "__main__":
    main()
