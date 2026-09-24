#!/usr/bin/env python3
"""Summarize expert reuse in Laguna SSD-streaming selection records."""

from __future__ import annotations

import argparse
import struct
from collections import defaultdict
from pathlib import Path


EXPERT_BYTES = 5_308_416
N_SELECTED = 10
RECORD = struct.Struct("<13i")


def read_records(path: Path) -> dict[int, dict[int, set[int]]]:
    data = path.read_bytes()
    if len(data) % RECORD.size:
        raise SystemExit(
            f"truncated file: {len(data)} bytes, records are {RECORD.size} bytes"
        )

    rows: dict[int, dict[int, set[int]]] = defaultdict(dict)
    for offset in range(0, len(data), RECORD.size):
        layer, row, n_selected, *experts = RECORD.unpack_from(data, offset)
        if n_selected != N_SELECTED:
            raise SystemExit(
                f"record {offset // RECORD.size}: n_selected={n_selected}, expected 10"
            )
        if layer in rows[row]:
            raise SystemExit(f"duplicate record for row {row}, layer {layer}")
        rows[row][layer] = set(experts)
    return rows


def consecutive_runs(values: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for value in values:
        if not runs or value != runs[-1][-1] + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return runs


def block_distinct_counts(
    rows: dict[int, dict[int, set[int]]], block_size: int
) -> list[int]:
    by_layer: dict[int, dict[int, set[int]]] = defaultdict(dict)
    for row, layers in rows.items():
        for layer, experts in layers.items():
            by_layer[layer][row] = experts

    counts: list[int] = []
    for layer_rows in by_layer.values():
        for run in consecutive_runs(sorted(layer_rows)):
            for start in range(0, len(run) - block_size + 1, block_size):
                experts: set[int] = set()
                for row in run[start : start + block_size]:
                    experts.update(layer_rows[row])
                counts.append(len(experts))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the records written by DS4_LAGUNA_RECORD_SELECTED_IDS."
    )
    parser.add_argument("record_file", type=Path)
    args = parser.parse_args()

    rows = read_records(args.record_file)
    if not rows:
        raise SystemExit("no records in the file")

    print(
        "scope=all_rows "
        "(the format does not tell prefill and decode apart; non-overlapping blocks)"
    )
    for block_size in (1, 2, 4, 8):
        counts = block_distinct_counts(rows, block_size)
        if not counts:
            print(f"block_size={block_size} samples=0")
            continue
        mean_distinct = sum(counts) / len(counts)
        ratio = mean_distinct / (block_size * N_SELECTED)
        bytes_per_token = mean_distinct * EXPERT_BYTES / block_size
        print(
            f"block_size={block_size} samples={len(counts)} "
            f"mean_distinct={mean_distinct:.6f} ratio={ratio:.6f} "
            f"bytes_per_token={bytes_per_token:.3f}"
        )


if __name__ == "__main__":
    main()
