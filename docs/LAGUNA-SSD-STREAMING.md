# Laguna SSD streaming

`--ssd-streaming` runs Laguna S 2.1 Q4_K_M on Metal without making the routed
experts resident. The non-routed weights (attention, router, shared experts,
embeddings and output, about 4.07 GiB) are mapped as selective views of the
GGUF; the routed experts (about 59.5 GiB) are read from the file on demand into
a bounded Metal expert cache. This lets the 63.56 GiB Q4_K_M model run on a
64 GB Mac without lowering the quantization.

## Supported configuration

- Model: the official Laguna S 2.1 Q4_K_M GGUF (`./download_model.sh
  laguna-q4`). Routed experts must be uniform Q4_K and the signal/shared
  projections Q8_0, as in that file.
- Backend: Metal only.
- Routing: a dedicated top-10 Q4 consumer. `DS4_STREAM_Q4_MAX_SELECTED` is 10
  for this path; the generic and GLM routed-MoE limit stays
  `DS4_METAL_MAX_ROUTED_EXPERT_USED == 8`, so GLM/DeepSeek streaming is not
  affected.
- Expert cache: required, with no resident-weight fallback. The default cap is
  8 GiB; with 5,308,416 bytes per expert this is 1,618 entries, allocated as
  two slabs of the default 4 GiB target. `--ssd-streaming-cache-experts` sets
  an explicit count or budget. At startup the cache is reduced, or startup is
  rejected, if the non-routed spans, KV, graph scratch and a 1 GiB reserve do
  not fit within 80% of Metal's recommended working set. At least 10 entries
  are required.
- Eviction: route hotness first, least-recent use as the tie breaker. The decay
  clock advances once per streaming decode row and halves all hotness counters
  every 16 rows; prefill does not advance it. Without the decay the 1,618-entry
  cache turns over too fast for accumulated hotness to stay informative.
- Prefill: blocks of 8 rows by default. For each layer the batch consumer loads
  the union of the experts selected by the block once and keeps those entries
  pinned for that layer's command buffer. While the missing experts are read,
  the GPU already runs the shared expert and the gate/up pass of the experts
  that are cached; a masked second pass covers the experts just loaded and a
  single unchanged down projection follows, so the result is bit-identical to
  the serial path. `DS4_LAGUNA_STREAM_PREFILL_CHUNK=N` (1..32) overrides the
  block size; `N=1` selects the strictly sequential path. If the cache cannot
  admit `N` (`N*10 + 10` entries for `N>1`) or the graph memory does not fit,
  startup picks the largest admissible `N` down to 1 and reports both values.
- Prefill numerics: signal/shared Q8 projections keep the single-row reduction
  order inside a block, so small projection drift cannot compound through later
  router choices. Batched attention, the router F32 projection and the routed
  Q4 kernels still differ from row decode, so `N>1` is not bit-identical to
  `N=1`; a single row, including a one-token prefill remainder, always takes
  the decode path.
- Residency: routed expert tensors are excluded from startup mapping.
  Full-model residency and resident full layers are disabled. The non-routed
  views are not wired explicitly, so first use can take page faults.

Rejected under Laguna streaming, at startup:

- DFlash (`--dflash`, `--dflash-draft`, `--dflash-p-min`) and MTP (`--mtp`)
  support models, and any speculative, row-argmax or feature-capture request.
- `--warm-weights`, `--ssd-streaming-cold`, `--ssd-streaming-full-layers`,
  `--ssd-streaming-preload-experts`, and the standalone Metal graph diagnostic.
- `DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE`,
  `DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE`,
  `DS4_METAL_ENABLE_STREAMING_COMPACT_ADDR`,
  `DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER`.

`--head-test` is rejected for Laguna in every mode, because the model has none
of the HC tensors that diagnostic needs.

## Usage

```sh
./download_model.sh laguna-q4
./ds4 -m gguf/laguna-s-2.1-Q4_K_M.gguf --ssd-streaming -c 4096 \
  -p "Explain this repository"
```

`ds4-server` and `ds4-agent` parse the same flag; the measurements below were
taken with the `ds4` CLI. Startup prints the chosen cache size, the non-routed
and excluded routed bytes, the KV and scratch estimates, and the selected
prefill block.

## Measured results

Mac Studio M2 Ultra, 64 GB unified memory, Metal, Laguna S 2.1 Q4_K_M, default
8 GiB expert cache (8.8 GB peak footprint).

- Decode speed depends mostly on how much free RAM macOS can use as page cache
  for the GGUF. With plenty of free RAM, decode runs at about 9 t/s
  (9.0--9.2 t/s over three runs of a 90-token prompt and 256 generated
  tokens); in that state about 80% of the requested expert bytes are served by
  the page cache, and the SSD reads average 1.65 GB/s, below its bandwidth.
  With little free RAM, or a cold page cache, decode settles at about
  6.3--6.6 t/s, the disk-bound floor. Swap does not grow in either case.
- A 4 GiB cache costs about 9% of decode speed (about 8.3 t/s) and halves the
  peak footprint to 4.5 GB.
- Block prefill on a 656-token prompt: 9.0--9.3 t/s sequential, 14.3--15.2 t/s
  with 8-row blocks, at the same peak footprint. The gain is +42% at 4k,
  +29% at 8k and +13% at 16k tokens, where quadratic global attention
  dominates.
- Correctness was checked token by token against llama.cpp (CPU, same GGUF,
  same token IDs, greedy, top-5 logprobs) on three BOS-prefixed prompts of
  about 600 tokens (English, Italian, C code) and on 4k, 8k and 16k contexts,
  both sequential and with 8-row blocks. Every greedy token matched except at
  near-ties, where the two runtimes swap nearly equal top candidates (at 16k
  with 8-row blocks, the first divergence came after 28 of 32 tokens at a
  top-1/top-2 gap of 0.03). Sequential and 8-row prefill produced the same
  32 greedy tokens at 4k, 8k and 16k.

Compare with the same explicit BOS token (`〈|EOS|〉`, id 2) on both runtimes,
and disable llama.cpp's automatic BOS insertion when passing token IDs.
Without BOS the model is out of distribution and the two runtimes can
disagree strongly; that divergence is not caused by streaming.

## Environment variables

Diagnostics and controlled experiments, not part of the supported
configuration:

| Variable | Behavior |
| --- | --- |
| `DS4_LAGUNA_STREAM_PREFILL_CHUNK=N` | Block prefill size, 1..32, default 8. `1` selects the sequential path. Invalid values fail at startup. |
| `DS4_LAGUNA_STREAM_BATCH_OVERLAP=0` | Disables the in-layer overlap of the batch consumer and restores the serial wait, read, MoE path. Outputs, cache contents, victims and counters are identical either way. |
| `DS4_LAGUNA_RECORD_SELECTED_IDS=FILE` | Records the routed top-10 selections of the single-row path (layer, position, expert IDs) as little-endian int32 records; run with `DS4_LAGUNA_STREAM_PREFILL_CHUNK=1` to capture the prompt too. `tools/laguna_expert_blocks.py FILE` reports how many distinct experts blocks of 1, 2, 4 and 8 rows touch; it was used to size the block prefill. |
| `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1` | Prints a cleanup summary: cache budget, hits, misses, evictions, buffer reuse, read bytes and time, Laguna row/MoE/wait counters, per-block union sizes, and a `batch overlap=` line with the all-resident/all-missing/mixed split and the router wait, load, resident submit and GPU tail times. |
| `DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS=1` | Disables the slab allocator. Diagnostic only; it is materially slower. |
| `DS4_METAL_STREAMING_EXPERT_SLAB_MB=N` | Slab target in MiB, default 4096. |
| `DS4_METAL_STREAMING_EXPERT_PREAD_THREADS=N` | Parallel `pread` workers, default 9, clamped to 1..18. |
| `DS4_METAL_STREAMING_EXPERT_PREAD_POOL=0` | Disables the persistent `pread` worker pool. |
| `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1` | Disables the read-ahead advisory. Diagnostic only; without it the measured `pread` time nearly doubled. |

## Known limitations

- The batch consumer still waits for the router at every layer, because the
  union is only known after selection. There is no speculative decoding and no
  cross-layer expert prefetch; the block union must fit the cache.
- Context admission depends on memory: KV covers the whole requested context,
  so large contexts reduce the expert cache or are rejected at startup.
- The synthetic Metal fixture scales expert entries down by 48; the real-size
  cache and two-slab behavior are only exercised with the model.
- GLM and DeepSeek streaming are not exercised by these tests, since their
  models are 81--430 GB. Their non-regression rests on the shape of the
  change: the Laguna consumer, its cache accounting, batched victim selection
  and overlap are only taken when the model is Laguna, and the shared decode
  callers keep their previous arguments.

## Tests

```sh
make ds4 ds4_test tests/test_laguna_stream_layout tests/test_laguna_stream_q4 \
  tests/test_metal_laguna_stream_q4 tests/test_laguna_stream_engine

./tests/test_laguna_stream_layout   # host: layout, tables, spans, budgets, rejections
./tests/test_laguna_stream_q4       # host: Q4 fixture bounds and offsets
./tests/test_metal_laguna_stream_q4 # Metal, no weights: cache, slabs, I/O errors, batch consumer
./ds4_test --metal-kernels

MODEL=gguf/laguna-s-2.1-Q4_K_M.gguf
./tests/test_laguna_stream_engine "$MODEL"                 # Metal + model
./tests/test_laguna_stream_engine "$MODEL" --long-context
```

`tests/test_laguna_stream_layout --inspect-model "$MODEL"` applies the layout
checks to the real GGUF read-only. The engine test covers reopen, session save
and restore, a decode that crosses the 512-row sliding window, injected read
errors, and block prefill: `N=1` must match a second sequential reference
exactly; the default and explicit `N=8` must be bit-identical; and block sizes
2, 4, 8 and 16 must reproduce the four greedy tokens of `N=1`, except at a
reference near-tie (top-1/top-2 gap at most 0.01, with the chosen token among
the reference top two). `--long-context` runs at context 1024 with a
517-token prefill that crosses the sliding window.
