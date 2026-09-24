#ifndef DS4_STREAM_Q4_H
#define DS4_STREAM_Q4_H

#include <stdint.h>

enum {
    DS4_STREAM_Q4_MAX_SELECTED = 10,
    DS4_LAGUNA_STREAM_PREFILL_CHUNK_DEFAULT = 8,
    DS4_LAGUNA_STREAM_PREFILL_CHUNK_MAX = 32,
    DS4_LAGUNA_STREAM_PREFILL_MARGIN = 10,
};

/* The margin keeps one decode top-10 selection on top of the chunk worst case.
 * N=1 keeps the original admission rule instead, which the byte-identical
 * sequential path needs. */
static inline uint32_t ds4_laguna_stream_prefill_required_cache(
        uint32_t chunk) {
    if (chunk == 0 || chunk > DS4_LAGUNA_STREAM_PREFILL_CHUNK_MAX) return 0;
    if (chunk == 1) return DS4_STREAM_Q4_MAX_SELECTED;
    return chunk * DS4_STREAM_Q4_MAX_SELECTED +
           DS4_LAGUNA_STREAM_PREFILL_MARGIN;
}

static inline int ds4_laguna_stream_prefill_cache_admitted(
        uint32_t chunk, uint32_t cache_capacity) {
    const uint32_t required =
        ds4_laguna_stream_prefill_required_cache(chunk);
    return required != 0 && cache_capacity >= required;
}

/* Stable union of the experts selected by the rows of one chunk. */
static inline int ds4_laguna_stream_prefill_union(
        const int32_t *ids,
        uint32_t n_rows,
        uint32_t n_selected,
        uint32_t n_total,
        int32_t *unique_ids,
        uint32_t unique_cap,
        uint32_t *unique_count) {
    if (unique_count) *unique_count = 0;
    if (!ids || !unique_ids || !unique_count ||
        n_rows == 0 || n_rows > DS4_LAGUNA_STREAM_PREFILL_CHUNK_MAX ||
        n_selected != DS4_STREAM_Q4_MAX_SELECTED ||
        n_total == 0 || n_total > 256 || unique_cap < n_total) {
        return 0;
    }
    uint8_t seen[256] = {0};
    const uint64_t count = (uint64_t)n_rows * n_selected;
    for (uint64_t i = 0; i < count; i++) {
        const int32_t id = ids[i];
        if (id < 0 || (uint32_t)id >= n_total) return 0;
        if (!seen[(uint32_t)id]) {
            seen[(uint32_t)id] = 1;
            unique_ids[(*unique_count)++] = id;
        }
    }
    return 1;
}

/* Host checks for the cache-only Q4 consumer; no GPU or model access. */
static inline int ds4_stream_q4_selection_valid(
        uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_bytes, uint64_t down_bytes,
        uint32_t n_total, uint32_t n_selected, uint32_t budget,
        const int32_t *ids) {
    if (!ids || n_selected != DS4_STREAM_Q4_MAX_SELECTED ||
        n_total < n_selected || budget < n_selected ||
        gate_bytes == 0 || down_bytes == 0 ||
        n_total > UINT64_MAX / gate_bytes ||
        n_total > UINT64_MAX / down_bytes ||
        gate_offset > model_size || up_offset > model_size ||
        down_offset > model_size ||
        n_total * gate_bytes > model_size - gate_offset ||
        n_total * gate_bytes > model_size - up_offset ||
        n_total * down_bytes > model_size - down_offset) {
        return 0;
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= n_total) return 0;
        for (uint32_t j = 0; j < i; j++) {
            if (ids[i] == ids[j]) return 0;
        }
    }
    return 1;
}

#endif
