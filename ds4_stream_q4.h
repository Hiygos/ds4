#ifndef DS4_STREAM_Q4_H
#define DS4_STREAM_Q4_H

#include <stdint.h>

enum { DS4_STREAM_Q4_MAX_SELECTED = 10 };

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
