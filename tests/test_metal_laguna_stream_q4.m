/* Synthetic Metal fixture: needs a Metal device, no model weights. It includes
 * the backend so the cache can be observed without new production APIs. */
#define DS4_TEST_HOOKS
#include "../ds4_metal.m"
#include "laguna_stream_q4_fixture.h"

static int consume(laguna_stream_fixture *f, ds4_gpu_tensor *out,
                   ds4_gpu_tensor *mid, ds4_gpu_tensor *selected,
                   ds4_gpu_tensor *weights, ds4_gpu_tensor *x, uint32_t layer) {
    const uint64_t views = g_test_model_range_calls;
    const int ok = ds4_gpu_laguna_stream_routed_moe_one_tensor(
        out, mid, f->map, f->size, &f->desc, FIX_DIM, FIX_DIM, FIX_DIM,
        selected, weights, FIX_TOTAL, DS4_STREAM_Q4_MAX_SELECTED, layer, x);
    assert(g_test_model_range_calls == views);
    assert(!g_stream_expert_pending_load.active);
    return ok;
}

int main(void) {
    @autoreleasepool {
        laguna_stream_fixture f = fixture_open();
        if (!ds4_gpu_init()) {
            fprintf(stderr, "Metal unavailable: this fixture needs a Metal device\n");
            fixture_close(&f);
            return 1;
        }
        ds4_gpu_set_ssd_streaming(true);
        assert(ds4_gpu_set_model_fd(fileno(f.file)));
        const uint64_t offset = 0, size = FIX_PREFIX;
        assert(ds4_gpu_set_model_map_spans(f.map, f.size, &offset, &size, 1, FIX_PREFIX));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(FIX_DIM * sizeof(float));
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(10 * FIX_DIM * sizeof(float));
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(FIX_DIM * sizeof(float));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(10 * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(10 * sizeof(float));
        ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(FIX_TOTAL * sizeof(float));
        ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(FIX_TOTAL * sizeof(float));
        assert(out && mid && x && selected && weights && logits && probs);
        assert(ds4_gpu_tensor_fill_f32(x, 1.0f / FIX_DIM, FIX_DIM));

        ds4_gpu_set_streaming_expert_cache_budget(9);
        assert(!consume(&f, out, mid, selected, weights, x, 1));
        assert(ds4_gpu_stream_expert_cache_current_count() == 0);
        ds4_gpu_set_streaming_expert_cache_budget(10);

        for (unsigned pass = 0; pass < 4; pass++) {
            const unsigned first = pass == 0 ? 0 : 2;
            const unsigned layer = pass == 3 ? 2 : 1;
            float scores[FIX_TOTAL];
            for (unsigned i = 0; i < FIX_TOTAL; i++) {
                scores[i] = i >= first && i < first + 10 ? 0.1f * i : -10.0f;
            }
            assert(ds4_gpu_tensor_write(logits, 0, scores, sizeof(scores)));
            /* Leave the router queued: the consumer must complete it before reading IDs. */
            assert(ds4_gpu_begin_commands());
            assert(ds4_gpu_glm_router_select_tensor(
                selected, weights, probs, f.map, f.size, 0, logits, FIX_TOTAL, 10, 1.0f));
            const uint64_t hits = g_stream_expert_cache_hits;
            const uint64_t misses = g_stream_expert_cache_misses;
            const uint64_t reads = g_stream_expert_cache_pread_bytes;
            assert(consume(&f, out, mid, selected, weights, x, layer));
            assert(g_batch_cb != nil && !g_batch_has_work);
            const unsigned expected_hits = pass == 1 ? 8 : pass == 2 ? 10 : 0;
            assert(g_stream_expert_cache_hits - hits == expected_hits);
            assert(g_stream_expert_cache_misses - misses == 10 - expected_hits);
            assert(g_stream_expert_cache_pread_bytes - reads ==
                   (10 - expected_hits) * 3 * f.desc.gate_expert_bytes);
            assert(ds4_gpu_stream_expert_cache_current_count() == 10);
            assert(ds4_gpu_end_commands());

            int32_t ids[10];
            float ws[10], actual[FIX_DIM];
            assert(ds4_gpu_tensor_read(selected, 0, ids, sizeof(ids)));
            assert(ds4_gpu_tensor_read(weights, 0, ws, sizeof(ws)));
            assert(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)));
            double expected = 0;
            unsigned seen = 0;
            for (unsigned slot = 0; slot < 10; slot++) {
                assert(ids[slot] >= (int32_t)first && ids[slot] < (int32_t)(first + 10));
                const unsigned expert = (unsigned)ids[slot];
                assert(!(seen & (1u << expert)));
                seen |= 1u << expert;
                const double gate = fixture_quant(0, expert) / 16.0;
                const double up = fixture_quant(1, expert) / 16.0;
                expected += gate / (1.0 + exp(-gate)) * up * ws[slot] * fixture_quant(2, expert);
                ds4_gpu_stream_expert_cache_entry *e = &g_stream_expert_cache[layer][expert];
                assert(e->valid && !ds4_gpu_stream_expert_cache_entry_inflight(e));
                assert(e->gate_abs_offset == f.desc.gate_offset + expert * f.desc.gate_expert_bytes);
                assert(e->up_abs_offset == f.desc.up_offset + expert * f.desc.up_expert_bytes);
                assert(e->down_abs_offset == f.desc.down_offset + expert * f.desc.down_expert_bytes);
                assert(memcmp((char *)[e->gate_buffer contents] + e->gate_inner,
                              f.map + e->gate_abs_offset, f.desc.gate_expert_bytes) == 0);
                assert(memcmp((char *)[e->up_buffer contents] + e->up_inner,
                              f.map + e->up_abs_offset, f.desc.up_expert_bytes) == 0);
                assert(memcmp((char *)[e->down_buffer contents] + e->down_inner,
                              f.map + e->down_abs_offset, f.desc.down_expert_bytes) == 0);
            }
            assert((seen & (3u << 8)) == (3u << 8));
            for (unsigned row = 0; row < FIX_DIM; row++) {
                assert(isfinite(actual[row]) && fabs(actual[row] - expected) < 1e-4 * (1 + fabs(expected)));
            }
        }

        assert(setenv("DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE", "1", 1) == 0);
        assert(!consume(&f, out, mid, selected, weights, x, 1));
        assert(unsetenv("DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE") == 0);
        id<MTLComputePipelineState> saved = g_glm_q4_k_addr_pair_swiglu_f32_pipeline;
        g_glm_q4_k_addr_pair_swiglu_f32_pipeline = nil;
        assert(!consume(&f, out, mid, selected, weights, x, 1));
        g_glm_q4_k_addr_pair_swiglu_f32_pipeline = saved;
        f.desc.down_type = DS4_METAL_TENSOR_Q6_K;
        assert(!consume(&f, out, mid, selected, weights, x, 1));
        f.desc.down_type = DS4_METAL_TENSOR_Q4_K;

        /* Fresh layer plus truncated expert payload must fail, never fall back. */
        assert(ftruncate(fileno(f.file), FIX_PREFIX) == 0);
        assert(!consume(&f, out, mid, selected, weights, x, 3));
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(probs);
        ds4_gpu_cleanup();
        fixture_close(&f);
        puts("laguna-stream-q4-metal: OK (router, top-10, hit/miss, reuse, numeric, no fallback, I/O failure)");
    }
    return 0;
}
