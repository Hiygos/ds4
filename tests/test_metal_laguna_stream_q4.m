/* Synthetic Metal fixture: needs a Metal device, no model weights. It includes
 * the backend so the cache can be observed without new production APIs. */
#define DS4_TEST_HOOKS
#include "../ds4_metal.m"
#include "laguna_stream_q4_fixture.h"

static unsigned read_calls, fail_read;
static int read_error, truncate_read;

static ssize_t fault_pread(int fd, void *dst, size_t len, off_t off) {
    assert(g_test_laguna_stream_temporary_bytes > 0);
    const unsigned call = __atomic_add_fetch(&read_calls, 1, __ATOMIC_RELAXED);
    if (call == fail_read) {
        if (truncate_read) {
            assert(ftruncate(fd, off + (off_t)(len / 2)) == 0);
        } else {
            errno = read_error;
            return -1;
        }
    }
    /* A short but positive read must be completed too. */
    if (call == 1) len /= 2;
    return pread(fd, dst, len, off);
}

static void assert_cache_drained(void) {
    assert(g_stream_expert_cache_entry_count == 0);
    assert(!g_stream_expert_pending_load.active);
    assert(!g_stream_expert_pending_load.model_map);
    assert(!g_stream_expert_pending_load.n_tasks);
    assert(!g_stream_expert_pread_pool_tasks);
    assert(!g_stream_expert_pread_pool_remaining_workers);
    for (unsigned layer = 0; layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; layer++) {
        for (unsigned expert = 0; expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; expert++) {
            ds4_gpu_stream_expert_cache_entry *e = &g_stream_expert_cache[layer][expert];
            assert(!e->valid && !e->model_map && !e->gate_buffer && !e->up_buffer && !e->down_buffer);
        }
    }
    assert(g_test_laguna_stream_temporary_bytes == 0);
}

static int consume(laguna_stream_fixture *f, ds4_gpu_tensor *out,
                   ds4_gpu_tensor *mid, ds4_gpu_tensor *selected,
                   ds4_gpu_tensor *weights, ds4_gpu_tensor *x, uint32_t layer) {
    const uint64_t views = g_test_model_range_calls;
    const int ok = ds4_gpu_laguna_stream_routed_moe_one_tensor(
        out, mid, f->map, f->size, &f->desc, f->in_dim, f->mid_dim, f->out_dim,
        selected, weights, FIX_TOTAL, DS4_STREAM_Q4_MAX_SELECTED, layer, x);
    assert(g_test_model_range_calls == views);
    assert(!g_stream_expert_pending_load.active);
    assert(g_test_laguna_stream_temporary_bytes == 0);
    const uint64_t bytes = 2 * f->desc.gate_expert_bytes + f->desc.down_expert_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t limit = 10 * ((bytes + page - 1) / page * page);
    assert(g_test_laguna_stream_peak_bytes <= limit);
    assert(g_test_laguna_stream_peak_allocated_bytes <= limit);
    assert(g_laguna_stream_allocated_bytes <= limit);
    return ok;
}

static int run_case(bool rectangular, laguna_stream_fixture *previous) {
    @autoreleasepool {
        laguna_stream_fixture f = fixture_open_case(rectangular);
        if (previous->file) {
            assert(f.map != previous->map && fileno(f.file) != fileno(previous->file));
            assert_cache_drained();
            fixture_close(previous);
        }
        if (!ds4_gpu_init()) {
            fprintf(stderr, "Metal unavailable: this fixture needs a Metal device\n");
            fixture_close(&f);
            return 1;
        }
        ds4_gpu_set_ssd_streaming(true);
        assert(ds4_gpu_set_model_fd(fileno(f.file)));
        const uint64_t offset = 0, size = FIX_PREFIX;
    assert(ds4_gpu_set_model_map_spans(f.map, f.size, &offset, &size, 1, FIX_PREFIX));
    /* Positive counter check: zero fallbacks is not enough if the counter is
     * off. */
    uint64_t inner = UINT64_MAX, views = g_test_model_range_calls;
    assert(ds4_gpu_wrap_model_range(f.map, f.size, 0, 16, &inner));
    assert(g_test_model_range_calls == views + 1 && inner == 0);
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(f.out_dim * sizeof(float));
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(10 * f.mid_dim * sizeof(float));
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(f.in_dim * sizeof(float));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(10 * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(10 * sizeof(float));
        ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(FIX_TOTAL * sizeof(float));
        ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(FIX_TOTAL * sizeof(float));
        assert(out && mid && x && selected && weights && logits && probs);
        float input[512];
        for (unsigned c = 0; c < f.in_dim; c++)
            input[c] = rectangular ? (1.0f + c % 17) / (9 * f.in_dim) : 1.0f / f.in_dim;
    assert(ds4_gpu_tensor_write(x, 0, input, f.in_dim * sizeof(float)));
    assert(f.in_dim > 0 && f.mid_dim > 0 && f.out_dim > 0);

        const uint64_t bytes = 2 * f.desc.gate_expert_bytes + f.desc.down_expert_bytes;
        uint64_t slot_bytes = 0, limit = 0;
        assert(ds4_gpu_laguna_stream_allocation_limit(bytes, 10, &slot_bytes, &limit));
        const int slabs = ds4_gpu_stream_expert_slab_enabled();
        uint64_t slots_per_slab = ds4_gpu_stream_expert_slab_target_bytes() / slot_bytes;
        if (!slots_per_slab) slots_per_slab = 1;
        const uint64_t expected_allocs = slabs ? (10 + slots_per_slab - 1) / slots_per_slab : 10;
        g_test_laguna_stream_peak_bytes = 0;
        g_test_laguna_stream_peak_allocated_bytes = 0;
        ds4_gpu_set_streaming_expert_cache_budget(9);
        assert(!consume(&f, out, mid, selected, weights, x, 1));
        assert(ds4_gpu_stream_expert_cache_current_count() == 0);
        ds4_gpu_set_streaming_expert_cache_budget(10);

        for (unsigned pass = 0; pass < 16; pass++) {
            const unsigned first = pass == 0 ? 0 : pass < 4 ? 2 : pass % 3;
            const unsigned layer = pass < 3 ? 1 : 2 + pass % 4;
            float scores[FIX_TOTAL];
            for (unsigned i = 0; i < FIX_TOTAL; i++) {
                scores[i] = i >= first && i < first + 10 ? 0.1f * i : -10.0f;
            }
            assert(ds4_gpu_tensor_write(logits, 0, scores, sizeof(scores)));
            /* The consumer must complete the router before reading the IDs. */
            assert(ds4_gpu_begin_commands());
            assert(ds4_gpu_glm_router_select_tensor(
                selected, weights, probs, f.map, f.size, 0, logits, FIX_TOTAL, 10, 1.0f));
            const uint64_t hits = g_stream_expert_cache_hits;
            const uint64_t misses = g_stream_expert_cache_misses;
            const uint64_t reads = g_stream_expert_cache_pread_bytes;
            const uint64_t pool_tasks = g_test_laguna_stream_pool_tasks;
            const uint64_t scans = g_stream_expert_timing_reuse_scan_calls;
            const uint64_t batches = g_stream_expert_timing_prepare_batch_reuse_calls;
            const uint64_t prepared = g_stream_expert_timing_prepare_task_experts;
            const uint64_t buffers = g_stream_expert_timing_prepare_buffer_calls;
            assert(consume(&f, out, mid, selected, weights, x, layer));
            assert(g_batch_cb != nil && !g_batch_has_work);
            const unsigned expected_hits = pass == 1 ? 8 : pass == 2 ? 10 : 0;
            assert(g_stream_expert_cache_hits - hits == expected_hits);
            assert(g_stream_expert_cache_misses - misses == 10 - expected_hits);
            /* Ten misses must hand thirty reads to the pool before the
             * dispatch. */
            assert(g_test_laguna_stream_pool_tasks - pool_tasks == 3 * (10 - expected_hits));
            assert(g_stream_expert_pread_pool_thread_count > 1);
            /* With a full cache all misses share a single victim scan. */
            const unsigned expected_scans = pass != 0 && expected_hits != 10;
            assert(g_stream_expert_timing_reuse_scan_calls - scans == expected_scans);
            assert(g_stream_expert_timing_prepare_batch_reuse_calls - batches == expected_scans);
            assert(g_stream_expert_timing_prepare_task_experts - prepared == 10 - expected_hits);
            assert(g_stream_expert_timing_prepare_buffer_calls - buffers == (pass == 0 ? 10 : 0));
            assert(g_stream_expert_cache_pread_bytes - reads ==
                   (10 - expected_hits) * (2 * f.desc.gate_expert_bytes + f.desc.down_expert_bytes));
        assert(ds4_gpu_stream_expert_cache_current_count() == 10);
        unsigned occupied = 0;
        for (unsigned il = 0; il < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; il++)
            for (unsigned e = 0; e < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; e++)
                occupied += g_stream_expert_cache[il][e].valid != 0;
        assert(occupied == 10);
            assert(g_stream_expert_cache_buffer_allocs == expected_allocs);
            assert(g_stream_expert_cache_slab_count == (slabs ? expected_allocs : 0));
            assert(ds4_gpu_end_commands());

            int32_t ids[10];
            float ws[10], actual[512];
            assert(ds4_gpu_tensor_read(selected, 0, ids, sizeof(ids)));
            assert(ds4_gpu_tensor_read(weights, 0, ws, sizeof(ws)));
            assert(ds4_gpu_tensor_read(out, 0, actual, f.out_dim * sizeof(float)));
            double expected[512];
            double expected_mid[10 * FIX_DIM];
            float actual_mid[10 * FIX_DIM];
            fixture_reference(&f, ids, ws, input, expected, expected_mid);
            assert(ds4_gpu_tensor_read(mid, 0, actual_mid, sizeof(actual_mid)));
            for (unsigned i = 0; i < 10 * f.mid_dim; i++) {
                assert(isfinite(actual_mid[i]) &&
                       fabs(actual_mid[i] - expected_mid[i]) < 1e-7 + 1e-4 * fabs(expected_mid[i]));
            }
            unsigned seen = 0;
            for (unsigned slot = 0; slot < 10; slot++) {
                assert(ids[slot] >= (int32_t)first && ids[slot] < (int32_t)(first + 10));
                const unsigned expert = (unsigned)ids[slot];
                assert(!(seen & (1u << expert)));
                seen |= 1u << expert;
                ds4_gpu_stream_expert_cache_entry *e = &g_stream_expert_cache[layer][expert];
                assert(e->valid && !ds4_gpu_stream_expert_cache_entry_inflight(e));
                assert(e->model_map == f.map && e->model_size == f.size);
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
            for (unsigned row = 0; row < f.out_dim; row++) {
                assert(isfinite(actual[row]) &&
                       fabs(actual[row] - expected[row]) < 1e-4 * (1 + fabs(expected[row])));
            }
        }
        if (slabs) assert(g_test_laguna_stream_peak_bytes == limit);
        else assert(g_test_laguna_stream_peak_bytes >= 10 * bytes);

        /* A GPU copy keeps the bytes of the old selection while the next one
         * forces evictions. Try both the queued and the submitted case. */
        id<MTLBuffer> probe = [g_device newBufferWithLength:160 options:MTLResourceStorageModeShared];
        unsigned char expected_probe[160];
        int32_t queued_ids[10];
        assert(probe && ds4_gpu_tensor_read(selected, 0, queued_ids, sizeof(queued_ids)));
        assert(ds4_gpu_begin_commands());
        id<MTLBlitCommandEncoder> blit = [g_batch_cb blitCommandEncoder];
        assert(blit);
        for (unsigned i = 0; i < 10; i++) {
            ds4_gpu_stream_expert_cache_entry *e = &g_stream_expert_cache[5][queued_ids[i]];
            assert(e->valid && ds4_gpu_stream_expert_cache_mark_inflight(e));
            memcpy(expected_probe + 16 * i, (char *)[e->gate_buffer contents] + e->gate_inner, 16);
            [blit copyFromBuffer:e->gate_buffer sourceOffset:e->gate_inner
                       toBuffer:probe destinationOffset:16 * i size:16];
            ds4_gpu_stream_expert_cache_clear_entry(5, queued_ids[i], 1);
            assert(e->valid && ds4_gpu_stream_expert_cache_entry_inflight(e));
        }
        [blit endEncoding];
        g_batch_has_work = 1;
        assert(ds4_gpu_flush_commands());
        for (unsigned i = 0; i < 10; i++) {
            ds4_gpu_stream_expert_cache_clear_entry(5, queued_ids[i], 1);
            assert(g_stream_expert_cache[5][queued_ids[i]].valid);
        }
        assert(consume(&f, out, mid, selected, weights, x, 6));
        assert(memcmp([probe contents], expected_probe, sizeof(expected_probe)) == 0);
        assert(g_stream_expert_cache_buffer_allocs == expected_allocs);
        assert(ds4_gpu_end_commands());
        blit = nil;
        probe = nil;

        /* The whole slab stays accounted even with an empty cache; no eleventh
         * slot. */
        ds4_gpu_stream_expert_cache_clear_all(0);
        const uint64_t budget_errors = g_test_laguna_stream_budget_errors;
        const uint64_t allocated_bytes = g_laguna_stream_allocated_bytes;
        assert(!ds4_gpu_laguna_stream_alloc_buffer(slot_bytes, @"over_budget"));
        assert(g_test_laguna_stream_budget_errors == budget_errors + 1);
        assert(g_stream_expert_cache_buffer_allocs == expected_allocs);
        assert_cache_drained();
        assert(consume(&f, out, mid, selected, weights, x, 6));
        assert(g_laguna_stream_allocated_bytes == allocated_bytes);
        assert(g_stream_expert_cache_buffer_allocs == expected_allocs);

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

        /* Error in the parallel batch: no partially valid entry; a new
         * selection on the same fd works. */
        g_test_laguna_stream_pread = fault_pread;
        for (unsigned failure = 0; failure < (rectangular ? 3u : 2u); failure++) {
            read_calls = 0;
            fail_read = 6;
            read_error = failure == 0 ? EIO : EINTR;
            truncate_read = failure == 2;
            const uint64_t pool_tasks = g_test_laguna_stream_pool_tasks;
            assert(!consume(&f, out, mid, selected, weights, x, 7 + failure));
            assert(g_test_laguna_stream_pool_tasks - pool_tasks == 30);
            assert(read_calls >= fail_read);
            assert_cache_drained();
            assert(g_laguna_stream_allocated_bytes == allocated_bytes);
            assert(g_stream_expert_cache_slab_count == (slabs ? expected_allocs : 0));
            if (slabs) assert(g_stream_expert_cache_free_slot_count == 10);
            if (!truncate_read && (rectangular || failure == 0)) {
                read_calls = fail_read = 0;
                assert(consume(&f, out, mid, selected, weights, x, 7 + failure));
                assert(read_calls == 31);
            }
            assert(g_stream_expert_cache_buffer_allocs == expected_allocs);
        }
        g_test_laguna_stream_pread = NULL;
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(probs);
        ds4_gpu_cleanup();
        assert_cache_drained();
        assert(g_model_fd == -1 && !g_model_map_ptr && g_model_view_count == 0);
        assert(!g_stream_expert_cache_expert_bytes);
        assert(!g_laguna_stream_buffers && !g_laguna_stream_free_buffers);
        assert(!g_laguna_stream_allocated_bytes);
        assert(!g_stream_expert_cache_slab_count && !g_stream_expert_cache_slab_total_slots);
        assert(!g_stream_expert_cache_free_slot_count && !g_stream_expert_cache_slab_slot_bytes);
        /* Keep the old file open until the new mapping is created. */
        *previous = f;
        printf("laguna-stream-q4-metal: OK (%s, slabs=%d, allocations=%llu, top-10, eviction, queued/inflight, budget, numeric, I/O, EINTR, cleanup)\n",
               rectangular ? "rectangular/distinct rows" : "square", slabs,
               (unsigned long long)expected_allocs);
    }
    return 0;
}

/* Host limits: padding and unused slots do not vanish from the budget. */
static void check_allocation_limits(void) {
    const uint64_t page = (uint64_t)getpagesize();
    uint64_t slot = 0, limit = 0;
    assert(ds4_gpu_laguna_stream_allocation_limit(3 * page + 2, 10, &slot, &limit));
    assert(slot == 4 * page && limit == 40 * page);
    assert(ds4_gpu_laguna_stream_allocation_limit(5308416, 1618, &slot, &limit));
    assert(slot == 5308416 && limit == 8589017088ull);
    assert(!ds4_gpu_laguna_stream_allocation_limit(0, 10, &slot, &limit));
    assert(!ds4_gpu_laguna_stream_allocation_limit(page, 0, &slot, &limit));
    assert(!ds4_gpu_laguna_stream_allocation_limit(UINT64_MAX, 10, &slot, &limit));
    assert(!ds4_gpu_laguna_stream_allocation_limit(UINT64_MAX / page * page, 2, &slot, &limit));
    assert(!ds4_gpu_laguna_stream_allocation_limit(1, UINT32_MAX, &slot, &limit));
    puts("laguna-stream-budget: OK (padding, full capacity, overflow)");
}

/* A requested slot must account for the whole slab, even before the pread. */
static int check_slab_capacity(void) {
    @autoreleasepool {
        if (!ds4_gpu_init()) return 0;
        ds4_gpu_set_ssd_streaming(true);
        ds4_gpu_set_streaming_expert_cache_budget(10);
        const uint64_t page = (uint64_t)getpagesize();
        const uint64_t gate_bytes = page + 1, down_bytes = page;
        assert(ds4_gpu_stream_expert_cache_note_expert_size(gate_bytes, down_bytes));
        g_laguna_stream_cache_used = 1;
        id<MTLBuffer> gate = nil, up = nil, down = nil;
        NSUInteger gate_inner = 0, up_inner = 0, down_inner = 0;
        assert(ds4_gpu_stream_expert_alloc_slab_slot(gate_bytes, down_bytes,
            &gate, &up, &down, &gate_inner, &up_inner, &down_inner, 1));
        assert(g_stream_expert_cache_slab_count == 1 && g_stream_expert_cache_slab_total_slots == 10);
        assert(g_stream_expert_cache_slab_slots_used[0] == 1);
        assert(!g_stream_expert_cache_bytes && !g_stream_expert_cache_entry_count);
        assert(g_laguna_stream_allocated_bytes == 40 * page);
        const uint64_t errors = g_test_laguna_stream_budget_errors;
        assert(!ds4_gpu_laguna_stream_alloc_buffer(page, @"over_budget"));
        assert(g_test_laguna_stream_budget_errors == errors + 1);
        assert(g_stream_expert_cache_buffer_allocs == 1);
        ds4_gpu_stream_expert_cache_clear_all(0);
        assert(g_stream_expert_cache_free_slot_count == 10);
        assert(g_laguna_stream_allocated_bytes == 40 * page);
        assert(ds4_gpu_stream_expert_alloc_slab_slot(gate_bytes, down_bytes,
            &gate, &up, &down, &gate_inner, &up_inner, &down_inner, 1));
        assert(g_stream_expert_cache_buffer_allocs == 1);
        assert(gate_inner == 9 * 4 * page && up_inner == gate_inner + gate_bytes);
        gate = up = down = nil;
        ds4_gpu_cleanup();
        assert(!g_stream_expert_cache_slab_count && !g_laguna_stream_allocated_bytes);
        puts("laguna-stream-slab: OK (unused slots, padding, allocation limit, reuse)");
    }
    return 1;
}

static void check_legacy_selection_limit(void) {
    /* The two extra physical slots do not enable the legacy
     * early-load/prefetch. */
    static const char sentinel;
    ds4_gpu_stream_expert_table table = {.model_map = &sentinel,
        .model_size = 1048576, .layer = 1, .n_total_expert = 256,
        .gate_offset = 0, .up_offset = 262144, .down_offset = 524288,
        .gate_expert_bytes = 144, .down_expert_bytes = 144};
    int32_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    assert(unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD") == 0);
    assert(unsetenv("DS4_METAL_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD") == 0);
    g_ssd_streaming_mode = 1;
    for (unsigned n = 9; n <= 10; n++) {
        assert(ds4_gpu_stream_expert_cache_begin_selected_load(&table, ids, n));
        assert(!g_stream_expert_pending_load.active);
        assert(ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
            &table, (const ds4_gpu_tensor *)(uintptr_t)1, n));
        ds4_gpu_glm_stream_selected_prefetch_set(&table, ids, n);
        assert(!g_glm_stream_selected_prefetch.active);
        assert(!g_stream_expert_cache_entry_count && !g_stream_expert_cache_buffer_allocs);
    }
    g_ssd_streaming_mode = 0;
    puts("laguna-stream-legacy-limit: OK (9/10 remain unsupported)");
}

/* Batch consumer (block prefill) at a full cache: four overlapping rows give a
 * union of 25 misses. Victims must be taken in blocks of ten (three scans, no
 * per-miss preparation) and must be exactly the 25 entries with the lowest
 * hotness and last_used, i.e. the ones the miss-by-miss choice would have
 * evicted. */
static int check_batch_victims(laguna_stream_fixture *f, uint64_t bytes) {
    enum { ROWS = 4, STEP = 5, UNION = (ROWS - 1) * STEP + 10, LAYER = 15 };
    static bool victim[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER]
                      [DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    memset(victim, 0, sizeof(victim));
    for (unsigned k = 0; k < UNION; k++) {
        unsigned best_layer = UINT32_MAX, best_expert = UINT32_MAX;
        for (unsigned il = 0; il < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; il++)
            for (unsigned e = 0; e < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; e++) {
                if (!g_stream_expert_cache[il][e].valid || victim[il][e]) continue;
                if (best_layer == UINT32_MAX) { best_layer = il; best_expert = e; continue; }
                const uint32_t h = g_stream_expert_cache_route_hotness[il][e];
                const uint32_t bh = g_stream_expert_cache_route_hotness[best_layer][best_expert];
                if (h < bh || (h == bh && g_stream_expert_cache[il][e].last_used <
                                          g_stream_expert_cache[best_layer][best_expert].last_used)) {
                    best_layer = il; best_expert = e;
                }
            }
        assert(best_layer != UINT32_MAX && best_layer != LAYER);
        victim[best_layer][best_expert] = true;
    }

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(ROWS * FIX_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(ROWS * 10 * FIX_DIM * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(ROWS * FIX_DIM * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(ROWS * 10 * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(ROWS * 10 * sizeof(float));
    assert(out && mid && x && selected && weights);
    static float input[ROWS * FIX_DIM], actual_mid[ROWS * 10 * FIX_DIM];
    float ws[ROWS * 10];
    int32_t ids[ROWS * 10];
    for (unsigned i = 0; i < ROWS * FIX_DIM; i++) input[i] = 1.0f / FIX_DIM;
    for (unsigned r = 0; r < ROWS; r++)
        for (unsigned i = 0; i < 10; i++) {
            ids[r * 10 + i] = (int32_t)(r * STEP + i);
            ws[r * 10 + i] = 0.1f;
        }
    assert(ds4_gpu_tensor_write(x, 0, input, sizeof(input)));
    assert(ds4_gpu_tensor_write(selected, 0, ids, sizeof(ids)));
    assert(ds4_gpu_tensor_write(weights, 0, ws, sizeof(ws)));

    const uint64_t hits = g_stream_expert_cache_hits;
    const uint64_t misses = g_stream_expert_cache_misses;
    const uint64_t evictions = g_stream_expert_cache_evictions;
    const uint64_t reads = g_stream_expert_cache_pread_bytes;
    const uint64_t scans = g_stream_expert_timing_reuse_scan_calls;
    const uint64_t batches = g_stream_expert_timing_prepare_batch_reuse_calls;
    const uint64_t buffers = g_stream_expert_timing_prepare_buffer_calls;
    const uint64_t allocs = g_stream_expert_cache_buffer_allocs;
    assert(ds4_gpu_laguna_routed_moe_batch_tensor(
        out, mid, f->map, f->size,
        f->desc.gate_offset, f->desc.up_offset, f->desc.down_offset,
        f->desc.gate_type, f->desc.up_type, f->desc.down_type,
        f->desc.gate_expert_bytes, f->desc.gate_row_bytes,
        f->desc.up_expert_bytes, f->desc.up_row_bytes,
        f->desc.down_expert_bytes, f->desc.down_row_bytes,
        FIX_DIM, FIX_DIM, FIX_DIM, selected, weights, FIX_TOTAL, 10, LAYER,
        x, ROWS, 10 * FIX_DIM, false));
    assert(ds4_gpu_synchronize());

    assert(g_stream_expert_cache_misses - misses == UNION);
    assert(g_stream_expert_cache_hits - hits == ROWS * 10 - UNION);
    assert(g_stream_expert_cache_evictions - evictions == UNION);
    assert(g_stream_expert_cache_pread_bytes - reads == UNION * bytes);
    assert(g_stream_expert_timing_reuse_scan_calls - scans == (UNION + 9) / 10);
    assert(g_stream_expert_timing_prepare_batch_reuse_calls - batches == (UNION + 9) / 10);
    assert(g_stream_expert_timing_prepare_buffer_calls == buffers);
    assert(g_stream_expert_cache_buffer_allocs == allocs);
    assert(ds4_gpu_stream_expert_cache_current_count() == 1618);
    for (unsigned il = 0; il < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; il++)
        for (unsigned e = 0; e < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; e++) {
            if (il == LAYER) {
                assert((g_stream_expert_cache[il][e].valid != 0) == (e < UNION));
            } else if (victim[il][e]) {
                assert(!g_stream_expert_cache[il][e].valid);
            }
        }

    /* The reused buffers must hold the right bytes: every row against the
     * oracle. */
    assert(ds4_gpu_tensor_read(mid, 0, actual_mid, sizeof(actual_mid)));
    for (unsigned r = 0; r < ROWS; r++) {
        double expected[FIX_DIM], expected_mid[10 * FIX_DIM];
        fixture_reference(f, ids + r * 10, ws + r * 10, input + r * FIX_DIM,
                          expected, expected_mid);
        for (unsigned i = 0; i < 10 * FIX_DIM; i++) {
            const float a = actual_mid[r * 10 * FIX_DIM + i];
            assert(isfinite(a) &&
                   fabs(a - expected_mid[i]) < 1e-7 + 1e-4 * fabs(expected_mid[i]));
        }
    }
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(weights);
    puts("laguna-stream-batch-victims: OK (25-miss union, 3 scans, same victims)");
    return 1;
}

/* Fill the cache (1618 entries) always in the same order: layers 1..6 whole
 * plus experts 0..81 of layer 20. The state is identical at every call, so the
 * victims are identical for the two paths being compared. */
static void fill_cache_for_overlap(laguna_stream_fixture *f, ds4_gpu_tensor *out1,
                                   ds4_gpu_tensor *mid1, ds4_gpu_tensor *sel1,
                                   ds4_gpu_tensor *w1, ds4_gpu_tensor *x1) {
    ds4_gpu_stream_expert_cache_clear_all(0);
    ds4_gpu_stream_expert_cache_reset_route_hotness();
    int32_t ids[10];
    for (unsigned layer = 1; layer <= 7; layer++) {
        const unsigned count = layer == 7 ? 82 : 256, il = layer == 7 ? 20 : layer;
        for (unsigned done = 0; done < count; done += 10) {
            const unsigned first = count - done < 10 ? count - 10 : done;
            for (unsigned i = 0; i < 10; i++) ids[i] = (int32_t)(first + i);
            assert(ds4_gpu_tensor_write(sel1, 0, ids, sizeof(ids)));
            assert(ds4_gpu_laguna_stream_routed_moe_one_tensor(
                out1, mid1, f->map, f->size, &f->desc, FIX_DIM, FIX_DIM, FIX_DIM,
                sel1, w1, FIX_TOTAL, 10, il, x1));
        }
    }
    assert(ds4_gpu_stream_expert_cache_current_count() == 1618);
}

/* Every row of out and mid against the fixture's host oracle. */
static void assert_batch_reference(laguna_stream_fixture *f, unsigned rows,
                                   const int32_t *ids, const float *ws,
                                   const float *input, const float *out,
                                   const float *mid) {
    for (unsigned r = 0; r < rows; r++) {
        double expected[FIX_DIM], expected_mid[10 * FIX_DIM];
        fixture_reference(f, ids + r * 10, ws + r * 10, input + r * FIX_DIM,
                          expected, expected_mid);
        for (unsigned i = 0; i < 10 * FIX_DIM; i++) {
            const float a = mid[r * 10 * FIX_DIM + i];
            assert(isfinite(a) &&
                   fabs(a - expected_mid[i]) < 1e-7 + 1e-4 * fabs(expected_mid[i]));
        }
        for (unsigned i = 0; i < FIX_DIM; i++) {
            const float a = out[r * FIX_DIM + i];
            assert(isfinite(a) && fabs(a - expected[i]) < 1e-4 * (1 + fabs(expected[i])));
        }
    }
}

/* Consistent cache: per-layer and total counts equal the valid entries, and no
 * entry is still marked in flight. */
static void assert_cache_consistent(void) {
    uint32_t total = 0;
    for (unsigned il = 0; il < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; il++) {
        uint32_t valid = 0;
        for (unsigned e = 0; e < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; e++) {
            ds4_gpu_stream_expert_cache_entry *entry = &g_stream_expert_cache[il][e];
            if (!entry->valid) continue;
            assert(!ds4_gpu_stream_expert_cache_entry_inflight(entry));
            valid++;
        }
        assert(g_stream_expert_cache_layer_count[il] == valid);
        total += valid;
    }
    assert(g_stream_expert_cache_entry_count == total);
}

/* Batch consumer with overlap: at a full cache, mixed, all-missing and
 * all-resident unions, with experts shared across rows and across the two
 * masks, and with duplicate slots, must give out and mid bit-identical to the
 * serial path (DS4_LAGUNA_STREAM_BATCH_OVERLAP=0), with the same cache
 * afterwards. The blit after the router submit stands in for the shared expert:
 * independent work that must start before the read and stay correct. Then a
 * read error in the middle of the overlap must leave the cache consistent. */
static int check_batch_overlap(laguna_stream_fixture *f, ds4_gpu_tensor *out1,
                               ds4_gpu_tensor *mid1, ds4_gpu_tensor *sel1,
                               ds4_gpu_tensor *w1, ds4_gpu_tensor *x1) {
    enum { ROWS = 4, LAYER = 20, CASES = 5 };
    /* Layer 20 has experts 0..81 cached. */
    static const unsigned first_ids[CASES][ROWS] = {
        {0, 30, 60, 90},      /* mixed: 90..99 missing */
        {100, 130, 160, 190}, /* all missing */
        {0, 20, 40, 60},      /* all resident */
        {0, 5, 30, 76},       /* 5..9 in two rows; 76..85 straddle: 82..85 missing */
        {78, 80, 0, 10},      /* slots 8 and 9 repeat 0 and 1: 82..87 missing */
    };
    static const unsigned expected_missing[CASES] = {10, 40, 0, 4, 6};
    static const bool expected_mixed[CASES] = {true, false, false, true, true};
    static const bool duplicate_slots[CASES] = {false, false, false, false, true};
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(ROWS * FIX_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(ROWS * 10 * FIX_DIM * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(ROWS * FIX_DIM * sizeof(float));
    ds4_gpu_tensor *copy = ds4_gpu_tensor_alloc(ROWS * FIX_DIM * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(ROWS * 10 * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(ROWS * 10 * sizeof(float));
    assert(out && mid && x && copy && selected && weights);
    static float input[ROWS * FIX_DIM], copied[ROWS * FIX_DIM];
    static float result[2][ROWS * FIX_DIM], result_mid[2][ROWS * 10 * FIX_DIM];
    float ws[ROWS * 10];
    int32_t ids[ROWS * 10];
    for (unsigned i = 0; i < ROWS * FIX_DIM; i++)
        input[i] = (1.0f + (float)((i * 7u + i / FIX_DIM) % 13u)) / (9.0f * FIX_DIM);
    for (unsigned i = 0; i < ROWS * 10; i++) ws[i] = 0.05f + 0.01f * (float)(i % 7u);
    assert(ds4_gpu_tensor_write(x, 0, input, sizeof(input)));
    assert(ds4_gpu_tensor_write(weights, 0, ws, sizeof(ws)));

    for (unsigned c = 0; c < CASES; c++) {
        for (unsigned r = 0; r < ROWS; r++)
            for (unsigned i = 0; i < 10; i++)
                ids[r * 10 + i] = duplicate_slots[c] && i >= 8 ?
                    ids[r * 10 + i - 8] : (int32_t)(first_ids[c][r] + i);
        uint64_t delta[2][4];
        for (unsigned mode = 0; mode < 2; mode++) {
            assert(setenv("DS4_LAGUNA_STREAM_BATCH_OVERLAP", mode ? "1" : "0", 1) == 0);
            fill_cache_for_overlap(f, out1, mid1, sel1, w1, x1);
            assert(ds4_gpu_tensor_write(selected, 0, ids, sizeof(ids)));
            for (unsigned i = 0; i < ROWS * 10 * FIX_DIM; i++) result_mid[mode][i] = NAN;
            assert(ds4_gpu_tensor_write(mid, 0, result_mid[mode], sizeof(result_mid[mode])));
            const uint64_t hits = g_stream_expert_cache_hits;
            const uint64_t misses = g_stream_expert_cache_misses;
            const uint64_t evictions = g_stream_expert_cache_evictions;
            const uint64_t reads = g_stream_expert_cache_pread_bytes;
            const uint64_t layers = g_laguna_stream_batch_layers;
            const uint64_t overlapped = g_laguna_stream_batch_overlap_layers;
            const uint64_t mixed = g_laguna_stream_batch_mixed;
            const uint64_t missing = g_laguna_stream_batch_missing_experts;

            assert(ds4_gpu_begin_commands());
            bool overlap = !mode;
            assert(ds4_gpu_laguna_stream_batch_submit_router(LAYER, &overlap));
            assert(overlap == (mode == 1));
            assert(ds4_gpu_tensor_copy(copy, 0, x, 0, sizeof(input)));
            assert(ds4_gpu_laguna_routed_moe_batch_tensor(
                out, mid, f->map, f->size,
                f->desc.gate_offset, f->desc.up_offset, f->desc.down_offset,
                f->desc.gate_type, f->desc.up_type, f->desc.down_type,
                f->desc.gate_expert_bytes, f->desc.gate_row_bytes,
                f->desc.up_expert_bytes, f->desc.up_row_bytes,
                f->desc.down_expert_bytes, f->desc.down_row_bytes,
                FIX_DIM, FIX_DIM, FIX_DIM, selected, weights, FIX_TOTAL, 10, LAYER,
                x, ROWS, 10 * FIX_DIM, false));
            assert(g_laguna_stream_batch_router_layer == 0);
            assert(ds4_gpu_end_commands());

            assert(ds4_gpu_tensor_read(out, 0, result[mode], sizeof(result[mode])));
            assert(ds4_gpu_tensor_read(mid, 0, result_mid[mode], sizeof(result_mid[mode])));
            assert(ds4_gpu_tensor_read(copy, 0, copied, sizeof(copied)));
            assert(memcmp(copied, input, sizeof(input)) == 0);
            delta[mode][0] = g_stream_expert_cache_hits - hits;
            delta[mode][1] = g_stream_expert_cache_misses - misses;
            delta[mode][2] = g_stream_expert_cache_evictions - evictions;
            delta[mode][3] = g_stream_expert_cache_pread_bytes - reads;
            assert(delta[mode][1] == expected_missing[c]);
            assert(g_laguna_stream_batch_layers - layers == 1);
            assert(g_laguna_stream_batch_missing_experts - missing == expected_missing[c]);
            assert(g_laguna_stream_batch_mixed - mixed == expected_mixed[c]);
            assert(g_laguna_stream_batch_overlap_layers - overlapped ==
                   (mode == 1 && expected_mixed[c]));
            assert_cache_consistent();
            for (unsigned r = 0; r < ROWS; r++) {
                ds4_gpu_stream_expert_cache_entry *e =
                    &g_stream_expert_cache[LAYER][first_ids[c][r]];
                assert(e->valid && !ds4_gpu_stream_expert_cache_entry_inflight(e));
            }
        }
        /* The criterion is bit identity, not a tolerance. */
        assert(memcmp(result[0], result[1], sizeof(result[0])) == 0);
        assert(memcmp(result_mid[0], result_mid[1], sizeof(result_mid[0])) == 0);
        assert(memcmp(delta[0], delta[1], sizeof(delta[0])) == 0);
        /* And the common result is the right one, row by row. */
        assert_batch_reference(f, ROWS, ids, ws, input, result[1], result_mid[1]);
    }

    /* I/O error in the middle of the overlap, on the mixed case: the resident
     * pass is in flight when the read of the missing experts fails. The layer
     * must be empty again with a consistent count and no entry in flight, and
     * the same call must then succeed with the right result. */
    assert(setenv("DS4_LAGUNA_STREAM_BATCH_OVERLAP", "1", 1) == 0);
    fill_cache_for_overlap(f, out1, mid1, sel1, w1, x1);
    for (unsigned r = 0; r < ROWS; r++)
        for (unsigned i = 0; i < 10; i++) ids[r * 10 + i] = (int32_t)(first_ids[0][r] + i);
    assert(ds4_gpu_tensor_write(selected, 0, ids, sizeof(ids)));
    const uint64_t failures = g_laguna_stream_failures;
    const uint64_t overlapped = g_laguna_stream_batch_overlap_layers;
    const uint64_t layers = g_laguna_stream_batch_layers;
    /* The batch consumer tasks do not go through the fault_pread hook (they are
     * not tagged laguna): the error is produced by pointing the model
     * descriptor at a directory, where pread fails with EISDIR. */
    const int model_fd = g_model_fd;
    const int dir_fd = open(".", O_RDONLY);
    assert(dir_fd >= 0);
    g_model_fd = dir_fd;
    assert(ds4_gpu_begin_commands());
    bool overlap = false;
    assert(ds4_gpu_laguna_stream_batch_submit_router(LAYER, &overlap) && overlap);
    assert(!ds4_gpu_laguna_routed_moe_batch_tensor(
        out, mid, f->map, f->size,
        f->desc.gate_offset, f->desc.up_offset, f->desc.down_offset,
        f->desc.gate_type, f->desc.up_type, f->desc.down_type,
        f->desc.gate_expert_bytes, f->desc.gate_row_bytes,
        f->desc.up_expert_bytes, f->desc.up_row_bytes,
        f->desc.down_expert_bytes, f->desc.down_row_bytes,
        FIX_DIM, FIX_DIM, FIX_DIM, selected, weights, FIX_TOTAL, 10, LAYER,
        x, ROWS, 10 * FIX_DIM, false));
    g_model_fd = model_fd;
    assert(close(dir_fd) == 0);
    if (ds4_gpu_commands_active()) assert(ds4_gpu_end_commands());
    assert(g_laguna_stream_failures - failures == 1);
    assert(g_laguna_stream_batch_layers - layers == 1);
    assert(g_laguna_stream_batch_overlap_layers - overlapped == 1);
    assert(!g_stream_expert_pending_load.active);
    assert(g_stream_expert_cache_layer_count[LAYER] == 0);
    for (unsigned e = 0; e < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; e++)
        assert(!g_stream_expert_cache[LAYER][e].valid);
    assert_cache_consistent();
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_laguna_routed_moe_batch_tensor(
        out, mid, f->map, f->size,
        f->desc.gate_offset, f->desc.up_offset, f->desc.down_offset,
        f->desc.gate_type, f->desc.up_type, f->desc.down_type,
        f->desc.gate_expert_bytes, f->desc.gate_row_bytes,
        f->desc.up_expert_bytes, f->desc.up_row_bytes,
        f->desc.down_expert_bytes, f->desc.down_row_bytes,
        FIX_DIM, FIX_DIM, FIX_DIM, selected, weights, FIX_TOTAL, 10, LAYER,
        x, ROWS, 10 * FIX_DIM, false));
    assert(ds4_gpu_end_commands());
    assert(g_stream_expert_cache_layer_count[LAYER] == 40);
    assert_cache_consistent();
    assert(ds4_gpu_tensor_read(out, 0, result[1], sizeof(result[1])));
    assert(ds4_gpu_tensor_read(mid, 0, result_mid[1], sizeof(result_mid[1])));
    assert_batch_reference(f, ROWS, ids, ws, input, result[1], result_mid[1]);
    assert(unsetenv("DS4_LAGUNA_STREAM_BATCH_OVERLAP") == 0);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(copy); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(weights);
    puts("laguna-stream-batch-overlap: OK (mixed/all-missing/all-resident, shared and "
         "duplicate experts, bit-identical to serial, I/O failure cleanup)");
    return 1;
}

static int check_saturation(bool slabs) {
    @autoreleasepool {
        /* Same slot count as the real 8 GiB cache; payload 48 times smaller. */
        laguna_stream_fixture f = fixture_open_case(false);
        if (slabs) assert(unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS") == 0);
        else assert(setenv("DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS", "1", 1) == 0);
        assert(setenv("DS4_METAL_STREAMING_EXPERT_SLAB_MB", "90", 1) == 0);
        if (!ds4_gpu_init()) { fixture_close(&f); return 0; }
        ds4_gpu_set_ssd_streaming(true);
        ds4_gpu_set_streaming_expert_cache_budget(1618);
        assert(ds4_gpu_set_model_fd(fileno(f.file)));
        const uint64_t off = 0, len = FIX_PREFIX;
        assert(ds4_gpu_set_model_map_spans(f.map, f.size, &off, &len, 1, len));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(FIX_DIM * sizeof(float));
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(10 * FIX_DIM * sizeof(float));
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(FIX_DIM * sizeof(float));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(10 * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(10 * sizeof(float));
        assert(out && mid && x && selected && weights);
        float input[FIX_DIM], ws[10]; int32_t ids[10];
        for (unsigned i = 0; i < FIX_DIM; i++) input[i] = 1.0f / FIX_DIM;
        for (unsigned i = 0; i < 10; i++) ws[i] = 0.1f;
        assert(ds4_gpu_tensor_write(x, 0, input, sizeof(input)));
        assert(ds4_gpu_tensor_write(weights, 0, ws, sizeof(ws)));
        uint64_t slot = 0, limit = 0;
        const uint64_t bytes = 2 * f.desc.gate_expert_bytes + f.desc.down_expert_bytes;
        assert(bytes * 48 == 5308416);
        assert(ds4_gpu_laguna_stream_allocation_limit(bytes, 1618, &slot, &limit));
        unsigned total = 0, calls = 0;
        const uint64_t views = g_test_model_range_calls;
        g_test_laguna_stream_peak_allocated_bytes = 0;
        for (unsigned layer = 1; layer <= 7; layer++) {
            const unsigned count = layer == 7 ? 82 : 256;
            for (unsigned done = 0; done < count; done += 10) {
                const unsigned added = count - done < 10 ? count - done : 10;
                const unsigned first = added < 10 ? count - 10 : done;
                for (unsigned i = 0; i < 10; i++) ids[i] = (int32_t)(first + i);
                assert(ds4_gpu_tensor_write(selected, 0, ids, sizeof(ids)));
                uint64_t misses = g_stream_expert_cache_misses;
                assert(ds4_gpu_laguna_stream_routed_moe_one_tensor(
                    out, mid, f.map, f.size, &f.desc, FIX_DIM, FIX_DIM, FIX_DIM,
                    selected, weights, FIX_TOTAL, 10, layer, x));
                total += added; calls++;
                assert(g_stream_expert_cache_misses - misses == added);
                assert(ds4_gpu_stream_expert_cache_current_count() == total);
                assert(g_laguna_stream_allocated_bytes <= limit);
            }
        }
        unsigned occupied = 0;
        for (unsigned il = 0; il < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; il++)
            for (unsigned e = 0; e < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; e++)
                occupied += g_stream_expert_cache[il][e].valid != 0;
        assert(total == 1618 && occupied == 1618);
        assert(g_stream_expert_cache_slab_count == (slabs ? 2u : 0u));
        assert(g_stream_expert_cache_buffer_allocs == (slabs ? 2u : 1618u));
        assert(g_test_laguna_stream_peak_allocated_bytes == limit);
        const uint64_t allocs = g_stream_expert_cache_buffer_allocs;
        /* Reset the hotness of the old prompt: the overlapping tails must not
         * artificially protect some victims of the first saturation. */
        ds4_gpu_stream_expert_cache_reset_route_hotness();
        /* Replace every entry, then read the whole set back as hits. */
        for (unsigned pass = 0; pass < 2; pass++) {
            const uint64_t evictions = g_stream_expert_cache_evictions;
            unsigned reused = 0;
            for (unsigned layer = 8; layer <= 14; layer++) {
                const unsigned count = layer == 14 ? 82 : 256;
                for (unsigned done = 0; done < count; done += 10) {
                    const unsigned added = count - done < 10 ? count - done : 10;
                    const unsigned first = added < 10 ? count - 10 : done;
                    for (unsigned i = 0; i < 10; i++) ids[i] = (int32_t)(first + i);
                    assert(ds4_gpu_tensor_write(selected, 0, ids, sizeof(ids)));
                    const uint64_t reads = g_stream_expert_cache_pread_bytes;
                    assert(ds4_gpu_laguna_stream_routed_moe_one_tensor(
                        out, mid, f.map, f.size, &f.desc, FIX_DIM, FIX_DIM, FIX_DIM,
                        selected, weights, FIX_TOTAL, 10, layer, x));
                    calls++; reused += added;
                    assert(g_stream_expert_cache_pread_bytes - reads == (pass == 0 ? added * bytes : 0));
                    assert(ds4_gpu_stream_expert_cache_current_count() == 1618);
                    assert(g_stream_expert_cache_buffer_allocs == allocs);
                }
            }
            assert(reused == 1618);
            assert(g_stream_expert_cache_evictions - evictions == (pass == 0 ? 1618 : 0));
        }
        occupied = 0;
        for (unsigned il = 0; il < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; il++)
            for (unsigned e = 0; e < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; e++) {
                const bool expected = il >= 8 && il <= 14 && e < (il == 14 ? 82u : 256u);
                assert((g_stream_expert_cache[il][e].valid != 0) == expected);
                occupied += expected;
            }
        assert(occupied == 1618);
        assert(g_laguna_stream_moe_calls == calls && g_laguna_stream_token_rows == 26);
        assert(!g_stream_expert_cache_decode_tokens && !g_laguna_stream_failures);
        assert(g_test_model_range_calls == views);
        assert(check_batch_victims(&f, bytes));
        assert(check_batch_overlap(&f, out, mid, selected, weights, x));
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(weights);
        ds4_gpu_cleanup(); assert_cache_drained(); fixture_close(&f);
        printf("laguna-stream-saturation: OK (1618 slots, slabs=%d, reuse, counters)\n", slabs);
        return 1;
    }
}

/* Exercise the same workers and fault hook without a Metal device. */
static void check_pread_pool(void) {
    unsigned char source[30 * 64], loaded[sizeof(source)];
    for (unsigned i = 0; i < sizeof(source); i++) source[i] = (unsigned char)i;
    for (unsigned failure = 0; failure < 4; failure++) {
        FILE *file = tmpfile();
        assert(file && fwrite(source, 1, sizeof(source), file) == sizeof(source));
        assert(fflush(file) == 0);
        g_model_fd = fileno(file);
        ds4_gpu_stream_expert_pread_task tasks[30] = {0};
        for (unsigned i = 0; i < 30; i++) {
            tasks[i] = (ds4_gpu_stream_expert_pread_task){
                .offset = i * 64, .len = 64, .dst = loaded + i * 64, .laguna = 1,
            };
        }
        read_calls = 0;
        fail_read = failure ? 6 : 0;
        read_error = failure == 1 ? EIO : EINTR;
        truncate_read = failure == 3;
        g_test_laguna_stream_temporary_bytes = sizeof(loaded);
        g_test_laguna_stream_pread = fault_pread;
        const uint64_t before = g_test_laguna_stream_pool_tasks;
        uint64_t bytes = 0;
        double ms = 0;
        const int ok = ds4_gpu_stream_expert_pread_tasks(tasks, 30, &bytes, &ms);
        assert(ok == (failure == 0));
        assert(g_test_laguna_stream_pool_tasks - before == 30);
        assert(g_stream_expert_pread_pool_thread_count > 1);
        assert(!g_stream_expert_pread_pool_tasks && !g_stream_expert_pread_pool_remaining_workers);
        if (ok) {
            assert(read_calls == 31 && bytes == sizeof(source));
            assert(memcmp(source, loaded, sizeof(source)) == 0);
        } else {
            assert(read_calls >= fail_read && fail_read > 0);
            assert(bytes < sizeof(source));
        }
        g_test_laguna_stream_pread = NULL;
        g_test_laguna_stream_temporary_bytes = 0;
        g_model_fd = -1;
        fclose(file);
    }
    ds4_gpu_stream_expert_pread_pool_shutdown();
    puts("laguna-stream-pread: OK (pool, partial reads, EIO, EINTR, EOF, drain)");
}

int main(int argc, char **argv) {
    assert(setenv("DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY", "1", 1) == 0);
    assert(unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_TIMING_SUMMARY") == 0);
    assert(setenv("DS4_METAL_STREAMING_EXPERT_PREAD_POOL", "1", 1) == 0);
    assert(setenv("DS4_METAL_STREAMING_EXPERT_PREAD_THREADS", "9", 1) == 0);
    check_allocation_limits();
    check_legacy_selection_limit();
    check_pread_pool();
    if (argc == 2 && strcmp(argv[1], "--pread-only") == 0) return 0;
    assert(unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_COMBINED_BUFFER") == 0);
    assert(unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS") == 0);
    assert(setenv("DS4_METAL_STREAMING_EXPERT_SLAB_MB", "4096", 1) == 0);
    if (!check_slab_capacity()) return 1;
    laguna_stream_fixture previous = {0};
    for (unsigned mode = 0; mode < 3; mode++) {
        if (mode == 2) assert(setenv("DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS", "1", 1) == 0);
        else assert(unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS") == 0);
        assert(setenv("DS4_METAL_STREAMING_EXPERT_SLAB_MB", mode == 1 ? "1" : "4096", 1) == 0);
        if (run_case(false, &previous) || run_case(true, &previous)) return 1;
    }
    fixture_close(&previous);
    assert(check_saturation(true));
    assert(check_saturation(false));
    return 0;
}
