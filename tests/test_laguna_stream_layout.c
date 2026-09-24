/* Metadata-only tests. The optional GGUF probe never reads tensor payloads. */
#define DS4_NO_GPU
#define DS4_TEST_HOOKS
#include "../ds4.c"
#include <assert.h>
#ifdef NDEBUG
#error "Laguna tests require assertions"
#endif

static ds4_tensor *add_tensor(ds4_model *m, uint32_t type, uint32_t ndim,
                              uint64_t d0, uint64_t d1, uint64_t d2) {
    assert(m->n_tensors < 1024);
    ds4_tensor *t = &m->tensors[m->n_tensors++];
    *t = (ds4_tensor){.type = type, .ndim = ndim, .dim = {d0, d1, d2},
                      .elements = d0 * (ndim > 1 ? d1 : 1) * (ndim > 2 ? d2 : 1)};
    const uint64_t block = type == DS4_TENSOR_Q4_K ? 256 : type == DS4_TENSOR_Q8_0 ? 32 : 1;
    const uint64_t bytes = type == DS4_TENSOR_Q4_K ? 144 : type == DS4_TENSOR_Q8_0 ? 34 : 4;
    t->bytes = t->elements / block * bytes;
    return t;
}

static void synthetic_model(ds4_model *m, ds4_weights *w) {
    static const uint8_t sentinel;
    *m = (ds4_model){.map = &sentinel, .tensor_data_pos = 16384,
                     .tensors = calloc(1024, sizeof(ds4_tensor))};
    assert(m->tensors);
    memset(w, 0, sizeof(*w));
    g_ds4_shape = DS4_SHAPE_LAGUNA_S21;
#define ADD(field_, type_, nd_, d0_, d1_, d2_) \
    (field_) = add_tensor(m, DS4_TENSOR_##type_, nd_, d0_, d1_, d2_)
    ADD(w->token_embd, Q8_0, 2, 3072, 100352, 0);
    ADD(w->output, Q8_0, 2, 3072, 100352, 0);
    ADD(w->output_norm, F32, 1, 3072, 0, 0);
    for (uint32_t il = 0; il < 48; il++) {
        ds4_layer_weights *l = &w->layer[il];
        const uint32_t heads = il % 4 == 0 ? 48 : 72;
        g_ds4_head_counts[il] = heads;
        ADD(l->attn_norm, F32, 1, 3072, 0, 0);
        ADD(l->attn_q, Q8_0, 2, 3072, heads * 128, 0);
        ADD(l->attn_k, Q8_0, 2, 3072, 1024, 0);
        ADD(l->attn_v, Q8_0, 2, 3072, 1024, 0);
        ADD(l->attn_gate, Q8_0, 2, 3072, heads, 0);
        ADD(l->attn_q_norm, F32, 1, 128, 0, 0);
        ADD(l->attn_k_norm, F32, 1, 128, 0, 0);
        ADD(l->attn_output, Q8_0, 2, heads * 128, 3072, 0);
        ADD(l->ffn_norm, F32, 1, 3072, 0, 0);
        if (il == 0) {
            ADD(l->ffn_gate, Q8_0, 2, 3072, 12288, 0);
            ADD(l->ffn_up, Q8_0, 2, 3072, 12288, 0);
            ADD(l->ffn_down, Q8_0, 2, 12288, 3072, 0);
        } else {
            ADD(l->ffn_gate_inp, F32, 2, 3072, 256, 0);
            ADD(l->ffn_exp_probs_b, F32, 1, 256, 0, 0);
            ADD(l->ffn_gate_exps, Q4_K, 3, 3072, 1024, 256);
            ADD(l->ffn_up_exps, Q4_K, 3, 3072, 1024, 256);
            ADD(l->ffn_down_exps, Q4_K, 3, 1024, 3072, 256);
            ADD(l->ffn_gate_shexp, Q8_0, 2, 3072, 1024, 0);
            ADD(l->ffn_up_shexp, Q8_0, 2, 3072, 1024, 0);
            ADD(l->ffn_down_shexp, Q8_0, 2, 1024, 3072, 0);
        }
    }
#undef ADD
    /* An unbound non-routed tensor must be resident too. Reverse file order. */
    add_tensor(m, DS4_TENSOR_F32, 1, 16, 0, 0);
    m->size = m->tensor_data_pos;
    for (uint64_t i = m->n_tensors; i-- > 0;) {
        m->tensors[i].abs_offset = m->size;
        m->size += m->tensors[i].bytes + (i % 3 == 0 ? 64 : 0);
    }
}

static void check_plan(const ds4_model *m, const ds4_weights *w) {
    ds4_gpu_stream_expert_table tables[48];
    uint64_t entry = 0, non_routed = 0, routed = 0;
    ds4_model_map_span_vec spans;
    assert(laguna_stream_expert_tables_make(m, w, tables, &entry));
    assert(entry == 5308416);
    assert(tables[0].model_map == NULL);
    for (uint32_t il = 1; il < 48; il++) {
        const ds4_gpu_stream_expert_table *t = &tables[il];
        assert(t->model_map == m->map && t->model_size == m->size);
        assert(t->layer == il && t->n_total_expert == 256);
        assert(t->gate_expert_bytes == 1769472 && t->down_expert_bytes == 1769472);
        const ds4_tensor *projections[] = {w->layer[il].ffn_gate_exps,
            w->layer[il].ffn_up_exps, w->layer[il].ffn_down_exps};
        const uint64_t bases[] = {t->gate_offset, t->up_offset, t->down_offset};
        for (unsigned p = 0; p < 3; p++) {
            assert(bases[p] == projections[p]->abs_offset);
            for (uint32_t e = 0; e < 256; e++) {
                const uint64_t lo = bases[p] + e * 1769472ull;
                assert(lo >= bases[p] && lo + 1769472 <= m->size);
                assert(lo + 1769472 <= bases[p] + projections[p]->bytes);
            }
            assert(bases[p] + 256 * 1769472ull == bases[p] + projections[p]->bytes);
        }
    }
    assert(laguna_stream_model_spans(m, w, &spans, &non_routed, &routed));
    assert(spans.len > 0 && spans.v && non_routed > 0);
    assert(routed == 47ull * 256 * 5308416);
    uint64_t span_bytes = 0, all_bytes = 0;
    for (uint32_t s = 0; s < spans.len; s++) {
        assert(spans.v[s].off < spans.v[s].end && spans.v[s].end <= m->size);
        if (s) assert(spans.v[s - 1].end < spans.v[s].off);
        span_bytes += spans.v[s].end - spans.v[s].off;
    }
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        bool covered = false;
        /* Classify independently of the helper under test. */
        const bool is_routed = t->type == DS4_TENSOR_Q4_K && t->ndim == 3;
        all_bytes += t->bytes;
        for (uint32_t s = 0; s < spans.len; s++) {
            const ds4_model_map_span *v = &spans.v[s];
            if (is_routed) assert(v->end <= t->abs_offset || v->off >= t->abs_offset + t->bytes);
            else if (v->off <= t->abs_offset && v->end >= t->abs_offset + t->bytes) covered = true;
        }
        assert(is_routed || covered);
    }
    assert(non_routed == span_bytes && all_bytes == non_routed + routed);
    laguna_stream_cache_config cache;
    assert(laguna_stream_configure_cache(entry, 0, 0, UINT64_MAX, &cache));
    assert(cache.experts == 1618 && cache.payload_bytes == 8589017088ull);
    printf("laguna layout: non-routed=%" PRIu64 " routed=%" PRIu64
           " entry=%" PRIu64 " spans=%u cache8GiB=%u payload=%" PRIu64 "\n",
           non_routed, routed, entry, spans.len, cache.experts, cache.payload_bytes);
    free(spans.v);
}

static void check_rejections(ds4_model *m, ds4_weights *w) {
    unsigned mutated = 0;
    for (uint32_t il = 0; il < 48; il++) {
        ds4_layer_weights *l = &w->layer[il];
        ds4_tensor *ts[] = {l->attn_q, l->attn_k, l->attn_v, l->attn_gate,
            l->attn_output, l->attn_norm, l->attn_q_norm, l->attn_k_norm, l->ffn_norm,
            l->ffn_gate, l->ffn_up, l->ffn_down,
            l->ffn_gate_inp, l->ffn_exp_probs_b, l->ffn_gate_exps, l->ffn_up_exps,
            l->ffn_down_exps, l->ffn_gate_shexp, l->ffn_up_shexp, l->ffn_down_shexp};
        for (unsigned i = 0; i < sizeof(ts) / sizeof(*ts); i++) {
            if (!ts[i]) continue;
            mutated++;
            const ds4_tensor saved = *ts[i];
            ts[i]->type = DS4_TENSOR_Q6_K;
            assert(!laguna_stream_layout_supported(m, w));
            *ts[i] = saved;
            ts[i]->dim[0]++;
            assert(!laguna_stream_layout_supported(m, w));
            *ts[i] = saved;
            ts[i]->bytes--;
            assert(!laguna_stream_layout_supported(m, w));
            *ts[i] = saved;
        }
        const uint32_t heads = g_ds4_head_counts[il];
        g_ds4_head_counts[il] = 0;
        assert(!laguna_stream_layout_supported(m, w));
        g_ds4_head_counts[il] = heads;
    }
    assert(mutated == 12 + 47 * 17);
    ds4_tensor *t = w->layer[47].ffn_up_exps;
    const ds4_tensor saved = *t;
    ds4_tensor boundary = saved;
    boundary.abs_offset = m->size - boundary.bytes;
    assert(laguna_stream_tensor_valid(m, &boundary));
    boundary.abs_offset++;
    assert(!laguna_stream_tensor_valid(m, &boundary));
    t->abs_offset = m->size - t->bytes + 1;
    assert(!laguna_stream_layout_supported(m, w));
    t->abs_offset = UINT64_MAX - 8;
    assert(!laguna_stream_layout_supported(m, w));
    *t = saved;
    t->abs_offset = w->token_embd->abs_offset;
    assert(!laguna_stream_layout_supported(m, w));
    *t = saved;
    w->layer[47].ffn_up_exps = NULL;
    ds4_gpu_stream_expert_table tables[48];
    uint64_t entry = 99, nr = 99, r = 99;
    ds4_model_map_span_vec spans;
    assert(!laguna_stream_expert_tables_make(m, w, tables, &entry) && entry == 0);
    for (unsigned i = 0; i < 48; i++) assert(!tables[i].model_map);
    assert(!laguna_stream_model_spans(m, w, &spans, &nr, &r));
    assert(!spans.v && !nr && !r);
    w->layer[47].ffn_up_exps = t;
    w->layer[0].ffn_gate_exps = t;
    assert(!laguna_stream_layout_supported(m, w));
    w->layer[0].ffn_gate_exps = NULL;
    w->layer[47].ffn_up_exps = w->layer[47].ffn_gate_exps;
    assert(!laguna_stream_layout_supported(m, w));
    w->layer[47].ffn_up_exps = t;
    ds4_tensor *extra = &m->tensors[m->n_tensors - 1];
    const ds4_tensor extra_saved = *extra;
    extra->ndim = 2; extra->dim[0] = UINT64_MAX; extra->dim[1] = 2;
    assert(!laguna_stream_layout_supported(m, w));
    extra->ndim = 1; extra->dim[0] = UINT64_MAX / 2;
    assert(!laguna_stream_layout_supported(m, w));
    *extra = extra_saved;
    extra->abs_offset = w->layer[47].ffn_gate_exps->abs_offset;
    assert(!laguna_stream_layout_supported(m, w));
    *extra = extra_saved;
    g_ds4_shape.n_expert_used = 8;
    assert(!laguna_stream_layout_supported(m, w));
    g_ds4_shape = DS4_SHAPE_LAGUNA_S21;
    g_ds4_shape.n_layer = 47;
    assert(!laguna_stream_layout_supported(m, w));
    g_ds4_shape = DS4_SHAPE_LAGUNA_S21;
    g_ds4_shape.n_expert = 255;
    assert(!laguna_stream_layout_supported(m, w));
    g_ds4_shape = DS4_SHAPE_LAGUNA_S21;
    g_ds4_shape.n_leading_dense = 0;
    assert(!laguna_stream_layout_supported(m, w));
    g_ds4_shape = DS4_SHAPE_LAGUNA_S21;
    g_ds4_shape.family = DS4_MODEL_FAMILY_DEEPSEEK4;
    assert(!laguna_stream_layout_supported(m, w));
    g_ds4_shape = DS4_SHAPE_LAGUNA_S21;
    ds4_tensor *globals[] = {w->token_embd, w->output, w->output_norm};
    for (unsigned i = 0; i < 3; i++) {
        const uint32_t type = globals[i]->type;
        globals[i]->type = DS4_TENSOR_Q4_K;
        assert(!laguna_stream_layout_supported(m, w));
        globals[i]->type = type;
    }
    assert(laguna_stream_layout_supported(m, w));
}

static void check_cache(void) {
    const uint64_t entry = 5308416;
    laguna_stream_cache_config c;
    assert(!laguna_stream_configure_cache(entry, 9, 0, UINT64_MAX, &c) && !c.experts);
    assert(laguna_stream_configure_cache(entry, 10, 0, UINT64_MAX, &c) && c.experts == 10);
    assert(!laguna_stream_configure_cache(entry, 0, 10 * entry - 1, UINT64_MAX, &c));
    assert(laguna_stream_configure_cache(entry, 0, 10 * entry, UINT64_MAX, &c) && c.experts == 10);
    assert(!laguna_stream_configure_cache(entry, 0, 0, 9 * entry, &c));
    assert(laguna_stream_configure_cache(entry, 0, 0, 10 * entry, &c) && c.experts == 10);
    assert(!laguna_stream_configure_cache(entry, 11, 0, 10 * entry, &c));
    assert(!laguna_stream_configure_cache(entry, 10, entry * 10, UINT64_MAX, &c));
    assert(!laguna_stream_configure_cache(UINT64_MAX, 10, 0, UINT64_MAX, &c));
    assert(!laguna_stream_configure_cache(0, 0, 0, UINT64_MAX, &c));
    assert(!laguna_stream_configure_cache(entry, UINT32_MAX, 0, UINT64_MAX, &c));
    assert(laguna_stream_configure_cache(entry, 0, UINT64_MAX, UINT64_MAX, &c));
    assert(c.experts == 47 * 256);
    assert(!laguna_stream_configure_cache(entry, 0, 0, 0, &c));
}

static void check_streaming_requests(void) {
    ds4_engine_options opt = {.backend = DS4_BACKEND_METAL, .ssd_streaming = true};
    const char *blocked[] = {
        "DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE",
        "DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE",
        "DS4_METAL_ENABLE_STREAMING_COMPACT_ADDR",
        "DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER",
    };
    char *saved[4];
    for (unsigned i = 0; i < 4; i++) {
        const char *value = getenv(blocked[i]);
        saved[i] = value ? strdup(value) : NULL;
        unsetenv(blocked[i]);
    }
    assert(!laguna_stream_options_error(&opt));
#define REJECT(field_, value_) do { \
    ds4_engine_options bad = opt; bad.field_ = (value_); \
    assert(laguna_stream_options_error(&bad)); \
} while (0)
    REJECT(backend, DS4_BACKEND_CPU);
    REJECT(backend, DS4_BACKEND_CUDA);
    REJECT(dflash_path, "draft.gguf");
    REJECT(dflash_draft_tokens, 1);
    REJECT(dflash_p_min_set, true);
    REJECT(mtp_path, "mtp.gguf");
    REJECT(warm_weights, true);
    REJECT(ssd_streaming_full_layers_set, true);
    REJECT(ssd_streaming_full_layers, 1);
    REJECT(ssd_streaming_preload_experts, 10);
    REJECT(ssd_streaming_cold, true);
    REJECT(metal_graph_test, true);
#undef REJECT
    for (unsigned i = 0; i < 4; i++) {
        assert(setenv(blocked[i], "0", 1) == 0);
        assert(strcmp(laguna_stream_options_error(&opt), blocked[i]) == 0);
        unsetenv(blocked[i]);
    }
    for (unsigned i = 0; i < 4; i++) {
        if (saved[i]) { setenv(blocked[i], saved[i], 1); free(saved[i]); }
    }
    assert(laguna_stream_request_supported(NULL, NULL, NULL));
    /* A non-null pointer is enough to reject even a single verifier row. */
    for (unsigned mask = 1; mask < 8; mask++)
        assert(!laguna_stream_request_supported(mask & 1 ? &opt : NULL,
                mask & 2 ? &opt : NULL, mask & 4 ? &opt : NULL));
    ds4_engine engine = {.backend = DS4_BACKEND_METAL, .ssd_streaming = true};
    ds4_session session = {.engine = &engine};
    char err[128] = {0};
    int accepted = -1;
    assert(ds4_session_eval_speculative_argmax(&session, 0, 1, 2,
            &accepted, 1, err, sizeof(err)) == -1);
    assert(strstr(err, "does not support speculative decoding"));
    assert(accepted == -1 && session.checkpoint.len == 0);
}

static void check_memory_admission(void) {
    uint64_t kv, scratch;
    assert(laguna_stream_graph_bytes(256, &kv, &scratch));
    assert(kv == 48ull * 256 * 1024 * 2 * 2);
    assert(scratch > 0 && scratch < 1048576);
    const uint64_t one_row_scratch = scratch;
    const uint32_t contexts[] = {1, 256, 512, 513, 1024};
    uint64_t previous = 0;
    for (unsigned i = 0; i < sizeof(contexts) / sizeof(contexts[0]); i++) {
        const uint32_t ctx = contexts[i], swa = ctx < 512 ? ctx : 512;
        assert(laguna_stream_graph_bytes(ctx, &kv, &scratch));
        /* Twelve full layers and thirty-six SWA layers, fp16 KV. */
        assert(kv == (12ull * ctx + 36ull * swa) * 1024 * 2 * 2);
        assert(kv > previous && scratch == one_row_scratch);
        previous = kv;
        const uint64_t rec = 48ull << 30, non_routed = 5ull << 30;
        assert(laguna_stream_available_cache_bytes(rec, non_routed, ctx) ==
               rec / 5 * 4 - non_routed - kv - scratch - (1ull << 30));
        const uint64_t cap = laguna_stream_available_cache_bytes(rec, non_routed, ctx);
        laguna_stream_cache_config config;
        assert(laguna_stream_configure_cache(5308416, 0, 0, cap, &config));
        assert(config.experts == 1618 && config.budget_bytes == (8ull << 30));
        assert(!laguna_stream_configure_cache(5308416, 0, cap + 1, cap, &config));
    }
    assert(!laguna_stream_graph_bytes(0, &kv, &scratch));
    assert(!laguna_stream_graph_bytes(UINT32_MAX, &kv, &scratch));
    const uint64_t gib = 1ull << 30, entry = 5308416;
    const uint64_t cap = laguna_stream_available_cache_bytes(48 * gib, 5 * gib, 256);
    assert(cap > 8 * gib);
    assert(!laguna_stream_available_cache_bytes(0, 5 * gib, 256));
    assert(!laguna_stream_available_cache_bytes(48 * gib, UINT64_MAX, 256));
    assert(!laguna_stream_available_cache_bytes(6 * gib, 5 * gib, 256));
    laguna_stream_cache_config cache;
    assert(laguna_stream_configure_cache(entry, 0, 0, cap, &cache));
    assert(cache.experts == 1618 && cache.budget_bytes == 8 * gib);
    assert(!laguna_stream_configure_cache(entry, 0, cap + 1, cap, &cache));
    assert(laguna_stream_configure_cache(entry, 0, 0, 20 * entry, &cache));
    assert(cache.experts == 20);
}

static void check_engine_rejections(void) {
    /* GGUF deliberately without weights: every rejection must come before
     * binding. */
    char path[] = "/tmp/laguna-flags-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    assert(fp);
    const uint32_t header[] = {DS4_GGUF_MAGIC, 3};
    const uint64_t counts[] = {0, 1}, key_len = 20, value_len = 6;
    const uint32_t string_type = 8;
    assert(fwrite(header, sizeof(header), 1, fp) == 1);
    assert(fwrite(counts, sizeof(counts), 1, fp) == 1);
    assert(fwrite(&key_len, sizeof(key_len), 1, fp) == 1);
    assert(fwrite("general.architecture", 20, 1, fp) == 1);
    assert(fwrite(&string_type, sizeof(string_type), 1, fp) == 1);
    assert(fwrite(&value_len, sizeof(value_len), 1, fp) == 1);
    assert(fwrite("laguna", 6, 1, fp) == 1);
    assert(fflush(fp) == 0 && ftruncate(fd, 128) == 0);
    assert(fclose(fp) == 0);
    const ds4_engine_options base = {.model_path = path,
        .backend = DS4_BACKEND_METAL, .ssd_streaming = true};
    ds4_engine_options cases[7] = {base, base, base, base, base, base, base};
    cases[0].backend = DS4_BACKEND_CPU;
    cases[1].backend = DS4_BACKEND_CUDA;
    cases[2].dflash_path = "unused-draft.gguf";
    cases[3].warm_weights = true;
    cases[4].head_test = true;
    cases[5].head_test = true; cases[5].ssd_streaming = false;
    cases[6].mtp_path = "unused-mtp.gguf";
    const char *messages[] = {"Metal backend", "Metal backend", "DFlash",
        "--warm-weights", "--head-test is not supported for Laguna",
        "--head-test is not supported for Laguna", "MTP"};
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        FILE *log = tmpfile();
        assert(log);
        int saved = dup(STDERR_FILENO);
        assert(saved >= 0 && dup2(fileno(log), STDERR_FILENO) >= 0);
        ds4_engine *engine = NULL;
        int rc = ds4_engine_open(&engine, &cases[i]);
        assert(fflush(stderr) == 0 && dup2(saved, STDERR_FILENO) >= 0);
        close(saved);
        rewind(log);
        char error[2048] = {0};
        assert(fread(error, 1, sizeof(error) - 1, log) > 0);
        if (!strstr(error, messages[i])) fprintf(stderr, "case %u: %s", i, error);
        assert(rc != 0 && engine == NULL && strstr(error, messages[i]));
        fclose(log);
    }
    assert(unlink(path) == 0);
    ds4_engine engine = {0};
    int token = 2;
    ds4_tokens prompt = {.v = &token, .len = 1};
    assert(ds4_engine_head_test(&engine, &prompt) != 0);
    engine.ssd_streaming = true;
    assert(ds4_engine_head_test(&engine, &prompt) != 0);
}

int main(int argc, char **argv) {
    ds4_model m;
    ds4_weights w;
    if (argc == 3 && strcmp(argv[1], "--inspect-model") == 0) {
        model_open(&m, argv[2], false, false);
        config_validate_model(&m);
        weights_bind(&w, &m, false, 0, UINT32_MAX, true);
        check_plan(&m, &w);
        model_close(&m);
        return 0;
    }
    assert(argc == 1);
    synthetic_model(&m, &w);
    check_plan(&m, &w);
    check_rejections(&m, &w);
    check_cache();
    check_streaming_requests();
    check_memory_admission();
    check_engine_rejections();
    free(m.tensors);
    puts("laguna-stream-layout-host: OK (all layers, tables, spans, overflow, cache, flags, DFlash, memory)");
    return 0;
}
