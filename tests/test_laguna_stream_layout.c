/* Metadata-only tests. The optional GGUF probe never reads tensor payloads. */
#define DS4_NO_GPU
#define DS4_TEST_HOOKS
#include "../ds4.c"
#include <assert.h>

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
    for (uint32_t il = 0; il < 48; il++) {
        ds4_layer_weights *l = &w->layer[il];
        ds4_tensor *ts[] = {l->attn_q, l->attn_k, l->attn_v, l->attn_gate,
            l->attn_output, l->attn_norm, l->attn_q_norm, l->attn_k_norm, l->ffn_norm,
            l->ffn_gate, l->ffn_up, l->ffn_down,
            l->ffn_gate_inp, l->ffn_exp_probs_b, l->ffn_gate_exps, l->ffn_up_exps,
            l->ffn_down_exps, l->ffn_gate_shexp, l->ffn_up_shexp, l->ffn_down_shexp};
        for (unsigned i = 0; i < sizeof(ts) / sizeof(*ts); i++) {
            if (!ts[i]) continue;
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
    free(m.tensors);
    puts("laguna-stream-layout-host: OK (all layers, tables, spans, overflow, cache)");
    return 0;
}
