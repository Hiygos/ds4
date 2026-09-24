/* Test on the real GGUF; needs Metal and the model file. It includes the
 * backend to inject EIO and check engine teardown. */
#define DS4_TEST_HOOKS
#include "../ds4_metal.m"
#include "../ds4.h"
#include <assert.h>
#include <math.h>
#ifdef NDEBUG
#error "Laguna tests require assertions"
#endif

static unsigned fault_calls, emitted;
extern int ds4_test_laguna_stream_batch(ds4_engine *, const ds4_tokens *,
                                       int, float *, int);
static ssize_t engine_fault_pread(int fd, void *dst, size_t len, off_t off) {
    unsigned n = __atomic_add_fetch(&fault_calls, 1, __ATOMIC_RELAXED);
    if (n == 1) { errno = EIO; return -1; }
    return pread(fd, dst, len, off);
}

static void count_token(void *ud, int token) {
    int *first = ud;
    assert(token >= 0);
    if (!emitted) *first = token;
    emitted++;
}

static void check_closed(int old_fd) {
    assert(old_fd >= 0 && fcntl(old_fd, F_GETFD) == -1 && errno == EBADF);
    assert(g_model_fd == -1 && !g_model_map_ptr && !g_model_view_count);
    assert(!g_stream_expert_pending_load.active && !g_stream_expert_pending_load.model_map);
    assert(!g_stream_expert_pending_load.n_tasks && !g_stream_expert_pread_pool_tasks);
    assert(!g_stream_expert_pread_pool_remaining_workers && !g_stream_expert_pread_pool_thread_count);
    assert(!g_stream_expert_cache_entry_count && !g_stream_expert_cache_expert_bytes);
    assert(!g_stream_expert_cache_slab_count && !g_stream_expert_cache_slab_total_slots);
    assert(!g_stream_expert_cache_free_slot_count && !g_laguna_stream_allocated_bytes);
    assert(!g_laguna_stream_buffers && !g_laguna_stream_free_buffers);
    assert(!g_laguna_stream_moe_calls && !g_laguna_stream_token_rows);
    assert(!g_test_laguna_stream_temporary_bytes);
    for (unsigned il = 0; il < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; il++)
        for (unsigned id = 0; id < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; id++) {
            ds4_gpu_stream_expert_cache_entry *e = &g_stream_expert_cache[il][id];
            assert(!e->valid && !e->model_map && !e->gate_buffer && !e->up_buffer && !e->down_buffer);
        }
}

static void copy_logits(ds4_session *s, float *dst, int vocab) {
    assert(ds4_session_copy_logits(s, dst, vocab) == vocab);
    for (int i = 0; i < vocab; i++) assert(isfinite(dst[i]));
}

static void compare_logits(ds4_session *s, const float *expected, int vocab) {
    float *actual = malloc((size_t)vocab * sizeof(float));
    assert(actual && vocab > 0);
    copy_logits(s, actual, vocab);
    for (int i = 0; i < vocab; i++)
        assert(fabsf(actual[i] - expected[i]) <= 1e-5f * (1 + fabsf(expected[i])));
    free(actual);
}

static void check_resume(ds4_engine *engine, ds4_session **session, int ctx,
                         int vocab, bool long_context) {
    char err[256] = {0};
    ds4_session *s = *session;
    if (long_context) {
        /* 500-token prompt, then decode past the 512-row SWA window. */
        assert(ds4_session_pos(s) == 500);
        for (unsigned i = 0; i < 24; i++) {
            int token = ds4_session_argmax(s);
            assert(token >= 0 && ds4_session_eval(s, token, err, sizeof(err)) == 0);
        }
        assert(ds4_session_pos(s) == 524);
    }
    const int saved_pos = ds4_session_pos(s);
    ds4_tokens prefix = {0};
    ds4_tokens_copy(&prefix, ds4_session_tokens(s));
    assert(prefix.len == saved_pos && saved_pos > 1);
    FILE *file = tmpfile();
    assert(file && ds4_session_save_payload(s, file, err, sizeof(err)) == 0);
    assert(fflush(file) == 0);
    off_t bytes = ftello(file);
    assert(bytes > 0 && (uint64_t)bytes == ds4_session_payload_bytes(s));
    float *saved = malloc((size_t)vocab * sizeof(float));
    float *future = malloc((size_t)vocab * sizeof(float));
    assert(saved && future);
    copy_logits(s, saved, vocab);
    int tokens[4];
    for (unsigned i = 0; i < 4; i++) {
        tokens[i] = ds4_session_argmax(s);
        assert(tokens[i] >= 0 && ds4_session_eval(s, tokens[i], err, sizeof(err)) == 0);
    }
    copy_logits(s, future, vocab);
    ds4_session_free(s);
    s = NULL;
    assert(ds4_session_create(&s, engine, ctx) == 0 && s);
    rewind(file);
    assert(ds4_session_load_payload(s, file, (uint64_t)bytes, err, sizeof(err)) == 0);
    assert(ds4_session_pos(s) == saved_pos && ds4_session_common_prefix(s, &prefix) == saved_pos);
    compare_logits(s, saved, vocab);
    for (unsigned i = 0; i < 4; i++) {
        assert(ds4_session_argmax(s) == tokens[i]);
        assert(ds4_session_eval(s, tokens[i], err, sizeof(err)) == 0);
    }
    compare_logits(s, future, vocab);
    assert(ds4_session_pos(s) == saved_pos + 4);
    printf("laguna-stream-resume: OK ctx=%d saved=%d final=%d bytes=%lld\n",
           ctx, saved_pos, ds4_session_pos(s), (long long)bytes);
    free(saved); free(future); fclose(file); ds4_tokens_free(&prefix);
    *session = s;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && strcmp(argv[2], "--long-context"))) {
        fprintf(stderr, "usage: %s MODEL.gguf [--long-context]\n", argv[0]);
        return 1;
    }
    const bool long_context = argc == 3;
    const int ctx = long_context ? 1024 : 256;
    assert(setenv("DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY", "1", 1) == 0);
    assert(unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_TIMING_SUMMARY") == 0);
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = 8ull << 30,
        .context_size = ctx};
    float *reference = NULL;
    int reference_vocab = 0;
    for (unsigned cycle = 0; cycle < 3; cycle++) {
        ds4_engine *engine = NULL;
        assert(ds4_engine_open(&engine, &opt) == 0 && engine);
        assert(ds4_engine_is_laguna(engine) && ds4_engine_layer_count(engine) == 48);
        assert(g_model_fd >= 0 && g_model_map_ptr && g_model_view_count > 0);
        assert(!g_stream_expert_cache_entry_count && !g_laguna_stream_moe_calls);
        int model_fd = g_model_fd;
        int vocab = ds4_engine_vocab_size(engine);
        assert(vocab > 0);
        ds4_context_memory mem = ds4_context_memory_estimate_with_prefill_mode(
            DS4_BACKEND_METAL, ctx, 0, true);
        const uint64_t expected_kv = (12ull * ctx + 36ull * (ctx < 512 ? ctx : 512)) * 4096;
        assert(mem.prefill_cap == 1 && mem.raw_bytes == expected_kv && mem.scratch_bytes > 0);
        assert(mem.raw_cap == (uint32_t)ctx && mem.comp_cap == (uint32_t)(ctx < 512 ? ctx : 512));
        assert(mem.total_bytes == mem.raw_bytes + mem.scratch_bytes);
        printf("laguna-stream-memory: ctx=%d KV=%llu scratch=%llu prefill=1\n",
               ctx, (unsigned long long)mem.raw_bytes, (unsigned long long)mem.scratch_bytes);
        ds4_tokens prompt = {0};
        ds4_chat_begin(engine, &prompt);
        ds4_tokenize_text(engine, "The capital of Italy is", &prompt);
        assert(prompt.len > 1 && prompt.len < 64);
        char err[256] = {0};
        ds4_session *s = NULL;
        assert(ds4_session_create(&s, engine, ctx) == 0 && s);
        /* The public limit applies to the prompt; the GPU scratch stays at one
         * row. */
        assert(ds4_session_prefill_cap(s) == ctx && ds4_session_ctx(s) == ctx);
        if (cycle == 1) {
            fault_calls = 0; g_test_laguna_stream_pread = engine_fault_pread;
            assert(ds4_session_sync(s, &prompt, err, sizeof(err)) != 0);
            g_test_laguna_stream_pread = NULL;
            assert(fault_calls > 0 && err[0] && g_laguna_stream_failures == 1);
            assert(!g_stream_expert_cache_entry_count && !g_stream_expert_pending_load.active);
        } else {
            uint64_t calls = g_laguna_stream_moe_calls, rows = g_laguna_stream_token_rows;
            assert(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
            assert(ds4_session_pos(s) == prompt.len);
            assert(g_laguna_stream_moe_calls - calls == (uint64_t)prompt.len * 47);
            assert(g_laguna_stream_token_rows - rows == (uint64_t)prompt.len);
            assert(!g_stream_expert_cache_decode_tokens);
            if (cycle == 0) {
                reference_vocab = vocab;
                reference = malloc((size_t)vocab * sizeof(float)); assert(reference);
                copy_logits(s, reference, vocab);
                float *batch = malloc((size_t)vocab * sizeof(float)); assert(batch);
                rows = g_laguna_stream_token_rows;
                calls = g_laguna_stream_moe_calls;
                assert(ds4_test_laguna_stream_batch(engine, &prompt, ctx, batch, vocab));
                assert(g_laguna_stream_token_rows - rows == (uint64_t)prompt.len);
                assert(g_laguna_stream_moe_calls - calls == (uint64_t)prompt.len * 47);
                for (int i = 0; i < vocab; i++) {
                    assert(isfinite(batch[i]));
                    assert(fabsf(batch[i] - reference[i]) <= 1e-5f * (1 + fabsf(reference[i])));
                }
                free(batch);
                if (long_context) {
                    ds4_tokens sentence = {0};
                    ds4_tokenize_text(engine, " This is a test of the sliding attention window.", &sentence);
                    assert(sentence.len > 0);
                    for (unsigned i = 0; prompt.len < 500; i++)
                        ds4_tokens_push(&prompt, sentence.v[i % sentence.len]);
                    ds4_tokens_free(&sentence);
                    assert(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
                    assert(ds4_session_pos(s) == 500);
                }
                check_resume(engine, &s, ctx, vocab, long_context);
            } else {
                assert(reference_vocab == vocab && reference);
                compare_logits(s, reference, vocab);
                /* Single session row, then extend the same prefix. */
                ds4_tokens one = {.v = prompt.v, .len = 1};
                ds4_session_invalidate(s);
                rows = g_laguna_stream_token_rows;
                assert(ds4_session_sync(s, &one, err, sizeof(err)) == 0);
                assert(ds4_session_pos(s) == 1 && g_laguna_stream_token_rows == rows + 1);
                assert(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
                assert(g_laguna_stream_token_rows == rows + (uint64_t)prompt.len);
                compare_logits(s, reference, vocab);
                /* generate_argmax entry: a one-row batch with scratch=1. */
                emitted = 0; int first = -1;
                rows = g_laguna_stream_token_rows;
                assert(ds4_engine_generate_argmax(engine, &prompt, 1, ctx,
                    count_token, NULL, &first, NULL, NULL) == 0);
                assert(emitted == 1 && first >= 0 && g_laguna_stream_token_rows == rows + (uint64_t)prompt.len);
            }
        }
        ds4_session_free(s); ds4_tokens_free(&prompt);
        ds4_engine_close(engine); check_closed(model_fd);
        printf("laguna-stream-engine: cycle %u OK (%s)\n", cycle,
               cycle == 1 ? "injected EIO then close" : "open, run, close");
    }
    free(reference);
    puts("laguna-stream-engine: OK (prefill, row, resume, close/open, reopen after EIO)");
    return 0;
}
