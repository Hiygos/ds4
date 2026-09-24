/* Host checks of bounds and offsets, without Metal initialization. */
#include "laguna_stream_q4_fixture.h"

static int valid(const laguna_stream_fixture *f, uint32_t budget, const int32_t *ids) {
    return ds4_stream_q4_selection_valid(
        f->size, f->desc.gate_offset, f->desc.up_offset, f->desc.down_offset,
        f->desc.gate_expert_bytes, f->desc.down_expert_bytes,
        FIX_TOTAL, DS4_STREAM_Q4_MAX_SELECTED, budget, ids);
}

static void run_case(bool rectangular) {
    laguna_stream_fixture f = fixture_open_case(rectangular);
    /* These sizes make the cross-row comparisons non-vacuous too. */
    assert(f.in_dim == (rectangular ? 512u : 256u));
    assert(f.mid_dim == 256u && f.out_dim == f.in_dim);
    int32_t ids[DS4_STREAM_Q4_MAX_SELECTED] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    assert(!valid(&f, 9, ids));
    assert(valid(&f, 10, ids));
    for (unsigned matrix = 0; matrix < 3; matrix++) {
        const uint64_t base = matrix == 0 ? f.desc.gate_offset :
                              matrix == 1 ? f.desc.up_offset : f.desc.down_offset;
        const unsigned rows = matrix == 2 ? f.out_dim : f.mid_dim;
        const unsigned columns = matrix == 2 ? f.mid_dim : f.in_dim;
        const uint64_t row_bytes = columns / 256 * FIX_ROW;
        for (unsigned slot = 0; slot < DS4_STREAM_Q4_MAX_SELECTED; slot++) {
            for (unsigned row = 0; row < rows; row++) {
              for (unsigned b = 0; b < columns / 256; b++) {
                unsigned char block[FIX_ROW];
                const uint64_t offset = base + (uint64_t)ids[slot] * rows * row_bytes +
                                        row * row_bytes + b * FIX_ROW;
                assert(pread(fileno(f.file), block, sizeof(block), (off_t)offset) == sizeof(block));
                uint16_t d;
                memcpy(&d, block, sizeof(d));
                const double scale = ldexp(1.0 + (d & 1023) / 1024.0, (d >> 10) - 15);
                assert(scale == fixture_scale(&f, matrix, row));
                for (unsigned c = 0; c < 256; c++) {
                    const unsigned packed = block[16 + (c / 64) * 32 + c % 32];
                    const unsigned q = (packed >> ((c % 64 >= 32) ? 4 : 0)) & 15;
                    assert(q == fixture_value(&f, matrix, ids[slot], b * 256 + c));
                }
              }
            }
        }
    }
    float x[512], weights[10];
    double expected[512];
    for (unsigned c = 0; c < f.in_dim; c++) x[c] = (1.0f + c % 17) / (9 * f.in_dim);
    for (unsigned s = 0; s < 10; s++) weights[s] = (s + 1) / 55.0f;
    double mid[10 * FIX_DIM];
    fixture_reference(&f, ids, weights, x, expected, mid);
    for (unsigned r = 0; r < f.out_dim; r++) assert(isfinite(expected[r]) && expected[r] > 0);
    if (rectangular) {
        assert(f.desc.gate_row_bytes != f.desc.down_row_bytes);
        for (unsigned r = 1; r < f.out_dim; r++) assert(expected[r] > expected[r - 1]);
        for (unsigned s = 0; s < 10; s++)
            for (unsigned r = 1; r < f.mid_dim; r++)
                assert(mid[s * f.mid_dim + r] > mid[s * f.mid_dim + r - 1]);
    }
    ids[0] = FIX_TOTAL;
    assert(!valid(&f, 10, ids));
    ids[0] = -1;
    assert(!valid(&f, 10, ids));
    ids[0] = ids[1];
    assert(!valid(&f, 10, ids));
    ids[0] = 9;
    f.size--;
    assert(!valid(&f, 10, ids));
    f.size++;
    const uint64_t bytes = f.desc.gate_expert_bytes;
    f.desc.gate_expert_bytes = UINT64_MAX;
    assert(!valid(&f, 10, ids));
    f.desc.gate_expert_bytes = bytes;
    f.desc.down_offset = UINT64_MAX;
    assert(!valid(&f, 10, ids));
    fixture_close(&f);
}

static void test_prefill_union_and_admission(void) {
    int32_t ids[4 * DS4_STREAM_Q4_MAX_SELECTED];
    for (uint32_t row = 0; row < 4; row++)
        for (uint32_t col = 0; col < DS4_STREAM_Q4_MAX_SELECTED; col++)
            ids[row * DS4_STREAM_Q4_MAX_SELECTED + col] =
                (int32_t)(col + (row % 2u) * 5u);
    int32_t unique[256] = {0};
    uint32_t count = 0;
    assert(ds4_laguna_stream_prefill_union(ids, 4,
            DS4_STREAM_Q4_MAX_SELECTED, 256, unique, 256, &count));
    assert(count == 15);
    for (uint32_t i = 0; i < count; i++) assert(unique[i] == (int32_t)i);

    ids[17] = 256;
    assert(!ds4_laguna_stream_prefill_union(ids, 4,
            DS4_STREAM_Q4_MAX_SELECTED, 256, unique, 256, &count));
    ids[17] = 12;
    assert(!ds4_laguna_stream_prefill_union(ids, 0,
            DS4_STREAM_Q4_MAX_SELECTED, 256, unique, 256, &count));
    assert(!ds4_laguna_stream_prefill_union(ids, 4, 8,
            256, unique, 256, &count));

    assert(ds4_laguna_stream_prefill_required_cache(1) == 10);
    assert(ds4_laguna_stream_prefill_required_cache(2) == 30);
    assert(ds4_laguna_stream_prefill_required_cache(4) == 50);
    assert(ds4_laguna_stream_prefill_required_cache(8) == 90);
    assert(ds4_laguna_stream_prefill_required_cache(32) == 330);
    assert(ds4_laguna_stream_prefill_required_cache(0) == 0);
    assert(ds4_laguna_stream_prefill_required_cache(33) == 0);
    assert(ds4_laguna_stream_prefill_cache_admitted(1, 10));
    assert(!ds4_laguna_stream_prefill_cache_admitted(2, 29));
    assert(ds4_laguna_stream_prefill_cache_admitted(2, 30));
    assert(!ds4_laguna_stream_prefill_cache_admitted(8, 89));
    assert(ds4_laguna_stream_prefill_cache_admitted(8, 90));
}

int main(void) {
    run_case(false);
    run_case(true);
    test_prefill_union_and_admission();
    puts("laguna-stream-q4-host: OK (square/rectangular, top-10, union/dedup, admission)");
    return 0;
}
