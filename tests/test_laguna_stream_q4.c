/* Host-only bounds and file-offset checks; no Metal initialization. */
#include "laguna_stream_q4_fixture.h"

static int valid(const laguna_stream_fixture *f, uint32_t budget, const int32_t *ids) {
    return ds4_stream_q4_selection_valid(
        f->size, f->desc.gate_offset, f->desc.up_offset, f->desc.down_offset,
        f->desc.gate_expert_bytes, f->desc.down_expert_bytes,
        FIX_TOTAL, DS4_STREAM_Q4_MAX_SELECTED, budget, ids);
}

static void run_case(bool rectangular) {
    laguna_stream_fixture f = fixture_open_case(rectangular);
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

int main(void) {
    run_case(false);
    run_case(true);
    puts("laguna-stream-q4-host: OK (square/rectangular, distinct rows, top-10, bounds, overflow, budget)");
    return 0;
}
