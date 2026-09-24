/* Host-only bounds and file-offset checks; no Metal initialization. */
#include "laguna_stream_q4_fixture.h"

static int valid(const laguna_stream_fixture *f, uint32_t budget, const int32_t *ids) {
    return ds4_stream_q4_selection_valid(
        f->size, f->desc.gate_offset, f->desc.up_offset, f->desc.down_offset,
        f->desc.gate_expert_bytes, f->desc.down_expert_bytes,
        FIX_TOTAL, DS4_STREAM_Q4_MAX_SELECTED, budget, ids);
}

int main(void) {
    laguna_stream_fixture f = fixture_open();
    int32_t ids[DS4_STREAM_Q4_MAX_SELECTED] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    assert(!valid(&f, 9, ids));
    assert(valid(&f, 10, ids));
    for (unsigned matrix = 0; matrix < 3; matrix++) {
        const uint64_t base = FIX_PREFIX + matrix * FIX_TOTAL * FIX_DIM * FIX_ROW;
        for (unsigned slot = 0; slot < DS4_STREAM_Q4_MAX_SELECTED; slot++) {
            for (unsigned row = 0; row < FIX_DIM; row += FIX_DIM - 1) {
                unsigned char block[FIX_ROW];
                const uint64_t offset = base + (uint64_t)ids[slot] * FIX_DIM * FIX_ROW + row * FIX_ROW;
                assert(pread(fileno(f.file), block, sizeof(block), (off_t)offset) == sizeof(block));
                const unsigned quant = fixture_quant(matrix, (unsigned)ids[slot]);
                for (unsigned i = 16; i < sizeof(block); i++) assert(block[i] == quant * 17u);
            }
        }
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
    puts("laguna-stream-q4-host: OK (top-10, ID 8/9, offsets, bounds, budget 9/10)");
    return 0;
}
