#ifndef LAGUNA_STREAM_Q4_FIXTURE_H
#define LAGUNA_STREAM_Q4_FIXTURE_H

#include "../ds4_gpu.h"
#include "../ds4_stream_q4.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum { FIX_DIM = 256, FIX_TOTAL = 12, FIX_ROW = 144, FIX_PREFIX = 16384 };

typedef struct {
    FILE *file;
    unsigned char *map;
    uint64_t size;
    ds4_gpu_laguna_moe_desc desc;
} laguna_stream_fixture;

static unsigned fixture_quant(unsigned matrix, unsigned expert) {
    return 1u + (expert * (2u * matrix + 1u)) % 15u;
}

static laguna_stream_fixture fixture_open(void) {
    const uint64_t expert_bytes = FIX_DIM * FIX_ROW;
    const uint64_t tensor_bytes = FIX_TOTAL * expert_bytes;
    laguna_stream_fixture f = {0};
    f.size = FIX_PREFIX + 3 * tensor_bytes;
    f.desc = (ds4_gpu_laguna_moe_desc) {
        .gate_offset = FIX_PREFIX,
        .up_offset = FIX_PREFIX + tensor_bytes,
        .down_offset = FIX_PREFIX + 2 * tensor_bytes,
        .gate_type = 12, .up_type = 12, .down_type = 12,
        .gate_expert_bytes = expert_bytes, .gate_row_bytes = FIX_ROW,
        .up_expert_bytes = expert_bytes, .up_row_bytes = FIX_ROW,
        .down_expert_bytes = expert_bytes, .down_row_bytes = FIX_ROW,
    };
    f.file = tmpfile();
    assert(f.file && ftruncate(fileno(f.file), (off_t)f.size) == 0);
    f.map = mmap(NULL, f.size, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(f.file), 0);
    assert(f.map != MAP_FAILED);
    memset(f.map, 0, f.size);
    for (unsigned matrix = 0; matrix < 3; matrix++) {
        for (unsigned expert = 0; expert < FIX_TOTAL; expert++) {
            for (unsigned row = 0; row < FIX_DIM; row++) {
                unsigned char *block = f.map + FIX_PREFIX + matrix * tensor_bytes +
                                      expert * expert_bytes + row * FIX_ROW;
                /* Constant Q4_K rows, scale=1, min=0; d=1/16 or 1/256. */
                const uint16_t d = matrix == 2 ? 0x1c00 : 0x2c00;
                memcpy(block, &d, sizeof(d));
                memset(block + 4, 1, 4);
                memset(block + 12, 1, 4);
                memset(block + 16, fixture_quant(matrix, expert) * 17u, 128);
            }
        }
    }
    assert(msync(f.map, f.size, MS_SYNC) == 0);
    return f;
}

static void fixture_close(laguna_stream_fixture *f) {
    assert(munmap(f->map, f->size) == 0);
    assert(fclose(f->file) == 0);
}

#endif
