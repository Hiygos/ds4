#ifndef LAGUNA_STREAM_Q4_FIXTURE_H
#define LAGUNA_STREAM_Q4_FIXTURE_H

#include "../ds4_gpu.h"
#include "../ds4_stream_q4.h"
#include <assert.h>
#include <math.h>
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
    unsigned in_dim, mid_dim, out_dim;
    bool rectangular;
    ds4_gpu_laguna_moe_desc desc;
} laguna_stream_fixture;

static unsigned fixture_quant(unsigned matrix, unsigned expert) {
    return 1u + (expert * (2u * matrix + 1u)) % 15u;
}

static double fixture_scale(const laguna_stream_fixture *f, unsigned matrix, unsigned row) {
    return f->rectangular ? (row + 1.0) / (matrix == 2 ? 16384.0 : 4096.0) :
                            1.0 / (matrix == 2 ? 256.0 : 16.0);
}

static unsigned fixture_value(const laguna_stream_fixture *f, unsigned matrix,
                               unsigned expert, unsigned column) {
    return f->rectangular ? 1u + (expert * (2u * matrix + 1u) + column * (matrix + 3u)) % 15u :
                            fixture_quant(matrix, expert);
}

static laguna_stream_fixture fixture_open_case(bool rectangular) {
    laguna_stream_fixture f = {0};
    f.rectangular = rectangular;
    f.in_dim = f.out_dim = rectangular ? 512 : FIX_DIM;
    f.mid_dim = FIX_DIM;
    const uint64_t gate_row = f.in_dim / 256 * FIX_ROW;
    const uint64_t down_row = f.mid_dim / 256 * FIX_ROW;
    const uint64_t gate_bytes = f.mid_dim * gate_row;
    const uint64_t down_bytes = f.out_dim * down_row;
    f.size = FIX_PREFIX + FIX_TOTAL * (2 * gate_bytes + down_bytes);
    f.desc = (ds4_gpu_laguna_moe_desc) {
        .gate_offset = FIX_PREFIX,
        .up_offset = FIX_PREFIX + FIX_TOTAL * gate_bytes,
        .down_offset = FIX_PREFIX + 2 * FIX_TOTAL * gate_bytes,
        .gate_type = 12, .up_type = 12, .down_type = 12,
        .gate_expert_bytes = gate_bytes, .gate_row_bytes = gate_row,
        .up_expert_bytes = gate_bytes, .up_row_bytes = gate_row,
        .down_expert_bytes = down_bytes, .down_row_bytes = down_row,
    };
    f.file = tmpfile();
    assert(f.file && ftruncate(fileno(f.file), (off_t)f.size) == 0);
    f.map = mmap(NULL, f.size, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(f.file), 0);
    assert(f.map != MAP_FAILED);
    memset(f.map, 0, f.size);
    for (unsigned matrix = 0; matrix < 3; matrix++) {
        const unsigned rows = matrix == 2 ? f.out_dim : f.mid_dim;
        const unsigned columns = matrix == 2 ? f.mid_dim : f.in_dim;
        const uint64_t base = matrix == 0 ? f.desc.gate_offset :
                              matrix == 1 ? f.desc.up_offset : f.desc.down_offset;
        const uint64_t row_bytes = columns / 256 * FIX_ROW;
        for (unsigned expert = 0; expert < FIX_TOTAL; expert++) {
            for (unsigned row = 0; row < rows; row++) {
                for (unsigned b = 0; b < columns / 256; b++) {
                    unsigned char *block = f.map + base + expert * rows * row_bytes +
                                           row * row_bytes + b * FIX_ROW;
                    /* Exact normal binary16 scale, group scales=1, min=0. */
                    const float scale = (float)fixture_scale(&f, matrix, row);
                    uint32_t bits;
                    memcpy(&bits, &scale, sizeof(bits));
                    const uint16_t d = (uint16_t)((((bits >> 23) - 112u) << 10) |
                                                  ((bits >> 13) & 1023u));
                    memcpy(block, &d, sizeof(d));
                    memset(block + 4, 1, 4);
                    memset(block + 12, 1, 4);
                    for (unsigned q = 0; q < 128; q++) {
                        const unsigned c = b * 256 + (q / 32) * 64 + q % 32;
                        block[16 + q] = fixture_value(&f, matrix, expert, c) |
                                       (fixture_value(&f, matrix, expert, c + 32) << 4);
                    }
                }
            }
        }
    }
    assert(msync(f.map, f.size, MS_SYNC) == 0);
    return f;
}

/* Dense scalar oracle from the recipe, independent of file strides. */
static void fixture_reference(const laguna_stream_fixture *f, const int32_t ids[10],
                               const float weights[10], const float *x,
                               double *out, double *weighted_mid) {
    memset(out, 0, f->out_dim * sizeof(*out));
    for (unsigned slot = 0; slot < 10; slot++) {
        double mid[FIX_DIM];
        for (unsigned row = 0; row < f->mid_dim; row++) {
            double gate = 0, up = 0;
            for (unsigned c = 0; c < f->in_dim; c++) {
                gate += fixture_value(f, 0, ids[slot], c) * (double)x[c];
                up += fixture_value(f, 1, ids[slot], c) * (double)x[c];
            }
            gate *= fixture_scale(f, 0, row);
            up *= fixture_scale(f, 1, row);
            mid[row] = gate / (1 + exp(-gate)) * up;
            if (weighted_mid) weighted_mid[slot * f->mid_dim + row] = mid[row] * weights[slot];
        }
        for (unsigned row = 0; row < f->out_dim; row++) {
            double sum = 0;
            for (unsigned c = 0; c < f->mid_dim; c++)
                sum += fixture_value(f, 2, ids[slot], c) * mid[c];
            out[row] += weights[slot] * fixture_scale(f, 2, row) * sum;
        }
    }
}

static void fixture_close(laguna_stream_fixture *f) {
    assert(munmap(f->map, f->size) == 0);
    assert(fclose(f->file) == 0);
}

#endif
