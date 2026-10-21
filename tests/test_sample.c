/* The C sampler vs the reference sampler on the same logits and noise (tests/data/sample_golden.bin): the masked ids
   the model sees at every step and the final codes must be identical, ties included. Also the schedule against
   Python's numbers. */
#include "../src/sm_internal.h"
#include "check.h"
#include <stdlib.h>
#include <string.h>

typedef struct { const float *g; uint32_t L; } noise_t;
static float noise(void *u, uint32_t step, uint32_t pos, uint32_t code) {
    const noise_t *n = u; return n->g[((size_t)step * n->L + pos) * 513 + code];
}

static int rd(FILE *f, void *p, size_t bytes) { return fread(p, 1, bytes, f) == bytes; }

int main(void) {
    CHECK(sm_nmask(0, 10, 50) == 50); CHECK(sm_nmask(9, 10, 50) == 1);     /* cos(pi/2) * 50 = 3e-15 -> 1 */
    CHECK(sm_nmask(5, 10, 4) == 3);                                        /* cos(5/9 pi/2) * 4 = 2.57 -> 3 (Python agrees) */
    CHECK(sm_nmask(0, 1, 7) == 7);
    FILE *f = fopen("tests/data/sample_golden.bin", "rb"); CHECK(f != NULL); if (!f) return CHECK_DONE();
    uint32_t h[4]; int cases = 0, diff_steps = 0, diff_codes = 0;
    while (rd(f, h, sizeof h)) {
        const uint32_t L = h[0], T = h[1], R = h[2], nf = h[3];
        int32_t fx[4] = {0}; for (uint32_t i = 0; i < nf; i++) CHECK(rd(f, fx + 2 * i, 8));
        float *ml = malloc(sizeof(float) * T * L * 512), *rl = malloc(sizeof(float) * R * L * 513), *g = malloc(sizeof(float) * (T + R) * L * 513);
        int32_t *after = malloc(sizeof(int32_t) * (T - 1) * L), *want = malloc(sizeof(int32_t) * L * 6);
        CHECK(rd(f, ml, sizeof(float) * T * L * 512) && rd(f, rl, sizeof(float) * R * L * 513) && rd(f, g, sizeof(float) * (T + R) * L * 513));
        CHECK(rd(f, after, sizeof(int32_t) * (T - 1) * L) && rd(f, want, sizeof(int32_t) * L * 6));
        int32_t *ids = malloc(sizeof(int32_t) * L), *pick = malloc(sizeof(int32_t) * L), codes[100 * 6];
        float *score = malloc(sizeof(float) * L), *prob = malloc(sizeof(float) * L), tmp[513]; uint8_t *mask = malloc(L);
        noise_t nz = {g, L};
        for (uint32_t j = 0; j < L; j++) { ids[j] = 512; score[j] = 0; }
        for (uint32_t i = 0; i < nf; i++) { ids[fx[2 * i]] = fx[2 * i + 1]; score[fx[2 * i]] = 1e5f; }
        for (uint32_t i = 0; i < T; i++) {
            sm_lowest(score, L, sm_nmask(i, T, L - nf), mask);
            for (uint32_t j = 0; j < L; j++) if (mask[j]) ids[j] = 512;
            if (i && memcmp(ids, after + (size_t)(i - 1) * L, sizeof(int32_t) * L)) diff_steps++;
            sm_pick(ml + (size_t)i * L * 512, L, 512, mask, 1.0f, 52, noise, &nz, i, pick, prob, tmp);
            for (uint32_t j = 0; j < L; j++) if (mask[j]) { ids[j] = pick[j]; score[j] = prob[j]; } else score[j] = 1e5f;
        }
        for (uint32_t j = 0; j < L; j++) codes[j * 6] = ids[j];
        for (uint32_t q = 1; q <= R; q++) {
            sm_pick(rl + (size_t)(q - 1) * L * 513, L, 513, NULL, 1.0f, 52, noise, &nz, T + q - 1, pick, NULL, tmp);
            for (uint32_t j = 0; j < L; j++) codes[j * 6 + q] = pick[j] >= 512 ? -1 : pick[j];
        }
        for (uint32_t j = 0; j < L * 6; j++) diff_codes += codes[j] != want[j];
        free(ml); free(rl); free(g); free(after); free(want); free(ids); free(pick); free(score); free(prob); free(mask); cases++;
    }
    fclose(f);
    printf("sampler vs reference: %d cases, %d steps with different model inputs, %d different final codes\n", cases, diff_steps, diff_codes);
    CHECK(cases == 12); CHECK(diff_steps == 0); CHECK(diff_codes == 0);
    return CHECK_DONE();
}
