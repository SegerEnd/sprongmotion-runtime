/* The matrix multiply on Q8_0 and F32 weights: bit-identical to the naive loop on every shape the model uses
   (and odd M), each epilogue, fp16 conversion exact, then the speed in GFLOP/s. */
#include "../src/sm_internal.h"
#include "check.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned rs = 7;
static float frand(void) { rs = rs * 1103515245u + 12345u; return ((float)((rs >> 8) & 0xFFFF) / 65535.0f) * 2 - 1; }

static uint8_t *make_q8(uint32_t N, uint32_t K) {
    uint8_t *w = malloc((size_t)N * (K / 32) * SM_Q8_BLOCK);
    for (size_t b = 0; b < (size_t)N * (K / 32); b++) {
        uint8_t *blk = w + b * SM_Q8_BLOCK; const uint16_t h = (uint16_t)(0x1C00 + (rs = rs * 1103515245u + 12345u) % 0x0800);  /* scales ~0.004..0.008 */
        blk[0] = (uint8_t)h; blk[1] = (uint8_t)(h >> 8);
        for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t)(int8_t)(frand() * 127);
    }
    return w;
}

static float weight(const sm_lin *l, uint32_t n, uint32_t k) {
    if (l->type == SM_T_F32) return ((const float *)l->w)[(size_t)n * l->k + k];
    const uint8_t *blk = (const uint8_t *)l->w + ((size_t)n * (l->k / 32) + k / 32) * SM_Q8_BLOCK;
    return sm_half((uint16_t)(blk[0] | blk[1] << 8)) * (float)(int8_t)blk[2 + k % 32];
}

static void naive(const sm_lin *l, const float *X, uint32_t M, float *Y, sm_epi epi, const float *add) {
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < l->n; n++) {
            float acc = 0;
            for (uint32_t k = 0; k < l->k; k++) acc += X[(size_t)m * l->k + k] * weight(l, n, k);
            const float y = l->b ? acc + l->b[n] : acc;
            Y[(size_t)m * l->n + n] = epi == SM_EPI_GELU ? sm_gelu(y) : epi == SM_EPI_ADD ? y + add[(size_t)m * l->n + n] : y;
        }
}

int main(void) {
    const uint16_t h[] = {0x3C00, 0xC000, 0x0001, 0x03FF, 0x7BFF, 0x7C00, 0x0000, 0x8000};   /* 1, -2, subnormals, max, inf, +-0 */
    const float f[] = {1.0f, -2.0f, 5.9604644775390625e-8f, 6.0975551605224609e-5f, 65504.0f, INFINITY, 0.0f, -0.0f};
    for (int i = 0; i < 8; i++) { const float g = sm_half(h[i]); CHECK(!memcmp(&g, &f[i], 4)); }
    int half_bad = 0;   /* every finite fp16: (-1)^s * (e ? 1024 + m : m) * 2^(max(e, 1) - 25) */
    for (uint32_t b = 0; b < 65536; b++) {
        const uint32_t e = b >> 10 & 31, m = b & 1023;
        if (e == 31) continue;
        const float want = ldexpf((float)(e ? 1024 + m : m), (int)(e ? e : 1) - 25), got = sm_half((uint16_t)b);
        const float sw = b >> 15 ? -want : want;
        half_bad += memcmp(&got, &sw, 4) != 0;
    }
    CHECK(half_bad == 0);
    const uint32_t shapes[][3] = {{1, 32, 16}, {5, 64, 32}, {141, 384, 1152}, {141, 384, 1536}, {141, 1536, 384}, {282, 384, 512}, {7, 768, 272}, {2, 64, 256}, {9, 512, 528}};
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        const uint32_t M = shapes[s][0], K = shapes[s][1], N = shapes[s][2];
        const size_t mn = (size_t)M * N;
        float *X = malloc(sizeof(float) * M * K), *b = malloc(sizeof(float) * N), *A = malloc(sizeof(float) * mn),
              *B = malloc(sizeof(float) * mn), *R = malloc(sizeof(float) * mn), *F = malloc(sizeof(float) * N * K);
        for (size_t i = 0; i < (size_t)M * K; i++) X[i] = frand();
        for (uint32_t i = 0; i < N; i++) b[i] = frand();
        for (size_t i = 0; i < mn; i++) R[i] = frand();
        for (size_t i = 0; i < (size_t)N * K; i++) F[i] = frand();
        uint8_t *W = make_q8(N, K);
        sm_lin ls[6] = {{W, b, SM_T_Q8_0, K, N, NULL}, {W, NULL, SM_T_Q8_0, K, N, NULL}, {F, b, SM_T_F32, K, N, NULL}};
        float *P = malloc(sizeof(float) * 3 * N * K);   /* the same three packed at load */
        for (size_t i = 0; i < 3; i++) ls[3 + i] = ls[i], sm_pack(&ls[3 + i], P + i * N * K);
        for (size_t i = 0; i < 6; i++)
            for (sm_epi e = SM_EPI_NONE; e <= SM_EPI_ADD; e++) {
                naive(&ls[i], X, M, A, e, R);
                sm_linear(NULL, &ls[i], X, M, B, e, R);
                CHECK(!memcmp(A, B, sizeof(float) * mn));
            }
        free(X); free(b); free(A); free(B); free(R); free(F); free(W); free(P);
    }
    const uint32_t M = 282, K = 384, N = 1536; int reps = 200;   /* speed: a CFG-stacked pass of the 14M FF layer */
    float *X = malloc(sizeof(float) * M * K), *Y = malloc(sizeof(float) * M * N); uint8_t *W = make_q8(N, K);
    for (size_t i = 0; i < (size_t)M * K; i++) X[i] = frand();
    sm_lin l = {W, NULL, SM_T_Q8_0, K, N, NULL};
    float *P = malloc(sizeof(float) * N * K);
    for (int packed = 0; packed < 2; packed++) {
        if (packed) sm_pack(&l, P);
        const clock_t t0 = clock(); for (int r = 0; r < reps; r++) sm_linear(NULL, &l, X, M, Y, SM_EPI_NONE, NULL);
        const double s = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("gemm q8%s %ux%ux%u: %.1f GFLOP/s (1 thread)\n", packed ? " packed" : "", M, K, N,
               2.0 * M * K * N * reps / s / 1e9);
    }
    free(X); free(Y); free(W); free(P);
    return CHECK_DONE();
}
