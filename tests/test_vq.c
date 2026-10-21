/* The C decoder vs torch (tests/data/vq_golden.bin), and its speed. */
#include "../src/sm_internal.h"
#include "check.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { char name[64]; uint32_t nd, dim[4]; float *data; } arr;
static arr A[64]; static uint32_t nA;

static const arr *find(const char *name) {
    for (uint32_t i = 0; i < nA; i++) if (!strcmp(A[i].name, name)) return &A[i];
    fprintf(stderr, "missing %s\n", name); exit(2);
}
static float *kco(const char *name) {  /* torch conv weight [O][C][K] -> [K][C][O] */
    const arr *a = find(name); const uint32_t O = a->dim[0], C = a->dim[1], K = a->dim[2];
    float *w = malloc(sizeof(float) * O * C * K);
    for (uint32_t o = 0; o < O; o++) for (uint32_t c = 0; c < C; c++) for (uint32_t k = 0; k < K; k++)
        w[((size_t)k * C + c) * O + o] = a->data[((size_t)o * C + c) * K + k];
    return w;
}
static sm_conv cv(const char *n, float **keep) {
    char w[96], b[96]; snprintf(w, sizeof w, "%s.weight", n); snprintf(b, sizeof b, "%s.bias", n);
    *keep = kco(w); return (sm_conv){*keep, find(b)->data};
}

int main(int argc, char **argv) {
    FILE *f = fopen(argc > 1 ? argv[1] : "tests/data/vq_golden.bin", "rb"); CHECK(f != NULL); if (!f) return CHECK_DONE();
    CHECK(fread(&nA, 4, 1, f) == 1);
    for (uint32_t i = 0; i < nA; i++) {
        uint32_t len; size_t n = 1; arr *a = &A[i];
        CHECK(fread(&len, 4, 1, f) == 1 && len < 64); CHECK(fread(a->name, 1, len, f) == len); a->name[len] = 0;
        CHECK(fread(&a->nd, 4, 1, f) == 1 && a->nd <= 4); CHECK(fread(a->dim, 4, a->nd, f) == a->nd);
        for (uint32_t d = 0; d < a->nd; d++) n *= a->dim[d];
        a->data = malloc(sizeof(float) * n); CHECK(fread(a->data, sizeof(float), n, f) == n);
    }
    fclose(f);
    float *keep[12]; int nk = 0; sm_vq v = {0};
    v.codebook = find("codebooks")->data; v.mean = find("mean")->data; v.std = find("std")->data;
    v.in = cv("decoder.inp", &keep[nk++]);
    for (int r = 0; r < 3; r++) { char n[64];
        snprintf(n, sizeof n, "decoder.res.%d.conv1", r); v.res1[r] = cv(n, &keep[nk++]);
        snprintf(n, sizeof n, "decoder.res.%d.conv2", r); v.res2[r] = cv(n, &keep[nk++]); }
    v.up = cv("decoder.up", &keep[nk++]); v.mid = cv("decoder.mid", &keep[nk++]); v.out = cv("decoder.out", &keep[nk++]);
    double worst = 0, secs = 0; uint32_t frames = 0;
    for (int i = 0; i < 6; i++) {
        char n[32]; snprintf(n, sizeof n, "codes_%d", i); const arr *c = find(n); snprintf(n, sizeof n, "feat_%d", i); const arr *ref = find(n);
        const uint32_t L = c->dim[0]; int16_t *codes = malloc(sizeof(int16_t) * L * 6);
        for (uint32_t k = 0; k < L * 6; k++) codes[k] = (int16_t)c->data[k];
        float *feat = malloc(sizeof(float) * 2 * L * SM_F), *scr = malloc(sizeof(float) * sm_vq_scratch_floats(L));
        const clock_t t0 = clock(); sm_vq_decode(&v, codes, L, feat, scr); secs += (double)(clock() - t0) / CLOCKS_PER_SEC; frames += 2 * L;
        CHECK(ref->dim[0] == 2 * L);
        for (size_t k = 0; k < (size_t)2 * L * SM_F; k++) worst = fmax(worst, fabs(feat[k] - ref->data[k]) / (fabs(ref->data[k]) + 1));
        free(codes); free(feat); free(scr);
    }
    NEAR(worst, 0, 2e-5);
    printf("decoder: max rel diff %.2e, %.2f ms per 10 s clip (1 thread)\n", worst, secs / frames * 200 * 1000);
    for (int i = 0; i < nk; i++) free(keep[i]);
    for (uint32_t i = 0; i < nA; i++) free(A[i].data);
    return CHECK_DONE();
}
