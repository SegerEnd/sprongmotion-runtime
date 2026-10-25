/* The model file: load, tokenizer and decoder vs the reference, damaged files, allocations, misuse. */
#define _POSIX_C_SOURCE 200809L   /* posix_memalign */
#include "../src/sm_internal.h"
#include "check.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint8_t *read_all(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n); if (fread(b, 1, *n, f) != *n) { free(b); b = NULL; }
    fclose(f); return b;
}

/* counting allocator: every alloc must be freed */
typedef struct { long live, total; } counter;
static void *c_alloc(void *u, size_t size, size_t align) { counter *c = u; void *p = NULL; if (posix_memalign(&p, align, size ? size : 1)) return NULL; c->live++; c->total++; return p; }
static void  c_free(void *u, void *p) { counter *c = u; c->live--; free(p); }

static void test_tokenizer(const sm_model *m, const char *path) {
    size_t n; uint8_t *b = read_all(path, &n); CHECK(b != NULL); if (!b) return;
    sm_tok_work *w = malloc(sizeof *w); uint32_t count; memcpy(&count, b, 4); size_t at = 4; uint32_t bad = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t len; memcpy(&len, b + at, 4); const char *text = (const char *)b + at + 4; int32_t want[40], got[40];
        memcpy(want, b + at + 4 + len, sizeof want); at += 4 + len + sizeof want;
        CHECK(sm_tok_encode(&m->tok, text, len, w, got) >= 2);
        if (memcmp(want, got, sizeof want)) { if (bad++ < 5) fprintf(stderr, "tokenizer differs on \"%.*s\"\n", (int)len, text); }
    }
    printf("tokenizer: %u texts, %u differ from the reference\n", count, bad);
    CHECK(bad == 0);
    int32_t ids[40]; CHECK(sm_tok_encode(&m->tok, "\xC3\x28", 2, w, ids) == -1);   /* invalid UTF-8 */
    free(w); free(b);
}

/* the codes_i and feat_i arrays of vq_golden.bin, through the loaded file; the time on one thread */
static void test_decoder(const sm_model *m, const char *path) {
    CHECK(load_goldens(path));
    double worst = 0, secs = 0;
    uint32_t frames = 0;
    for (int i = 0; i < 6; i++) {
        char name[32];
        snprintf(name, sizeof name, "codes_%d", i);
        const golden *c = find_golden(name);
        snprintf(name, sizeof name, "feat_%d", i);
        const golden *want = find_golden(name);
        const uint32_t L = c->dim[0];
        int16_t *codes = malloc(sizeof(int16_t) * L * 6);
        float *feat = malloc(sizeof(float) * 2 * L * SM_F), *scratch = malloc(sizeof(float) * sm_vq_scratch_floats(L));
        for (uint32_t k = 0; k < L * 6; k++) codes[k] = (int16_t)c->data[k];
        const clock_t t0 = clock();
        sm_vq_decode(&m->vq, codes, L, feat, scratch);
        secs += (double)(clock() - t0) / CLOCKS_PER_SEC;
        frames += 2 * L;
        CHECK(want->dim[0] == 2 * L);
        for (size_t k = 0; k < (size_t)2 * L * SM_F; k++)
            worst = fmax(worst, fabs(feat[k] - want->data[k]) / (fabs(want->data[k]) + 1));
        free(codes);
        free(feat);
        free(scratch);
    }
    printf("decoder: max rel diff %.2e, %.2f ms per 10 s clip (1 thread)\n", worst, secs / frames * 200 * 1000);
    NEAR(worst, 0, 2e-5);
    free_goldens();
}

static void test_corrupt(const uint8_t *file, size_t n) {  /* any damage: an error status or a valid model, never a crash */
    uint8_t *b = malloc(n); unsigned s = 12345; int ok = 0, bad = 0; char err[128];
    counter c = {0, 0}; sm_hooks h = SM_HOOKS_INIT; h.alloc = c_alloc; h.free = c_free; h.user = &c;
    for (int i = 0; i < 3000; i++) {
        memcpy(b, file, n); size_t len = n;
        s = s * 1103515245u + 12345u;
        if (i % 3 == 0) len = (s >> 4) % (i < 1500 ? 70000 : n);                    /* truncated */
        else for (int k = 0; k < 1 + i % 4; k++) { s = s * 1103515245u + 12345u; b[(s >> 4) % 70000] ^= (uint8_t)(1u << (s % 8)); }  /* header bits */
        sm_model *m = NULL;
        const sm_status st = sm_model_load_memory(b, len, SM_LOAD_BORROW | SM_LOAD_NOCHECK, &h, &m, err, sizeof err);
        if (st == SM_OK) { ok++; CHECK(m != NULL); sm_model_free(m); } else { bad++; CHECK(m == NULL && err[0]); }
    }
    memcpy(b, file, n); b[n - 100] ^= 1;                                               /* a data bit: the CRC finds it */
    sm_model *m = NULL; CHECK(sm_model_load_memory(b, n, SM_LOAD_BORROW, &h, &m, err, sizeof err) == SM_E_CHECKSUM && !m);
    printf("corrupt files: %d rejected, %d still valid (flips in unused bytes), %ld allocations, %ld not freed\n", bad, ok, c.total, c.live);
    CHECK(c.live == 0); free(b);
}

static void test_api(const uint8_t *file, size_t n) {
    char err[128]; sm_model *m = (sm_model *)1;
    CHECK(sm_model_load_memory(NULL, n, 0, NULL, &m, err, sizeof err) == SM_E_ARG);
    sm_hooks h = {0}; CHECK(sm_model_load_memory(file, n, 0, &h, &m, err, sizeof err) == SM_E_ARG);   /* struct_size 0 */
    CHECK(sm_model_load_file("/nonexistent/x.gguf", NULL, &m, err, sizeof err) == SM_E_IO);
    CHECK(sm_model_load_memory(file, n, 0, NULL, &m, NULL, 0) == SM_OK);                                /* err is optional */
    CHECK(!strcmp(sm_model_meta(m, "general.architecture"), "sprongmotion"));
    CHECK(!strncmp(sm_model_id(m), "SprongMotion-14M@", 17));
    CHECK(sm_model_info(m, SM_I_FPS) == 20 && sm_model_info(m, SM_I_JOINTS) == 22 && sm_model_info(m, 999) == -1);
    CHECK(!strcmp(sm_status_name(SM_E_CHECKSUM), "SM_E_CHECKSUM") && !strcmp(sm_status_name(-1234), "SM_E_UNKNOWN"));
    sm_model_free(m); sm_model_free(NULL);
}

/* the bytes right after a GGUF key or tensor name (8-byte length, then the text), or NULL */
static uint8_t *after(uint8_t *b, size_t n, const char *name) {
    const size_t len = strlen(name);
    for (size_t i = 0; i + 8 + len < n; i++)
        if (b[i] == len && !memcmp(b + i + 1, "\0\0\0\0\0\0\0", 7) && !memcmp(b + i + 8, name, len)) return b + i + 8 + len;
    return NULL;
}

/* sprongmotion.defaults.* set the defaults within the runtime's limits; format 2 only; the skeleton must be there */
static void test_settings(const uint8_t *file, size_t n) {
    uint8_t *b = malloc(n); sm_model *m = NULL; double lo, hi, d; char err[128];
    memcpy(b, file, n); uint8_t *v = after(b, n, "sprongmotion.defaults.steps"); CHECK(v != NULL); if (!v) { free(b); return; }
    const uint32_t steps[3] = {12, 99, 0};   /* in range, above the limit 64, below the minimum 1 */
    for (int i = 0; i < 3; i++) {
        memcpy(v + 4, &steps[i], 4);
        const sm_status s = sm_model_load_memory(b, n, SM_LOAD_BORROW, NULL, &m, err, sizeof err);
        CHECK(i ? s == SM_E_FORMAT && !m : s == SM_OK && sm_param_bounds(m, SM_P_STEPS, &lo, &hi, &d, NULL) == SM_OK);
        if (!i) CHECK(d == 12 && lo == 1 && hi == 64 && strstr(sm_model_schema_json(m), "\"default\":12}"));
        sm_model_free(m); m = NULL;
    }
    memcpy(b, file, n); v = after(b, n, "sprongmotion.format_version"); CHECK(v != NULL);
    if (v) { const uint32_t one = 1; memcpy(v + 4, &one, 4); }   /* format 1: not loaded */
    CHECK(sm_model_load_memory(b, n, SM_LOAD_BORROW, NULL, &m, err, sizeof err) == SM_E_VERSION && !m);
    memcpy(b, file, n); v = after(b, n, "skeleton.offsets"); CHECK(v != NULL); if (v) v[-1] = 'S';
    CHECK(sm_model_load_memory(b, n, SM_LOAD_BORROW, NULL, &m, err, sizeof err) == SM_E_FORMAT);
    CHECK(strstr(err, "skeleton") != NULL);
    float off[66]; CHECK(sm_model_load_memory(file, n, 0, NULL, &m, NULL, 0) == SM_OK);
    CHECK(sm_skeleton_offsets(m, off, 66) == SM_OK);
    CHECK(off[0] == 0 && off[3 * 4 + 1] < -0.3f && off[3 * 4 + 1] > -0.5f);   /* the knee below the hip */
    sm_model_free(m); free(b);
}

int main(void) {
    size_t n; uint8_t *file = read_all("tests/data/test.gguf", &n); CHECK(file != NULL); if (!file) return CHECK_DONE();
    counter c = {0, 0}; sm_hooks h = SM_HOOKS_INIT; h.alloc = c_alloc; h.free = c_free; h.user = &c; char err[128];
    sm_model *m = NULL; const sm_status s = sm_model_load_file("tests/data/test.gguf", &h, &m, err, sizeof err);
    if (s != SM_OK) fprintf(stderr, "load: %s\n", err);
    CHECK(s == SM_OK);
    if (m) { test_tokenizer(m, "tests/data/tok_golden.bin"); test_decoder(m, "tests/data/vq_golden.bin"); sm_model_free(m); }
    CHECK(c.live == 0);
    test_corrupt(file, n); test_api(file, n); test_settings(file, n);
    free(file); return CHECK_DONE();
}
