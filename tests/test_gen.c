/* The text encoder and generator vs torch; needs $SM_GOLDEN_F32 and $SM_GOLDEN_Q8, else skipped. */
#include "../src/sm_internal.h"
#include "check.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const uint32_t LENS[4] = {1, 17, 49, 100};
#define N_PROMPTS 10

static const golden *find(const char *fmt, int i, int j) {
    char name[64];
    snprintf(name, sizeof name, fmt, i, j);
    return find_golden(name);
}

static sm_model *load_model(const char *var, uint32_t threads) {
    const char *path = getenv(var); char err[128]; sm_model *m = NULL;
    sm_hooks h = SM_HOOKS_INIT; h.n_threads = threads;
    if (!path) return NULL;
    if (sm_model_load_file(path, &h, &m, err, sizeof err) == SM_OK) return m;
    fprintf(stderr, "%s: %s\n", path, err);
    return NULL;
}

static int32_t *ids_of(int p, uint32_t *W) {
    const golden *a = find("p%d.ids", p, 0); int32_t *ids = malloc(sizeof(int32_t) * a->n);
    for (size_t i = 0; i < a->n; i++) ids[i] = (int32_t)a->data[i];
    *W = (uint32_t)a->n; return ids;
}

static sm_status gen_codes(const sm_gen *g, const float *vec, const float *words, uint32_t W, uint32_t L,
                           const sm_sampling *s, uint64_t seed, int16_t *codes, float *scratch) {
    sm_run r = {W, L, 0, 0, *s, seed, codes, NULL, NULL, 0};
    sm_status st = sm_gen_start(g, &r, vec, words, scratch);
    while (st == SM_OK) st = sm_gen_pass(g, &r, scratch);
    return st == SM_DONE ? SM_OK : st;
}

static sm_sampling sampling(void) {
    const float *s = find("sampling", 0, 0)->data;
    return (sm_sampling){(uint32_t)s[0], (uint32_t)s[1], (uint32_t)s[5], s[2], s[3], s[4]};
}

/* max over rows of max |got - want| / std(want row); got has row stride gs, the combined rows come from raw */
static double row_err(const float *raw, uint32_t L, uint32_t gs, uint32_t n, float cfg, const float *want) {
    double worst = 0;
    for (uint32_t part = 0; part < 3; part++)
        for (uint32_t j = 0; j < L; j++) {
            const float *w = want + ((size_t)part * L + j) * n; double mean = 0, var = 0, mx = 0;
            for (uint32_t c = 0; c < n; c++) mean += w[c] / (double)n;
            for (uint32_t c = 0; c < n; c++) var += (w[c] - mean) * (w[c] - mean) / (double)n;
            for (uint32_t c = 0; c < n; c++) {
                const float cv = raw[(size_t)j * gs + c], uv = raw[(size_t)(L + j) * gs + c];
                const float got = part == 0 ? cv : part == 1 ? uv : uv + (cv - uv) * cfg;
                mx = fmax(mx, fabs(got - w[c]));
            }
            worst = fmax(worst, mx / sqrt(var));
        }
    return worst;
}

static void hist_of(const sm_gen *g, const float *codes, uint32_t L, uint32_t q, float *hist, float *row) {
    memset(hist, 0, sizeof(float) * L * g->d);
    for (uint32_t l = 1; l <= q; l++)
        for (uint32_t j = 0; j < L; j++) {
            sm_row(&g->in[l], (uint32_t)codes[j * 6 + l - 1], row);
            for (uint32_t i = 0; i < g->d; i++) hist[(size_t)j * g->d + i] += row[i];
        }
}

static void test_noise(void) {
    const golden *a = find("noise", 0, 0); const sm_sampling s = sampling(); size_t exact = 0; double worst = 0;
    for (uint32_t t = 0; t < a->dim[0]; t++)
        for (uint32_t j = 0; j < a->dim[1]; j++)
            for (uint32_t c = 0; c < a->dim[2]; c++) {
                const float want = a->data[((size_t)t * a->dim[1] + j) * a->dim[2] + c];
                const float got = t < s.steps ? sm_hash_gumbel(7, 0, t, j, c) : sm_hash_gumbel(7, 1, t - s.steps, j, c);
                exact += got == want; worst = fmax(worst, fabs(got - want));
            }
    printf("hash noise: %zu of %zu identical, max diff %.2e\n", exact, a->n, worst);
    NEAR(worst, 0, 1e-6);
}

/* per pass: text encoder, base pass (all masked) and residual passes 1 and 5, all with torch's inputs */
static void test_passes(const sm_model *m) {
    const sm_gen *g = &m->gen; const sm_sampling s = sampling();
    float *ts = malloc(sizeof(float) * sm_text_scratch_floats(&m->text)), vec[512], row[1024];
    float *words = malloc(sizeof(float) * 40 * 256), *hist = malloc(sizeof(float) * 100 * g->d);
    float *gs = malloc(sizeof(float) * sm_gen_scratch_floats(g, 40, 100));
    double e_vec = 0, e_base = 0, e_res = 0; size_t word_flips = 0, n_words = 0, word_far = 0;
    for (int p = 0; p < N_PROMPTS; p++) {
        uint32_t W; int32_t *ids = ids_of(p, &W);
        const float *gv = find("p%d.vec", p, 0)->data, *gw = find("p%d.words", p, 0)->data;
        sm_text_encode(&m->text, ids, W, vec, words, ts);
        for (uint32_t i = 0; i < m->text.out.n; i++) e_vec = fmax(e_vec, fabs(vec[i] - gv[i]));
        for (size_t i = 0; i < (size_t)W * m->text.d; i++) {   /* fp16 words: a value at a rounding edge may flip */
            const float d = fabsf(words[i] - gw[i]); word_flips += d > 0; n_words++;
            int e; frexpf(fmaxf(fabsf(words[i]), fabsf(gw[i])), &e);
            word_far += d > fmaxf(ldexpf(1, e - 11), 1e-6f);   /* one fp16 step, or a tiny value */
        }
        const uint32_t L = find("p%d.base", p, 0)->dim[1]; int32_t mask[100];
        for (uint32_t j = 0; j < L; j++) mask[j] = 512;
        sm_gen_prepare(g, gv, gw, W, L, gs);
        const float *raw = sm_gen_base(g, W, L, mask, gs);
        e_base = fmax(e_base, row_err(raw, L, 512, 512, s.cfg, find("p%d.base", p, 0)->data));
        const float *codes = find("p%d.res_codes", p, 0)->data;
        for (uint32_t q = 1; q <= 5; q += 4) {
            hist_of(g, codes, L, q, hist, row);
            raw = sm_gen_res(g, W, L, q, hist, gs);
            e_res = fmax(e_res, row_err(raw, L, g->res_out[q].n, 513, s.res_cfg, find("p%d.res%d", p, (int)q)->data));
        }
        free(ids);
    }
    printf("T2 text: vec max diff %.2e, words %zu of %zu differ (one fp16 step), %zu by more than that and 1e-6\n",
           e_vec, word_flips, n_words, word_far);
    printf("T2 logits (max diff / row std): base %.2e, residual %.2e\n", e_base, e_res);
    NEAR(e_vec, 0, 1e-4); CHECK(word_far == 0 && word_flips * 100 < n_words);
    NEAR(e_base, 0, 2e-4); NEAR(e_res, 0, 2e-4);
    free(ts); free(words); free(gs); free(hist);
}

static int desc(const void *a, const void *b) {
    const float x = *(const float *)a, y = *(const float *)b;
    return (x < y) - (x > y);
}

typedef struct { uint64_t seed; uint32_t stream; } key;
static float noise(void *u, uint32_t step, uint32_t pos, uint32_t code) {
    const key *k = u; return sm_hash_gumbel(k->seed, k->stream, step, pos, code);
}

/* best minus second best of x / temp + noise over the top k, as sm_pick sees row j */
static float margin(const float *x, uint32_t N, const sm_sampling *s, key k, uint32_t step, uint32_t j) {
    float sorted[513], top[2] = {-INFINITY, -INFINITY};
    memcpy(sorted, x, sizeof(float) * N); qsort(sorted, N, sizeof(float), desc);
    const float kth = s->top_k && s->top_k < N ? sorted[s->top_k - 1] : -INFINITY;
    for (uint32_t c = 0; c < N; c++) {
        if (x[c] < kth) continue;
        const float v = x[c] / s->temp + noise(&k, step, j, c);
        if (v > top[0]) { top[1] = top[0]; top[0] = v; } else if (v > top[1]) top[1] = v;
    }
    return top[0] - top[1];
}

static void mix(const float *raw, uint32_t L, uint32_t n, uint32_t used, float cfg, float *lg) {
    for (size_t j = 0; j < L; j++)
        for (size_t c = 0; c < used; c++) {
            const float cv = raw[j * n + c], uv = raw[(L + j) * n + c];
            lg[j * used + c] = uv + (cv - uv) * cfg;
        }
}

/* replays the base steps; gap[j] / when[j]: the margin and step where position j was last sampled */
static void replay_base(const sm_gen *g, uint32_t W, uint32_t L, const sm_sampling *s, uint64_t seed, float *gs,
                        float *gap, uint32_t *when) {
    int32_t ids[100], pick[100]; float score[100], prob[100], tmp[513], *lg = malloc(sizeof(float) * L * 512);
    uint8_t mask[100]; key k = {seed, 0};
    for (uint32_t j = 0; j < L; j++) { ids[j] = 512; score[j] = 0; }
    for (uint32_t i = 0; i < s->steps; i++) {
        sm_lowest(score, L, sm_nmask(i, s->steps, L), mask);
        for (uint32_t j = 0; j < L; j++) if (mask[j]) ids[j] = 512;
        mix(sm_gen_base(g, W, L, ids, gs), L, 512, 512, s->cfg, lg);
        sm_pick(lg, L, 512, mask, s->temp, s->top_k, noise, &k, i, pick, prob, tmp);
        for (uint32_t j = 0; j < L; j++) {
            score[j] = mask[j] ? prob[j] : 1e5f;
            if (!mask[j]) continue;
            ids[j] = pick[j]; when[j] = i; gap[j] = margin(lg + (size_t)j * 512, 512, s, k, i, j);
        }
    }
    free(lg);
}

/* the margin at the first code that differs from torch: by base step, then residual layer and position */
static float first_flip_gap(const sm_gen *g, const float *vec, const float *words, uint32_t W, uint32_t L,
                            const sm_sampling *s, uint64_t seed, const int16_t *got, const float *want, float *gs) {
    float gap[100], *hist = malloc(sizeof(float) * L * g->d), row[1024], codes[600], best = NAN;
    float *lg = malloc(sizeof(float) * L * 513); uint32_t when[100], first = UINT32_MAX;
    sm_gen_prepare(g, vec, words, W, L, gs);
    replay_base(g, W, L, s, seed, gs, gap, when);
    for (uint32_t j = 0; j < L; j++)
        if (got[j * 6] != (int16_t)want[j * 6] && when[j] < first) { first = when[j]; best = gap[j]; }
    for (uint32_t i = 0; i < 6 * L; i++) codes[i] = got[i] < 0 ? 512 : got[i];
    for (uint32_t q = 1; q <= s->res_layers && first == UINT32_MAX; q++) {
        hist_of(g, codes, L, q, hist, row);
        mix(sm_gen_res(g, W, L, q, hist, gs), L, g->res_out[q].n, 513, s->res_cfg, lg);
        for (uint32_t j = 0; j < L && first == UINT32_MAX; j++)
            if (got[j * 6 + q] != (int16_t)want[j * 6 + q]) {
                first = j; best = margin(lg + (size_t)j * 513, 513, s, (key){seed, 1}, q - 1, j);
            }
    }
    free(hist); free(lg);
    return best;
}

#define N_CLIPS (N_PROMPTS * 4 * 3)
typedef int16_t clip[600];
static const float *want_codes(int c, uint32_t *L) {   /* clip c: prompt c / 12, length (c / 3) % 4, seed c % 3 */
    *L = LENS[c / 3 % 4];
    return find("p%d.gen%d", c / 12, (int)*L)->data + (size_t)(c % 3) * *L * 6;
}

#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
#define MAX_L 17   /* sanitizer builds: the short clips only */
#else
#define MAX_L 100
#endif

/* every clip of at most max_l tokens end to end (text encoder included), into out */
static void generate_all(const sm_model *m, clip *out, uint32_t max_l, float *vecs, float *words) {
    const sm_sampling s = sampling(); const sm_gen *g = &m->gen;
    float *ts = malloc(sizeof(float) * sm_text_scratch_floats(&m->text));
    float *gs = malloc(sizeof(float) * sm_gen_scratch_floats(g, 40, 100));
    for (int c = 0; c < N_CLIPS; c++) {
        uint32_t W, L; int32_t *ids = ids_of(c / 12, &W);
        float *v = vecs + c / 12 * 512, *w = words + c / 12 * 40 * 256;
        want_codes(c, &L);
        if (c % 12 == 0) sm_text_encode(&m->text, ids, W, v, w, ts);
        if (L <= max_l) CHECK(gen_codes(g, v, w, W, L, &s, (uint64_t)(c % 3), out[c], gs) == SM_OK);
        free(ids);
    }
    free(ts); free(gs);
}

/* end to end: the F32 file vs torch's codes; the margin at the first flip of each clip that differs */
static void test_end_to_end(const sm_model *m, clip *got) {
    const sm_sampling s = sampling(); float *vecs = malloc(sizeof(float) * N_PROMPTS * 512);
    float *words = malloc(sizeof(float) * N_PROMPTS * 40 * 256);
    float *gs = malloc(sizeof(float) * sm_gen_scratch_floats(&m->gen, 40, 100));
    generate_all(m, got, MAX_L, vecs, words);
    int same = 0, clips = 0;
    for (int c = 0; c < N_CLIPS; c++) {
        uint32_t L, W, diff = 0; const float *want = want_codes(c, &L);
        if (L > MAX_L) continue;
        for (uint32_t i = 0; i < 6 * L; i++) diff += got[c][i] != (int16_t)want[i];
        clips++; same += !diff;
        if (!diff) continue;
        free(ids_of(c / 12, &W));
        const float gap = first_flip_gap(&m->gen, vecs + c / 12 * 512, words + c / 12 * 40 * 256, W, L, &s,
                                         (uint64_t)(c % 3), got[c], want, gs);
        printf("T5 prompt %d, %u tokens, seed %d: %u codes differ, noisy logit gap %.2e at the first flip\n",
               c / 12, L, c % 3, diff, gap);
    }
    printf("T5 end to end (F32): %d of %d clips identical\n", same, clips);
    CHECK(same * 100 >= 99 * clips);
    free(vecs); free(words); free(gs);
}

/* Q8_0 vs F32 in C, the same prompts and seeds, clips up to 49 tokens */
static void test_q8(const sm_model *q8, const clip *f32) {
    const uint32_t max_l = MAX_L < 49 ? MAX_L : 49;
    clip *got = malloc(sizeof(clip) * N_CLIPS); float *vecs = malloc(sizeof(float) * N_PROMPTS * 512);
    float *words = malloc(sizeof(float) * N_PROMPTS * 40 * 256);
    generate_all(q8, got, max_l, vecs, words);
    size_t codes = 0, agree = 0, base = 0, base_agree = 0; int same = 0, clips = 0;
    for (int c = 0; c < N_CLIPS; c++) {
        uint32_t L, d = 0; want_codes(c, &L);
        if (L > max_l) continue;
        for (uint32_t i = 0; i < 6 * L; i++) {
            d += got[c][i] != f32[c][i];
            if (i % 6 == 0) base_agree += got[c][i] == f32[c][i];
        }
        codes += 6 * L; agree += 6 * L - d; base += L; same += d == 0; clips++;
    }
    printf("T3 Q8_0 vs F32: %.1f%% of codes agree (layer 0: %.1f%%), %d of %d clips identical\n",
           100.0 * (double)agree / (double)codes, 100.0 * (double)base_agree / (double)base, same, clips);
    CHECK(agree * 2 > codes);
    free(got); free(vecs); free(words);
}

static double now(void) { struct timespec t; timespec_get(&t, TIME_UTC); return (double)t.tv_sec + (double)t.tv_nsec * 1e-9; }

/* the codes of prompt 0 at L tokens, seed 0; the best wall time of reps runs, in ms */
static double run_clip(const sm_model *m, uint32_t L, int reps, int16_t *codes) {
    const sm_sampling s = sampling(); uint32_t W; int32_t *ids = ids_of(0, &W); float vec[512], words[40 * 256];
    float *ts = malloc(sizeof(float) * sm_text_scratch_floats(&m->text));
    float *gs = malloc(sizeof(float) * sm_gen_scratch_floats(&m->gen, W, L)); double best = INFINITY;
    for (int r = 0; r < reps; r++) {
        const double t0 = now();
        sm_text_encode(&m->text, ids, W, vec, words, ts);
        CHECK(gen_codes(&m->gen, vec, words, W, L, &s, 0, codes, gs) == SM_OK);
        best = fmin(best, (now() - t0) * 1000);
    }
    free(ids); free(ts); free(gs);
    return best;
}

/* the same codes at 1, 2 and 4 threads (Q8_0, F32), and the speed per 5 s and 10 s clip at 1 and 4 threads */
static void test_threads(int speed) {
    const uint32_t threads[3] = {1, 2, 4}; const char *files[2] = {"SM_GOLDEN_Q8", "SM_GOLDEN_F32"};
    for (int f = 0; f < 2; f++) {
        int16_t want[600], got[600]; int same = 1;
        for (int t = 0; t < 3; t++) {
            sm_model *m = load_model(files[f], threads[t]);
            if (!m) return;
            run_clip(m, MAX_L < 49 ? MAX_L : 49, 1, t ? got : want);
            same &= !t || !memcmp(got, want, sizeof(int16_t) * 6 * (MAX_L < 49 ? MAX_L : 49));
            for (uint32_t L = 50; speed && f == 0 && t != 1 && L <= 100; L += 50)
                printf("speed (Q8_0, %u thread%s): %.0f ms per %u s clip\n", threads[t], t ? "s" : "",
                       run_clip(m, L, 3, got), L / 10);
            sm_model_free(m);
        }
        printf("%s: the same codes at 1, 2 and 4 threads: %s\n", files[f], same ? "yes" : "no");
        CHECK(same);
    }
}

int main(void) {
    CHECK(load_goldens("tests/data/gen_golden.bin")); if (!n_goldens) return CHECK_DONE();
    test_noise();
    sm_model *f32 = load_model("SM_GOLDEN_F32", 0), *q8 = load_model("SM_GOLDEN_Q8", 0);
    if (!f32 || !q8) printf("test_gen: no $SM_GOLDEN_F32 and $SM_GOLDEN_Q8, generator tests skipped\n");
    clip *codes = malloc(sizeof(clip) * N_CLIPS);
    if (f32) { test_passes(f32); test_end_to_end(f32, codes); }
    if (f32 && q8) test_q8(q8, (const clip *)codes);
    sm_model_free(f32); sm_model_free(q8); free(codes);
    if (f32 && q8) test_threads(MAX_L == 100);
    free_goldens();
    return CHECK_DONE();
}
