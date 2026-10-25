/* The public API through the shared library: misuse, parameters, decoding, BVH; with $SM_MODEL_DIR generation. */
#define _POSIX_C_SOURCE 200809L   /* posix_memalign */
#include "sm.h"
#include "check.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static atomic_long n_allocs;
static void *c_alloc(void *u, size_t size, size_t align) {
    (void)u; void *p = NULL; atomic_fetch_add(&n_allocs, 1);
    return posix_memalign(&p, align, size ? size : 1) ? NULL : p;
}
static void c_free(void *u, void *p) { (void)u; free(p); }

static sm_model *load(const char *dir, const char *file) {
    char path[512], err[128]; sm_model *m = NULL; sm_hooks h = SM_HOOKS_INIT; h.alloc = c_alloc; h.free = c_free;
    snprintf(path, sizeof path, "%s/%s", dir, file);
    if (sm_model_load_file(path, &h, &m, err, sizeof err) != SM_OK) fprintf(stderr, "%s: %s\n", path, err);
    return m;
}

/* every function with NULL, out-of-range or out-of-order arguments; plain outputs stay as they were */
static void test_misuse(const sm_model *m) {
    sm_ctx *c = NULL; sm_request *r = NULL; sm_motion *mo = NULL; sm_text *t = NULL; int32_t iv = 77; double dv = 7.5; uint32_t u = 9;
    const float *fp = (const float *)&u; float off[10] = {5}; size_t need = 3; char buf[4] = "abc";
    CHECK(sm_ctx_create(m, &c) == SM_OK && sm_request_create(m, &r) == SM_OK);
    CHECK(sm_motion_create(m, &mo) == SM_OK && sm_text_create(m, &t) == SM_OK);
    int16_t codes[12] = {1, 2, 3, 4, 5, 6, 512, 0, 0, 0, 0, 0}; sm_motion *outs[1] = {mo};
    const struct { sm_status got, want; int line; } cases[] = {
#define C(call, want) {call, want, __LINE__}
        C(sm_ctx_create(NULL, &c), SM_E_ARG), C(sm_ctx_create(m, NULL), SM_E_ARG), C(sm_request_create(m, NULL), SM_E_ARG),
        C(sm_motion_create(NULL, &mo), SM_E_ARG), C(sm_text_create(NULL, &t), SM_E_ARG),
        C(sm_ctx_reserve(NULL, 1, 1), SM_E_ARG), C(sm_ctx_reserve(c, 0, 1), SM_E_ARG), C(sm_ctx_reserve(c, 99, 1), SM_E_ARG),
        C(sm_ctx_reserve(c, 1, 0), SM_E_ARG), C(sm_ctx_reserve(c, 1, 17), SM_E_ARG),
        C(sm_request_clone(NULL, &r), SM_E_ARG), C(sm_request_clone(r, NULL), SM_E_ARG), C(sm_request_reset(NULL), SM_E_ARG),
        C(sm_request_set_int(NULL, SM_P_SEED, 1), SM_E_ARG), C(sm_request_set_int(r, 0, 1), SM_E_ARG),
        C(sm_request_set_int(r, 99, 1), SM_E_ARG), C(sm_request_set_int(r, SM_P_STEPS, 0), SM_E_ARG),
        C(sm_request_set_int(r, SM_P_SEED, -1), SM_E_ARG), C(sm_request_set_float(r, SM_P_STEPS, 2.5), SM_E_ARG),
        C(sm_request_set_float(r, SM_P_SECONDS, 0.4), SM_E_ARG), C(sm_request_set_float(r, SM_P_CFG, NAN), SM_E_ARG),
        C(sm_request_set_float(r, SM_P_SECONDS, 10.5), SM_E_ARG),
        C(sm_request_get_int(r, SM_P_CFG, &iv), SM_E_ARG), C(sm_request_get_int(r, SM_P_SEED, NULL), SM_E_ARG),
        C(sm_request_get_float(NULL, SM_P_CFG, &dv), SM_E_ARG), C(sm_request_get_float(r, 42, &dv), SM_E_ARG),
        C(sm_param_bounds(m, 0, NULL, NULL, NULL, NULL), SM_E_ARG),
        C(sm_request_set_by_name(r, "steps", "1O"), SM_E_ARG), C(sm_request_set_by_name(r, "nope", "1"), SM_E_ARG),
        C(sm_request_set_by_name(r, "steps", ""), SM_E_ARG), C(sm_request_set_by_name(r, "cfg", "1e2"), SM_E_ARG), C(sm_request_set_by_name(r, NULL, "1"), SM_E_ARG),
        C(sm_request_set_text(NULL, "a", -1), SM_E_ARG), C(sm_request_set_text(r, NULL, -1), SM_E_ARG),
        C(sm_request_set_text(r, "\xC3\x28", 2), SM_E_ARG), C(sm_request_set_text(r, "a\0b", 3), SM_E_ARG),
        C(sm_request_set_text_encoded(r, t), SM_E_ARG), C(sm_request_set_text_encoded(r, NULL), SM_E_ARG),
        C(sm_request_set_prefix(r, mo), SM_E_ARG), C(sm_request_set_prefix(NULL, NULL), SM_E_ARG),
        C(sm_ctx_encode_text(NULL, "a", -1, t), SM_E_ARG), C(sm_ctx_encode_text(c, "a", -1, NULL), SM_E_ARG),
        C(sm_text_vector(t, &fp, &u), SM_E_STATE), C(sm_text_vector(NULL, &fp, &u), SM_E_ARG),
        C(sm_text_set_vector(t, off, 10), SM_E_ARG), C(sm_text_set_vector(NULL, off, 10), SM_E_ARG),
        C(sm_text_unknown_words(t, buf, 4, &need, &u), SM_E_STATE), C(sm_text_unknown_words(NULL, 0, 0, 0, 0), SM_E_ARG),
        C(sm_ctx_generate(NULL, r, outs, 1), SM_E_ARG), C(sm_ctx_generate(c, NULL, outs, 1), SM_E_ARG),
        C(sm_ctx_generate(c, r, NULL, 1), SM_E_ARG), C(sm_ctx_generate(c, r, outs, 0), SM_E_ARG),
        C(sm_ctx_generate(c, r, outs, 17), SM_E_ARG), C(sm_ctx_begin(NULL, r, outs, 1), SM_E_ARG),
        C(sm_ctx_step(c, &u, &u), SM_E_STATE), C(sm_ctx_step(NULL, NULL, NULL), SM_E_ARG), C(sm_ctx_preview(c), SM_E_STATE),
        C(sm_ctx_finish(c), SM_E_STATE), C(sm_ctx_preview(NULL), SM_E_ARG), C(sm_ctx_finish(NULL), SM_E_ARG),
        C(sm_ctx_decode_codes(NULL, NULL, codes, 2, mo), SM_E_ARG), C(sm_ctx_decode_codes(c, NULL, NULL, 2, mo), SM_E_ARG),
        C(sm_ctx_decode_codes(c, NULL, codes, 0, mo), SM_E_ARG), C(sm_ctx_decode_codes(c, NULL, codes, 2, mo), SM_E_ARG),
        C(sm_ctx_decode_codes(c, NULL, codes, 2, NULL), SM_E_ARG), C(sm_motion_reserve(NULL, 1), SM_E_ARG),
        C(sm_motion_reserve(mo, -1), SM_E_ARG), C(sm_motion_copy(NULL, mo), SM_E_ARG), C(sm_motion_copy(mo, NULL), SM_E_ARG),
        C(sm_motion_write_bvh(mo, 100, buf, 4, &need), SM_E_STATE), C(sm_motion_write_bvh(NULL, 100, buf, 4, &need), SM_E_ARG),
        C(sm_motion_write_bvh(mo, 0, buf, 4, &need), SM_E_ARG), C(sm_motion_write_bvh(mo, 100, NULL, 0, NULL), SM_E_ARG),
        C(sm_skeleton_offsets(m, off, 10), SM_E_BUFFER), C(sm_skeleton_offsets(NULL, off, 66), SM_E_ARG),
#undef C
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++)
        if (cases[i].got != cases[i].want) { bad++; fprintf(stderr, "test_api.c:%d: got %s, want %s\n", cases[i].line,
                                                            sm_status_name(cases[i].got), sm_status_name(cases[i].want)); }
    printf("misuse: %zu calls, %d with the wrong status\n", sizeof cases / sizeof cases[0], bad);
    CHECK(bad == 0);
    CHECK(iv == 77 && dv == 7.5 && u == 9 && fp == (const float *)&u && off[0] == 5 && need == 3 && !strcmp(buf, "abc"));
    CHECK(sm_motion_frames(mo) == 0 && !sm_motion_joints(mo, &u) && u == 0 && sm_motion_seed(mo) == -1);
    CHECK(!sm_motion_offsets(mo, &u) && u == 0 && !sm_motion_offsets(NULL, NULL));
    CHECK(sm_request_last_error(r)[0] && sm_ctx_last_error(c)[0]);
    CHECK(sm_skeleton_joint_name(m, 22) == NULL && sm_skeleton_parent(m, 22) == -1 && sm_skeleton_parent(m, 0) == -1);
    CHECK(sm_model_info(m, 99) == -1 && !sm_model_meta(m, "nope"));
    sm_ctx_abort(c); sm_ctx_abort(NULL); sm_ctx_cancel(NULL);
    sm_text_free(t); sm_motion_free(mo); sm_request_free(r); sm_ctx_free(c);
    sm_text_free(NULL); sm_motion_free(NULL); sm_request_free(NULL); sm_ctx_free(NULL);
}

/* a rejected value names itself, the value and the range */
static void test_messages(const sm_model *m) {
    sm_ctx *c = NULL; sm_request *r = NULL; sm_motion *mo = NULL; char err[96], small[8], text[5001];
    CHECK(sm_ctx_create(m, &c) == SM_OK && sm_request_create(m, &r) == SM_OK && sm_motion_create(m, &mo) == SM_OK);
    const int16_t codes[6] = {512};
    memset(text, 'a', 5000), text[5000] = 0;
    CHECK(sm_request_set_float(r, SM_P_SECONDS, 12.5) == SM_E_ARG);
    CHECK(strstr(sm_request_last_error(r), "seconds = 12.5 is out of range 0.5 .. ") == sm_request_last_error(r));
    CHECK(sm_request_set_float(r, SM_P_STEPS, 2.5) == SM_E_ARG);
    CHECK(strstr(sm_request_last_error(r), "steps = 2.5 is not a whole number in 1 .. ") == sm_request_last_error(r));
    CHECK(sm_request_set_float(r, SM_P_CFG, NAN) == SM_E_ARG && !strcmp(sm_request_last_error(r), "cfg = nan is out of range 0 .. 20"));
    CHECK(sm_request_set_text(r, text, -1) == SM_E_ARG);
    CHECK(!strcmp(sm_request_last_error(r), "text bytes = 5000 is out of range 0 .. 4096"));
    CHECK(sm_ctx_reserve(c, 1, 17) == SM_E_ARG && !strcmp(sm_ctx_last_error(c), "max_variants = 17 is out of range 1 .. 16"));
    CHECK(sm_ctx_decode_codes(c, NULL, codes, 1, mo) == SM_E_ARG);
    CHECK(!strcmp(sm_ctx_last_error(c), "a base code = 512 is out of range 0 .. 511"));
    CHECK(sm_generate_simple(m, "a", 0.1, 1, mo, err, sizeof err) == SM_E_ARG && strstr(err, "seconds = 0.1 is out") == err);
    CHECK(sm_generate_simple(m, "a", 0.1, 1, mo, small, sizeof small) == SM_E_ARG && !strcmp(small, "seconds"));
    CHECK(sm_generate_simple(NULL, "a", 2, 1, mo, err, sizeof err) == SM_E_ARG && err[0]);
    CHECK(sm_generate_simple(m, "a", 2, 1, NULL, NULL, 0) == SM_E_ARG);
    sm_motion_free(mo); sm_request_free(r); sm_ctx_free(c);
}

static void test_params(const sm_model *m) {
    sm_request *r = NULL, *q = NULL; int n = 0; double lo, hi, d, v; int32_t is_int, iv;
    CHECK(sm_request_create(m, &r) == SM_OK);
    for (sm_param p = sm_param_next(0); p > 0; p = sm_param_next(p), n++) {
        CHECK(sm_param_from_name(sm_param_name(p)) == p && sm_param_bounds(m, p, &lo, &hi, &d, &is_int) == SM_OK);
        CHECK(sm_request_get_float(r, p, &v) == SM_OK && v == d && lo <= d && d <= hi);
        CHECK(sm_request_set_float(r, p, hi) == SM_OK && sm_request_get_float(r, p, &v) == SM_OK && v == hi);
        CHECK(is_int ? sm_request_get_int(r, p, &iv) == SM_OK && iv == (int32_t)hi : sm_request_set_int(r, p, 1) == SM_OK);
    }
    CHECK(n == 12);
    CHECK(sm_request_set_by_name(r, "seconds", "3.5") == SM_OK && sm_request_get_float(r, SM_P_SECONDS, &v) == SM_OK && v == 3.5);
    CHECK(sm_request_set_by_name(r, "temp", ".25") == SM_OK && sm_request_get_float(r, SM_P_TEMP, &v) == SM_OK && v == 0.25);
    CHECK(sm_request_set_by_name(r, "steps", "+12") == SM_OK && sm_request_get_int(r, SM_P_STEPS, &iv) == SM_OK && iv == 12);
    CHECK(sm_request_clone(r, &q) == SM_OK && sm_request_get_int(q, SM_P_STEPS, &iv) == SM_OK && iv == 12);
    CHECK(sm_request_reset(q) == SM_OK && sm_request_get_int(q, SM_P_STEPS, &iv) == SM_OK && iv == 10);
    const char *api = sm_api_schema_json(), *ms = sm_model_schema_json(m);
    CHECK(api[0] == '{' && api[strlen(api) - 1] == '}' && strstr(api, "\"name\":\"res_cfg\"") && !strstr(api, "idle"));
    CHECK(ms && ms[strlen(ms) - 1] == '}' && strstr(ms, "\"name\":\"seconds\",\"type\":\"float\""));
    CHECK(!strcmp(api, ms));   /* this file's ranges and defaults are the runtime's */
    printf("schema: %zu bytes (api), %zu (model)\n", strlen(api), strlen(ms));
    sm_request_free(r); sm_request_free(q);
}

static void qmul(const double a[4], const double b[4], double o[4]) {   /* w x y z */
    o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

/* FK of local rotations q (x y z w) on the offsets from root positions vs the joints: mean and max distance */
static void fk_error(const sm_model *m, const float *q, const float *off, const float *root, const float *j, uint32_t T,
                     double *mean, double *worst) {
    double sum = 0; *worst = 0;
    for (uint32_t t = 0; t < T; t++) {
        double G[22][4], P[22][3];
        for (int k = 0; k < 22; k++) {
            const float *l = q + ((size_t)t * 22 + (size_t)k) * 4; const int p = sm_skeleton_parent(m, (uint32_t)k);
            const double b[4] = {l[3], l[0], l[1], l[2]}, id[4] = {1, 0, 0, 0};
            qmul(p < 0 ? id : G[p], b, G[k]);
            if (p < 0) { for (int i = 0; i < 3; i++) P[k][i] = root[3 * t + (uint32_t)i]; continue; }
            const double *g = G[p], v[4] = {0, off[3 * k], off[3 * k + 1], off[3 * k + 2]};
            const double gi[4] = {g[0], -g[1], -g[2], -g[3]};
            double gv[4], r[4];
            qmul(g, v, gv); qmul(gv, gi, r);
            for (int i = 0; i < 3; i++) P[k][i] = P[p][i] + r[i + 1];
        }
        for (int k = 0; k < 22; k++) {
            const float *w = j + ((size_t)t * 22 + (size_t)k) * 3;
            const double e = sqrt(pow(P[k][0] - w[0], 2) + pow(P[k][1] - w[1], 2) + pow(P[k][2] - w[2], 2));
            sum += e; *worst = fmax(*worst, e);
        }
    }
    *mean = sum / (22.0 * T);
}

/* the BVH text read back (scale 100): offsets and root in metres, rotations x y z w from the ZYX Euler angles */
static uint32_t read_bvh(const sm_model *m, const char *bvh, float *off, float *root, float *q, uint32_t max_t) {
    int order[22], n = 0; uint32_t T = 0; char name[32]; const char *p = bvh;
    for (; (p = strpbrk(p, "RJEOM")) && *p != 'M'; p++) {
        if ((!strncmp(p, "ROOT ", 5) || !strncmp(p, "JOINT ", 6)) && sscanf(strchr(p, ' ') + 1, "%31s", name) == 1)
            for (uint32_t k = 0; k < 22; k++) if (!strcmp(name, sm_skeleton_joint_name(m, k))) order[n] = (int)k;
        if (!strncmp(p, "End Site", 8)) p = strstr(p, "OFFSET") + 6;
        if (!strncmp(p, "OFFSET", 6)) { float *o = off + 3 * order[n++]; sscanf(p + 6, "%f %f %f", o, o + 1, o + 2); }
    }
    for (int k = 0; k < 66; k++) off[k] /= 100;
    p = strstr(p, "Frame Time:"); p = strchr(p, '\n') + 1;
    for (char *e; T < max_t && *p; T++, p = e) {
        for (int k = 0; k < 3; k++) root[3 * T + (uint32_t)k] = strtof(p, &e) / 100, p = e;
        for (int i = 0; i < 22; i++) {
            double a[3], h[3][4] = {{0}}, zy[4], r[4];
            for (int k = 0; k < 3; k++) a[k] = strtod(p, &e) * 3.14159265358979323846 / 360, p = e;
            for (int k = 0; k < 3; k++) h[k][0] = cos(a[k]), h[k][3 - k] = sin(a[k]);   /* about z, y, x */
            qmul(h[0], h[1], zy); qmul(zy, h[2], r);
            float *o = q + ((size_t)T * 22 + (size_t)order[i]) * 4;
            o[0] = (float)r[1], o[1] = (float)r[2], o[2] = (float)r[3], o[3] = (float)r[0];
        }
        while (*e == ' ' || *e == '\n') e++;
    }
    return T;
}

static void test_decode(const sm_model *m) {
    sm_ctx *c = NULL; sm_motion *mo = NULL, *cp = NULL; uint32_t n; int16_t codes[40 * 6]; size_t need = 0;
    for (int i = 0; i < 40 * 6; i++) codes[i] = (int16_t)(i % 6 == 5 ? -1 : (i * 37) % 512);
    CHECK(sm_ctx_create(m, &c) == SM_OK && sm_motion_create(m, &mo) == SM_OK && sm_motion_create(m, &cp) == SM_OK);
    CHECK(sm_ctx_decode_codes(c, NULL, codes, 40, mo) == SM_OK && sm_motion_frames(mo) == 80 && sm_motion_seconds(mo) == 4.0);
    CHECK(sm_motion_joints(mo, &n) && n == 80 * 66 && sm_motion_codes(mo, &n) && n == 240 && sm_motion_seed(mo) == -1);
    CHECK(sm_motion_features(mo, &n) && n == 80 * 263 && sm_motion_contacts(mo, &n) && n == 320);
    CHECK(sm_motion_confidence(mo, &n)[0] == 1 && n == 40 && !memcmp(sm_motion_codes(mo, NULL), codes, sizeof codes));
    CHECK(sm_motion_write_bvh(mo, 100, NULL, 0, &need) == SM_OK && need > 1000);
    char *bvh = malloc(need), small[8] = "x";
    CHECK(sm_motion_write_bvh(mo, 100, small, 8, &need) == SM_E_BUFFER && small[0] == 'x');
    CHECK(sm_motion_write_bvh(mo, 100, bvh, need, &need) == SM_OK && strlen(bvh) + 1 == need);
    CHECK(!strncmp(bvh, "HIERARCHY\nROOT pelvis", 21) && strstr(bvh, "Frames: 80\nFrame Time: 0.05\n"));
    int lines = 0, open = 0;
    for (const char *p = bvh; *p; p++) lines += *p == '\n', open += (*p == '{') - (*p == '}');
    CHECK(open == 0 && strstr(bvh, "End Site") && !strchr(bvh, ','));
    printf("bvh: %zu bytes, %d lines\n", need, lines);
    CHECK(sm_motion_copy(cp, mo) == SM_OK && sm_motion_frames(cp) == 80);
    CHECK(!memcmp(sm_motion_joints(cp, NULL), sm_motion_joints(mo, NULL), 80 * 66 * sizeof(float)));
    CHECK(sm_motion_offsets(mo, &n) && n == 66);
    CHECK(!memcmp(sm_motion_offsets(cp, NULL), sm_motion_offsets(mo, NULL), 66 * sizeof(float)));
    free(bvh); sm_motion_free(mo); sm_motion_free(cp); sm_ctx_free(c);
}

/* without a generator or encoder: kept blocks still decode as one timeline; the rest says what is missing */
static void test_edit_api(const sm_model *m) {
    sm_ctx *c = NULL; sm_request *r = NULL; sm_motion *mo = NULL; int16_t codes[2][30 * 6]; float j[4 * 66] = {0};
    for (int i = 0; i < 30 * 6; i++) codes[0][i] = (int16_t)((i * 37) % 512), codes[1][i] = (int16_t)((i * 11) % 512);
    CHECK(sm_ctx_create(m, &c) == SM_OK && sm_request_create(m, &r) == SM_OK && sm_motion_create(m, &mo) == SM_OK);
    CHECK(sm_model_info(m, SM_I_ENCODER) == 0 && sm_model_info(m, SM_I_PATH_INPUT + 1) == -1);
    CHECK(sm_ctx_encode_motion(c, NULL, j, 4, mo) == SM_E_UNSUPPORTED && strstr(sm_ctx_last_error(c), "no motion encoder"));
    CHECK(sm_ctx_encode_motion(c, NULL, NULL, 4, mo) == SM_E_ARG && sm_ctx_encode_motion(NULL, NULL, j, 4, mo) == SM_E_ARG);
    const int16_t *keep[2] = {codes[0], codes[1]}; const char *texts[2] = {NULL, "a person waves"};
    double sec[2] = {3, 2.5}; const int32_t seeds[2] = {0, 0}; uint32_t first[3] = {0};
    CHECK(sm_ctx_generate_blocks(c, r, 2, NULL, NULL, sec, seeds, keep, first, mo) == SM_OK && sm_motion_frames(mo) == 110);
    CHECK(first[1] == 60 && first[2] == 110 && sm_motion_seed(mo) == -1);
    CHECK(!memcmp(sm_motion_codes(mo, NULL) + 30 * 6, codes[1], 25 * 6 * sizeof(int16_t)));
    keep[1] = NULL;
    CHECK(sm_ctx_generate_blocks(c, r, 2, texts, NULL, sec, seeds, keep, NULL, mo) == SM_E_UNSUPPORTED && sm_motion_frames(mo) == 0);
    keep[1] = codes[1], sec[1] = 11;
    CHECK(sm_ctx_generate_blocks(c, r, 2, NULL, NULL, sec, seeds, keep, NULL, mo) == SM_E_ARG &&
          strstr(sm_ctx_last_error(c), "block 1 seconds = 11"));
    sec[1] = 2.5, codes[1][6] = -1;
    CHECK(sm_ctx_generate_blocks(c, r, 2, NULL, NULL, sec, seeds, keep, NULL, mo) == SM_E_ARG &&
          strstr(sm_ctx_last_error(c), "block 1: a kept code"));
    CHECK(sm_ctx_generate_blocks(c, r, 2, NULL, NULL, sec, seeds, NULL, NULL, mo) == SM_E_UNSUPPORTED);
    CHECK(sm_ctx_generate_blocks(c, r, 0, NULL, NULL, sec, seeds, keep, NULL, mo) == SM_E_ARG);
    CHECK(sm_ctx_generate_blocks(c, NULL, 2, NULL, NULL, sec, seeds, keep, NULL, mo) == SM_E_ARG);
    CHECK(sm_ctx_generate_blocks(c, r, 2, NULL, NULL, NULL, seeds, keep, NULL, mo) == SM_E_ARG);
    CHECK(sm_request_set_codes(r, codes[0], 30) == SM_OK && sm_request_set_codes(r, NULL, 0) == SM_OK);
    CHECK(sm_request_set_codes(NULL, codes[0], 1) == SM_E_ARG && sm_request_set_int(r, SM_P_LOOP, 2) == SM_E_ARG);
    CHECK(sm_request_set_int(r, SM_P_LOOP, 1) == SM_OK && sm_ctx_decode_codes(c, r, codes[0], 30, mo) == SM_OK);
    sm_motion_free(mo); sm_request_free(r); sm_ctx_free(c);
}

static void test_model_file(const char *path) {
    sm_model *m = NULL; char err[64]; sm_hooks h = SM_HOOKS_INIT; h.struct_size = sizeof h + 8;
    CHECK(sm_model_load_file(path, &h, &m, err, sizeof err) == SM_E_ARG && !m && strstr(err, "sm_hooks.struct_size = ") == err);
    CHECK(sm_model_load_file("/nonexistent/m.gguf", NULL, &m, err, sizeof err) == SM_E_IO && !m);
    CHECK(sm_model_load_file(path, NULL, &m, NULL, 0) == SM_OK && sm_version() >> 16 == 1 && sm_build_info()[0]);
    CHECK(sm_model_info(NULL, 1) == -1);
    sm_model_free(m);
}

typedef struct { sm_ctx *c; atomic_int passes, sent; } watch;
static int32_t on_progress(void *u, uint32_t done, uint32_t total) {   /* holds the run at pass 2 until cancel is sent */
    watch *w = u; (void)done; (void)total;
    if (atomic_fetch_add(&w->passes, 1) == 1) while (!atomic_load(&w->sent)) {}
    return 0;
}
static void *canceller(void *u) { watch *w = u; while (atomic_load(&w->passes) < 2) {} sm_ctx_cancel(w->c); atomic_store(&w->sent, 1); return NULL; }
static int32_t stop_at_3(void *u, uint32_t done, uint32_t total) { (void)u; (void)total; return done >= 3; }

static void test_cancel(const sm_model *m, sm_request *r, sm_motion *mo) {
    sm_ctx *c; watch w = {NULL, 0, 0}; pthread_t th;
    CHECK(sm_ctx_create(m, &c) == SM_OK); w.c = c;
    sm_ctx_set_progress(c, on_progress, &w);
    pthread_create(&th, NULL, canceller, &w);
    CHECK(sm_ctx_generate(c, r, &mo, 1) == SM_E_CANCELLED && sm_motion_frames(mo) == 0);
    pthread_join(th, NULL);
    CHECK(sm_ctx_generate(c, r, &mo, 1) == SM_E_CANCELLED && strstr(sm_ctx_last_error(c), "sm_ctx_clear_cancel"));
    sm_ctx_clear_cancel(c); sm_ctx_set_progress(c, stop_at_3, NULL);
    CHECK(sm_ctx_generate(c, r, &mo, 1) == SM_E_CANCELLED && sm_motion_frames(mo) == 0);
    sm_ctx_set_progress(c, NULL, NULL);
    CHECK(sm_ctx_generate(c, r, &mo, 1) == SM_OK && sm_motion_frames(mo) > 0);
    printf("cancel: from another thread after %d passes, and from the progress callback\n", atomic_load(&w.passes));
    sm_ctx_free(c);
}

/* the same codes from sm_ctx_generate, the step API, a variant batch and a reused text encoding */
static void test_generate(const sm_model *m) {
    sm_ctx *c = NULL; sm_request *r = NULL; sm_motion *a = NULL, *b[3] = {0}; sm_text *t = NULL; uint32_t n, done, total, steps = 0; size_t need;
    CHECK(sm_ctx_create(m, &c) == SM_OK && sm_request_create(m, &r) == SM_OK && sm_text_create(m, &t) == SM_OK);
    CHECK(sm_motion_create(m, &a) == SM_OK);
    for (int i = 0; i < 3; i++) CHECK(sm_motion_create(m, &b[i]) == SM_OK);
    CHECK(sm_request_set_text(r, "a person walks forward and waves", -1) == SM_OK && sm_request_set_int(r, SM_P_SEED, 4) == SM_OK);
    CHECK(sm_request_set_float(r, SM_P_SECONDS, 3) == SM_OK && sm_ctx_generate(c, r, &a, 1) == SM_OK);
    CHECK(sm_motion_frames(a) == 60 && sm_motion_seed(a) == 4 && sm_motion_codes(a, &n) && n == 180);
    CHECK(sm_motion_confidence(a, &n)[0] > 0 && n == 30 && sm_ctx_memory_bytes(c) > 0);
    CHECK(sm_request_set_int(r, SM_P_SEED, 3) == SM_OK && sm_ctx_begin(c, r, b, 3) == SM_OK);
    CHECK(sm_ctx_begin(c, r, b, 3) == SM_E_STATE && sm_ctx_generate(c, r, b, 1) == SM_E_STATE);
    sm_status s;
    while ((s = sm_ctx_step(c, &done, &total)) == SM_OK) if (++steps == 5) CHECK(sm_ctx_preview(c) == SM_OK && sm_motion_frames(b[0]) == 60);
    CHECK(s == SM_DONE && done == total && total == 45 && steps + 1 == 45 && sm_ctx_step(c, &done, &total) == SM_E_STATE);
    CHECK(sm_ctx_finish(c) == SM_OK && sm_motion_seed(b[1]) == 4 && sm_motion_seed(b[2]) == 5);
    CHECK(!memcmp(sm_motion_codes(a, NULL), sm_motion_codes(b[1], NULL), 180 * sizeof(int16_t)));
    CHECK(!memcmp(sm_motion_joints(a, NULL), sm_motion_joints(b[1], NULL), 60 * 66 * sizeof(float)));
    CHECK(sm_ctx_begin(c, r, b, 1) == SM_OK && sm_ctx_step(c, NULL, NULL) == SM_OK);
    sm_ctx_abort(c);
    CHECK(sm_motion_frames(b[0]) == 0 && sm_ctx_finish(c) == SM_E_STATE);
    CHECK(sm_ctx_encode_text(c, "a person walks forward and waves", -1, t) == SM_OK && sm_request_set_text_encoded(r, t) == SM_OK);
    CHECK(sm_request_set_int(r, SM_P_SEED, 4) == SM_OK && sm_ctx_generate(c, r, &b[0], 1) == SM_OK);
    CHECK(!memcmp(sm_motion_codes(a, NULL), sm_motion_codes(b[0], NULL), 180 * sizeof(int16_t)));
    const float *vec; uint32_t dim; char words[256];
    CHECK(sm_text_vector(t, &vec, &dim) == SM_OK && (int32_t)dim == sm_model_info(m, SM_I_TEXT_DIM));
    CHECK(sm_text_set_vector(t, vec, dim) == SM_OK && sm_request_set_text_encoded(r, t) == SM_OK);
    CHECK(sm_ctx_generate(c, r, &b[0], 1) == SM_OK && sm_motion_frames(b[0]) == 60);   /* a vector, no words */
    CHECK(sm_ctx_encode_text(c, "a person does photosynthesis by a kaleidoscope", -1, t) == SM_OK);
    CHECK(sm_text_unknown_words(t, NULL, 0, &need, &n) == SM_OK && sm_text_unknown_words(t, words, need, &need, &n) == SM_OK);
    CHECK(n == 2 && !strcmp(words, "photosynthesis\nkaleidoscope") && need == 28);
    CHECK(sm_text_unknown_words(t, words, 27, &need, &n) == SM_E_BUFFER && need == 28);
    char err[96];
    CHECK(sm_generate_simple(m, "a person jumps", 2, 1, a, err, sizeof err) == SM_OK && !err[0] && sm_motion_frames(a) == 40);
    CHECK(sm_generate_simple(m, NULL, 2, 1, a, err, sizeof err) == SM_E_ARG && !strcmp(err, "text is NULL"));
    CHECK(sm_request_set_text(r, "a person walks", -1) == SM_OK);
    test_cancel(m, r, a);
    sm_text_free(t); sm_motion_free(a); for (int i = 0; i < 3; i++) sm_motion_free(b[i]);
    sm_request_free(r); sm_ctx_free(c);
}

/* foot lock: ankles still while in contact, the leg bones keep their lengths; in place: the root stays at x = z = 0 */
static void test_post(const sm_model *m) {
    sm_ctx *c = NULL; sm_request *r = NULL; sm_motion *on = NULL, *off = NULL; uint32_t n; double slide[2] = {0}, stretch = 0, low = 1; int cnt = 0;
    CHECK(sm_ctx_create(m, &c) == SM_OK && sm_request_create(m, &r) == SM_OK);
    CHECK(sm_motion_create(m, &on) == SM_OK && sm_motion_create(m, &off) == SM_OK);
    CHECK(sm_request_set_text(r, "a person walks forward", -1) == SM_OK && sm_request_set_float(r, SM_P_SECONDS, 6) == SM_OK);
    CHECK(sm_ctx_generate(c, r, &on, 1) == SM_OK && sm_request_set_int(r, SM_P_FOOT_LOCK, 0) == SM_OK);
    CHECK(sm_ctx_generate(c, r, &off, 1) == SM_OK);
    const uint8_t *ct = sm_motion_contacts(on, &n); const uint32_t T = sm_motion_frames(on);
    for (int k = 0; k < 2; k++) {
        const float *j = sm_motion_joints(k ? off : on, NULL);
        for (uint32_t t = 1; t < T; t++)
            for (int s = 0; s < 2; s++) {
                const int ank = 7 + s, col = 2 * s;
                if (!ct[4 * t + (uint32_t)col] || !ct[4 * (t - 1) + (uint32_t)col]) continue;
                const float *p = j + ((size_t)t * 22 + (size_t)ank) * 3, *q = p - 66;
                slide[k] += hypot(p[0] - q[0], p[2] - q[2]) * 20; cnt += !k;
            }
        for (uint32_t t = 0; !k && t < T; t++)
            for (int b = 4; b <= 8; b++) {   /* knees and ankles: the bone to their parent */
                const float *p = j + ((size_t)t * 22 + (size_t)b) * 3, *q = j + ((size_t)t * 22 + (size_t)sm_skeleton_parent(m, (uint32_t)b)) * 3;
                const float *p0 = sm_motion_joints(off, NULL) + (p - j), *q0 = sm_motion_joints(off, NULL) + (q - j);
                if (b == 6) continue;
                stretch = fmax(stretch, fabs(hypot(hypot(p[0] - q[0], p[1] - q[1]), p[2] - q[2]) -
                                             hypot(hypot(p0[0] - q0[0], p0[1] - q0[1]), p0[2] - q0[2])));
            }
        for (uint32_t t = 0; !k && t < T; t++) for (int f = 7; f <= 11; f += f == 8 ? 2 : 1) low = fmin(low, j[((size_t)t * 22 + (size_t)f) * 3 + 1]);
    }
    printf("foot lock: ankle slide in contact %.3f m/s (off: %.3f), bone length change %.1e m, lowest foot %.3f m\n",
           slide[0] / cnt, slide[1] / cnt, stretch, low);
    CHECK(slide[0] < 0.5 * slide[1] && stretch < 1e-4 && low >= -1e-4);
    double mean, worst, bmean, bworst; float ref[66], off_b[66]; size_t need;
    static float root_b[200 * 3], q_b[200 * 88];
    sm_skeleton_offsets(m, ref, 66);
    const float *rot = sm_motion_rotations(off, NULL), *root = sm_motion_root(off, NULL);
    fk_error(m, rot, ref, root, sm_motion_joints(off, NULL), T, &mean, &worst);
    CHECK(sm_motion_write_bvh(off, 100, NULL, 0, &need) == SM_OK);
    char *bvh = malloc(need);
    CHECK(sm_motion_write_bvh(off, 100, bvh, need, &need) == SM_OK && read_bvh(m, bvh, off_b, root_b, q_b, 200) == T);
    fk_error(m, q_b, off_b, root_b, sm_motion_joints(off, NULL), T, &bmean, &bworst);
    printf("rotations: FK vs the joints on the reference skeleton: mean %.1f mm, max %.1f mm; BVH (the clip's own): "
           "mean %.1f mm, max %.1f mm\n", mean * 1e3, worst * 1e3, bmean * 1e3, bworst * 1e3);
    CHECK(mean < 0.1 && bmean < 0.01 && bworst < 0.06);   /* the joints' bones change length within a clip */
    free(bvh);
    CHECK(sm_request_set_int(r, SM_P_IN_PLACE, 1) == SM_OK && sm_ctx_generate(c, r, &on, 1) == SM_OK);
    CHECK(!memcmp(sm_motion_root(on, NULL), sm_motion_root(off, NULL), T * 3 * sizeof(float)));
    const float *j = sm_motion_joints(on, NULL);
    CHECK(j[(T - 1) * 66] == 0 && j[(T - 1) * 66 + 2] == 0 && j[(T - 1) * 66 + 1] == sm_motion_root(off, NULL)[(T - 1) * 3 + 1]);
    uint32_t nc; const int16_t *codes = sm_motion_codes(on, &nc);
    CHECK(sm_ctx_decode_codes(c, r, codes, nc / 6, off) == SM_OK && !memcmp(sm_motion_joints(off, NULL), j, (size_t)T * 66 * sizeof(float)));
    CHECK(sm_ctx_decode_codes(c, NULL, codes, nc / 6, off) == SM_OK && sm_motion_joints(off, NULL)[(T - 1) * 66] != 0);
    sm_motion_free(on); sm_motion_free(off); sm_request_free(r); sm_ctx_free(c);
}

/* after reserve, nothing allocates: generation with variants and a prefix, the step API, preview, encode, decode */
static void test_no_alloc(const sm_model *m) {
    sm_ctx *c = NULL; sm_request *r = NULL; sm_motion *o[4] = {0}; sm_text *t = NULL; uint32_t d, tot;
    CHECK(sm_ctx_create(m, &c) == SM_OK && sm_request_create(m, &r) == SM_OK && sm_text_create(m, &t) == SM_OK);
    for (int i = 0; i < 4; i++) CHECK(sm_motion_create(m, &o[i]) == SM_OK && sm_motion_reserve(o[i], 10) == SM_OK);
    CHECK(sm_ctx_reserve(c, 10, 4) == SM_OK && sm_request_set_text(r, "a person runs", -1) == SM_OK);
    CHECK(sm_request_set_float(r, SM_P_SECONDS, 2) == SM_OK && sm_ctx_generate(c, r, o, 1) == SM_OK);   /* the prefix source */
    const long before = atomic_load(&n_allocs);
    CHECK(sm_request_set_prefix(r, o[0]) == SM_OK && sm_request_set_float(r, SM_P_SECONDS, 9.7) == SM_OK);
    CHECK(sm_ctx_generate(c, r, o, 4) == SM_OK && sm_ctx_begin(c, r, o, 2) == SM_OK);
    while (sm_ctx_step(c, &d, &tot) == SM_OK) CHECK(sm_ctx_preview(c) == SM_OK);
    CHECK(sm_ctx_finish(c) == SM_OK && sm_ctx_encode_text(c, "a person sits down", -1, t) == SM_OK);
    CHECK(sm_ctx_decode_codes(c, NULL, sm_motion_codes(o[1], NULL), 97, o[2]) == SM_OK);
    printf("allocations after reserve: %ld\n", atomic_load(&n_allocs) - before);
    CHECK(atomic_load(&n_allocs) == before && sm_motion_frames(o[3]) == 194);
    CHECK(sm_request_set_float(r, SM_P_SECONDS, 9.8) == SM_OK && sm_ctx_generate(c, r, o, 1) == SM_E_ARG);
    sm_text_free(t); for (int i = 0; i < 4; i++) sm_motion_free(o[i]);
    sm_request_free(r); sm_ctx_free(c);
}


/* chained clips (2 s, then 1 s segments, each from the last 0.3 s of the one before): acceleration at the seams vs
   inside, for the segments' own joints played one after another and for all their codes decoded at once */
static void test_seams(const sm_model *m) {
    static const char *prompts[3] = {"a person walks forward", "a person keeps walking forward at a steady pace",
                                     "a person jogs in a circle"};
    enum { SEGS = 4, T_MAX = 40 + 20 * (SEGS - 1), N = 3 * 3 };
    sm_ctx *c = NULL; sm_request *r = NULL; sm_motion *seg = NULL, *all = NULL; static float J[T_MAX * 66];
    static int16_t codes[T_MAX * 3]; static double acc[2][2][N * T_MAX]; size_t idx[2][2] = {{0}};
    CHECK(sm_ctx_create(m, &c) == SM_OK && sm_request_create(m, &r) == SM_OK);
    CHECK(sm_motion_create(m, &seg) == SM_OK && sm_motion_create(m, &all) == SM_OK);
    for (int clip = 0; clip < N; clip++) {
        uint32_t T = 0, L = 0, cuts[SEGS], n;
        sm_request_reset(r); sm_request_set_text(r, prompts[clip / 3], -1); sm_request_set_int(r, SM_P_SEED, clip % 3);
        for (int s = 0; s < SEGS; s++) {
            CHECK(sm_request_set_float(r, SM_P_SECONDS, s ? 1 : 2) == SM_OK && sm_ctx_generate(c, r, &seg, 1) == SM_OK);
            cuts[s] = T;
            memcpy(J + (size_t)T * 66, sm_motion_joints(seg, NULL), sizeof(float) * 66 * sm_motion_frames(seg));
            const int16_t *sc = sm_motion_codes(seg, &n);
            memcpy(codes + L * 6, sc, sizeof(int16_t) * n);
            T += sm_motion_frames(seg); L += n / 6;
            CHECK(sm_request_set_prefix(r, seg) == SM_OK);
        }
        CHECK(sm_ctx_decode_codes(c, NULL, codes, L, all) == SM_OK && sm_motion_frames(all) == T);
        for (int way = 0; way < 2; way++)
            for (uint32_t t = 1; t + 1 < T; t++) {
                const float *p = (way ? sm_motion_joints(all, NULL) : J) + (size_t)t * 66; double a = 0; int at = 0;
                for (int k = 0; k < 66; k += 3)
                    a += sqrt(pow(p[k + 66] - 2 * p[k] + p[k - 66], 2) + pow(p[k + 67] - 2 * p[k + 1] + p[k - 65], 2) +
                              pow(p[k + 68] - 2 * p[k + 2] + p[k - 64], 2));
                for (int s = 1; s < SEGS; s++) at |= t + 2 >= cuts[s] && t <= cuts[s] + 1;
                acc[way][at][idx[way][at]++] = a * 400 / 22;   /* m/s^2 at 20 fps, mean over joints */
            }
    }
    double med[2][2];
    for (int w = 0; w < 2; w++)
        for (int at = 0; at < 2; at++) qsort(acc[w][at], idx[w][at], sizeof(double), cmp_double), med[w][at] = acc[w][at][idx[w][at] / 2];
    printf("seams (1 s segments, median acceleration, m/s^2): one decode %.2f at seams vs %.2f inside; own joints %.2f vs %.2f\n",
           med[1][1], med[1][0], med[0][1], med[0][0]);
    CHECK(med[1][1] < 1.25 * med[1][0] && med[0][1] < 2 * med[0][0]);
    sm_motion_free(seg); sm_motion_free(all); sm_request_free(r); sm_ctx_free(c);
}

static int has_word(const sm_model *m, const char *w) {
    for (uint32_t i = 0; sm_model_vocab(m, i); i++) if (!strcmp(sm_model_vocab(m, i), w)) return 1;
    return 0;
}

/* the vocabulary: sorted, lowercase, NULL past the end; with a text encoder every word is known */
static void test_vocab(const sm_model *m, int full) {
    uint32_t n = 0, bad = 0, unknown = 0, u; const char *w, *prev = ""; size_t need;
    for (; (w = sm_model_vocab(m, n)); prev = w, n++) {
        bad += !*w || strcmp(prev, w) >= 0;
        for (const char *p = w; *p; p++) bad += (*p < 'a' || *p > 'z') && (unsigned char)*p < 0x80;
    }
    CHECK(!bad && !sm_model_vocab(m, n) && !sm_model_vocab(m, UINT32_MAX) && !sm_model_vocab(NULL, 0));
    if (!full) return;
    sm_ctx *c; sm_text *t;
    CHECK(sm_ctx_create(m, &c) == SM_OK && sm_text_create(m, &t) == SM_OK);
    for (uint32_t i = 0; i < n; i += 37) {
        CHECK(sm_ctx_encode_text(c, sm_model_vocab(m, i), -1, t) == SM_OK);
        CHECK(sm_text_unknown_words(t, NULL, 0, &need, &u) == SM_OK); unknown += u;
    }
    CHECK(n > 1000 && !unknown && has_word(m, "walk") && has_word(m, "jumps") && !has_word(m, "kaleidoscope"));
    printf("vocabulary: %u words (%s .. %s)\n", n, sm_model_vocab(m, 0), sm_model_vocab(m, n - 1));
    sm_text_free(t); sm_ctx_free(c);
}

int main(void) {
    sm_model *m = load("tests/data", "test.gguf");
    CHECK(m != NULL); if (!m) return CHECK_DONE();
    test_misuse(m); test_messages(m); test_params(m); test_decode(m); test_edit_api(m); test_model_file("tests/data/test.gguf");
    test_vocab(m, 0);
    sm_motion *mo; CHECK(sm_motion_create(m, &mo) == SM_OK && sm_generate_simple(m, "a", 2, 0, mo, NULL, 0) == SM_E_UNSUPPORTED);
    sm_motion_free(mo); sm_model_free(m);
    const char *dir = getenv("SM_MODEL_DIR");
    if (!dir || !(m = load(dir, "SprongMotion-14M-v1.0-Q8_0.gguf"))) { printf("test_api: no model in $SM_MODEL_DIR, generation skipped\n"); return CHECK_DONE(); }
    printf("model: %s, %d k generator parameters\n", sm_model_id(m), sm_model_info(m, SM_I_PARAMS_K));
    CHECK(sm_model_info(m, SM_I_PARAMS_K) > 1000);
    test_generate(m); test_post(m); test_no_alloc(m); test_seams(m); test_vocab(m, 1);
    sm_model_free(m);
    return CHECK_DONE();
}
