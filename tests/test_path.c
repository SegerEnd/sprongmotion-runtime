/* Root paths: checks, fuzzing, unchanged output without one; with a model the warp, with $SM_GATE_MODEL the gate. */
#include "../src/sm_internal.h"
#include "check.h"
#include <stdlib.h>

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

static uint64_t hash(const sm_motion *mo) {   /* joints, rotations, root, contacts, offsets */
    uint32_t n; uint64_t h = 0xcbf29ce484222325ull; const float *f;
    f = sm_motion_joints(mo, &n); h = fnv(h, f, n * 4u); f = sm_motion_rotations(mo, &n); h = fnv(h, f, n * 4u);
    f = sm_motion_root(mo, &n); h = fnv(h, f, n * 4u); const uint8_t *u = sm_motion_contacts(mo, &n); h = fnv(h, u, n);
    f = sm_motion_offsets(mo, &n); return fnv(h, f, n * 4u);
}

static int finite(const sm_motion *mo) {
    uint32_t n, m; const float *j = sm_motion_joints(mo, &n), *q = sm_motion_rotations(mo, &m);
    for (uint32_t i = 0; i < n; i++) if (!isfinite(j[i])) return 0;
    for (uint32_t i = 0; i < m; i++) if (!isfinite(q[i])) return 0;
    return n > 0 && isfinite(sm_motion_path_error(mo)) && sm_motion_path_error(mo) >= 0;
}

static int16_t codes[60 * 6];

/* rejected points name themselves; a rejected call keeps the path set before */
static void test_points(const sm_model *m) {
    sm_request *r = NULL; sm_ctx *c = NULL; sm_motion *a = NULL, *b = NULL;
    CHECK(sm_request_create(m, &r) == SM_OK && sm_ctx_create(m, &c) == SM_OK);
    CHECK(sm_motion_create(m, &a) == SM_OK && sm_motion_create(m, &b) == SM_OK);
    float p[1025 * 4];
    for (int i = 0; i < 1025; i++) p[4 * i] = NAN, p[4 * i + 1] = 0.01f * (float)i, p[4 * i + 2] = 0.1f * (float)i, p[4 * i + 3] = NAN;
    const struct { int i, k; float v; uint32_t n; const char *msg; } bad[] = {
        {0, 0, 0, 1, "path points = 1 is out of range 2 .. 1024"}, {0, 0, 0, 1025, "path points = 1025 is out of range 2 .. 1024"},
        {1, 1, NAN, 3, "path[1].x = nan is out of range -1000 .. 1000"}, {2, 2, 1e6f, 3, "path[2].z = 1000000 is out of range -1000 .. 1000"},
        {1, 3, INFINITY, 3, "path[1].heading = inf is out of range -1000 .. 1000"}, {0, 0, -1, 3, "path[0].t = -1 is out of range 0 .. 600"},
        {2, 0, 601, 3, "path[2].t = 601 is out of range 0 .. 600"}, {1, 0, 0, 3, "path[1].t = 0 is not after the time before it"},
    };
    CHECK(sm_request_set_path(r, p, 3) == SM_OK);
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        float q[3 * 4]; memcpy(q, p, sizeof q);
        if (bad[i].n == 3) q[4 * bad[i].i + bad[i].k] = bad[i].v;
        CHECK(sm_request_set_path(r, q, bad[i].n) == SM_E_ARG && !strcmp(sm_request_last_error(r), bad[i].msg));
        if (strcmp(sm_request_last_error(r), bad[i].msg)) fprintf(stderr, "  got \"%s\"\n", sm_request_last_error(r));
    }
    p[4] = 2, p[8] = 2;
    CHECK(sm_request_set_path(r, p, 3) == SM_E_ARG && strstr(sm_request_last_error(r), "path[2].t = 2 is not after"));
    p[8] = 7;
    CHECK(sm_request_set_path(r, NULL, 2) == SM_E_ARG && sm_request_set_path(NULL, p, 2) == SM_E_ARG);
    CHECK(sm_request_set_path(r, p, 3) == SM_OK && sm_ctx_decode_codes(c, r, codes, 60, a) == SM_E_ARG);
    CHECK(!strcmp(sm_ctx_last_error(c), "path[2].t = 7 is out of range 0 .. 6"));
    CHECK(sm_request_set_path(r, p, 1024) == SM_OK && sm_request_set_path(r, NULL, 0) == SM_OK);
    CHECK(sm_model_info(m, SM_I_PATH_INPUT) == 0 && sm_model_info(m, SM_I_PATH_INPUT + 1) == -1);
    sm_param pw = sm_param_from_name("path_warp"); double lo, hi, d; int32_t is_int;
    CHECK(pw == SM_P_PATH_WARP && sm_param_bounds(m, pw, &lo, &hi, &d, &is_int) == SM_OK && d == 1 && is_int);
    CHECK(sm_param_name(12) == NULL && strstr(sm_api_schema_json(), "\"name\":\"path\""));
    sm_motion_free(a); sm_motion_free(b); sm_ctx_free(c); sm_request_free(r);
}

/* without a path the output matches the stored hashes; a set and cleared path, and a path with the warp off, change
   nothing */
static void test_golden(const sm_model *m) {
    static const uint64_t golden[4] = {0x95ecc839c38c7862ull, 0x82a6180c9827e58bull, 0xf78669154625cef1ull,
                                       0x827c89241ade4e14ull};   /* foot lock 0/1 x in place 0/1 */
    sm_request *r = NULL; sm_ctx *c = NULL; sm_motion *mo = NULL; uint32_t n;
    CHECK(sm_request_create(m, &r) == SM_OK && sm_ctx_create(m, &c) == SM_OK && sm_motion_create(m, &mo) == SM_OK);
    const float p[3 * 4] = {NAN, 0, 0, NAN, NAN, 1, 2, NAN, NAN, -1, 5, NAN};
    for (int k = 0; k < 4; k++) {
        sm_request_set_int(r, SM_P_FOOT_LOCK, k & 1), sm_request_set_int(r, SM_P_IN_PLACE, k >> 1);
        CHECK(sm_ctx_decode_codes(c, r, codes, 60, mo) == SM_OK && hash(mo) == golden[k] && sm_motion_path_error(mo) == -1);
        CHECK(sm_request_set_path(r, p, 3) == SM_OK && sm_request_set_int(r, SM_P_PATH_WARP, 0) == SM_OK);
        CHECK(sm_ctx_decode_codes(c, r, codes, 60, mo) == SM_OK && hash(mo) == golden[k] && sm_motion_path_error(mo) > 0);
        CHECK(sm_request_set_int(r, SM_P_PATH_WARP, 1) == SM_OK && sm_ctx_decode_codes(c, r, codes, 60, mo) == SM_OK);
        CHECK(hash(mo) != golden[k] && sm_request_set_path(r, NULL, 0) == SM_OK);
        CHECK(sm_ctx_decode_codes(c, r, codes, 60, mo) == SM_OK && hash(mo) == golden[k]);
    }
    CHECK(sm_motion_speed(mo, &n) && n == 120 && sm_motion_speed(NULL, &n) == NULL && n == 0);
    CHECK(sm_motion_path_error(NULL) == -1);
    sm_motion_free(mo); sm_ctx_free(c); sm_request_free(r);
}


/* random, duplicate, collinear, closed, far and timed points: always a finite motion */
static void test_fuzz(const sm_model *m) {
    static float p[SM_MAX_PATH * 4];
    sm_request *r = NULL; sm_ctx *c = NULL; sm_motion *mo = NULL; unsigned s = 7; int bad = 0, runs = 0;
    CHECK(sm_request_create(m, &r) == SM_OK && sm_ctx_create(m, &c) == SM_OK && sm_motion_create(m, &mo) == SM_OK);
    static const uint32_t sizes[] = {2, 2, 3, 4, 5, 9, 40, 1024};
    for (int kind = 0; kind < 8; kind++)
        for (size_t z = 0; z < sizeof sizes / sizeof sizes[0]; z++, runs++) {
            const uint32_t n = sizes[z]; double t = 0;
            for (uint32_t i = 0; i < n; i++) {
                float *q = p + 4 * i;
                const double a = 2 * 3.14159265 * i / (n - 1.0);
                q[0] = NAN, q[3] = rnd(&s) < 0.5 ? NAN : (float)(rnd(&s) * 6);
                q[1] = (float)(kind == 0 ? rnd(&s) * 20 - 10 : kind == 1 ? 3 : kind == 2 ? 0.01 * i : kind == 3 ? 4 * sin(a)
                       : kind == 4 ? 999 * (rnd(&s) * 2 - 1) : kind == 5 ? 1e-7 * (i % 3) : kind == 6 ? (i % 2) * 2.0 : i / 10);
                q[2] = (float)(kind == 3 ? 4 - 4 * cos(a) : kind == 7 ? 0 : kind == 4 ? 999 * (rnd(&s) * 2 - 1)
                       : kind == 2 ? 0 : rnd(&s) * 5);
                if (kind >= 5 && rnd(&s) < 0.3 && (t += 0.01 + rnd(&s) * 5.9 / n) < 6) q[0] = (float)t;
            }
            sm_request_set_int(r, SM_P_LOOP, kind == 3);
            const sm_status st = sm_request_set_path(r, p, n);
            if (st != SM_OK) { bad++; fprintf(stderr, "fuzz %d/%u: %s\n", kind, n, sm_request_last_error(r)); continue; }
            if (sm_ctx_decode_codes(c, r, codes, 60, mo) != SM_OK || !finite(mo)) bad++;
        }
    printf("fuzzed paths: %d decodes, %d failed or not finite\n", runs, bad);
    CHECK(bad == 0);
    sm_motion_free(mo); sm_ctx_free(c); sm_request_free(r);
}

static double travel(const sm_motion *mo) {   /* the smoothed root's arc length */
    uint32_t n; const float *v = sm_motion_speed(mo, &n); double s = 0;
    for (uint32_t t = 0; t < n; t++) s += v[t] / 20;
    return s;
}

static double heading_at(int shape, double S) { return shape == 1 ? 0.5236 * sin(2 * 3.14159265358979 * S / 8) : shape == 2 ? S / 4 : 0; }

/* straight, gentle S (+-30 deg, 8 m) or a 4 m circle to the left, length L: a point every 0.25 m */
static uint32_t shape_points(int shape, double L, float *p) {
    double x = 0, z = 0, S = 0; uint32_t n = 1; int i = 0;
    p[0] = NAN, p[1] = 0, p[2] = 0, p[3] = NAN;
    while (S < L - 1e-9) {
        const double st = fmin(0.005, L - S), h = heading_at(shape, S + st / 2);
        x += st * sin(h), z += st * cos(h), S += st;
        if (++i % 50 && S < L - 1e-9) continue;
        float *q = p + 4 * n++; q[0] = NAN, q[1] = (float)x, q[2] = (float)z, q[3] = NAN;
    }
    if (!shape) memcpy(p + 4, p + 4 * (n - 1), 4 * sizeof(float)), n = 2;
    return n;
}

static double dist_to(const double *g, uint32_t lo, uint32_t hi, const float *q) {   /* point to polyline g[lo..hi] */
    double best = 1e9;
    for (uint32_t t = lo; t < hi; t++) {
        const double ax = g[2 * t], az = g[2 * t + 1], bx = g[2 * t + 2] - ax, bz = g[2 * t + 3] - az, l = bx * bx + bz * bz;
        const double u = l > 0 ? fmin(1, fmax(0, ((q[1] - ax) * bx + (q[2] - az) * bz) / l)) : 0;
        best = fmin(best, hypot(ax + u * bx - q[1], az + u * bz - q[2]));
    }
    return best;
}

/* the largest distance from a point to the warped root's path part P(S(t)) = root - R(dpsi) (g - g smoothed), with g
   the unwarped root and dpsi the hips' added yaw (a timed point: at its time +-1 frame); smoothed: to the warped root
   smoothed as the warp smooths, which also cuts inside a curve by about sigma^2 / 2r */
static double off_path(const sm_motion *w, const sm_motion *plain, const float *p, uint32_t n, int smoothed) {
    static double g[2 * 4000], P[2 * 4000]; static float sp[4000];
    const uint32_t T = sm_motion_frames(w); double worst = 0;
    const float *a = sm_motion_joints(plain, NULL), *b = sm_motion_joints(w, NULL);
    sm_root_smooth(smoothed ? b : a, T, 20, smoothed ? P : g, sp);
    for (uint32_t t = 0; t < T && !smoothed; t++) {
        const float *x = a + (size_t)t * 66, *y = b + (size_t)t * 66;
        const double d = atan2(-(y[5] - y[8]), y[3] - y[6]) - atan2(-(x[5] - x[8]), x[3] - x[6]), c = cos(d), s = sin(d);
        const double dx = x[0] - g[2 * t], dz = x[2] - g[2 * t + 1];
        P[2 * t] = y[0] - (dx * c + dz * s), P[2 * t + 1] = y[2] - (-dx * s + dz * c);
    }
    for (uint32_t i = 0; i < n; i++) {
        const int timed = !isnan(p[4 * i]);
        const uint32_t lo = timed ? (uint32_t)fmax(0, p[4 * i] * 20 - 1) : 0, hi = timed ? (uint32_t)fmin(T - 1, p[4 * i] * 20 + 1) : T - 1;
        worst = fmax(worst, dist_to(P, lo, hi, p + 4 * i));
    }
    return worst;
}

static double max_diff(const sm_motion *a, const sm_motion *b) {
    uint32_t n; const float *x = sm_motion_joints(a, &n), *y = sm_motion_joints(b, NULL); double d = 0;
    for (uint32_t i = 0; i < n; i++) d = fmax(d, fabs((double)x[i] - y[i]));
    return d;
}

static void test_warp(const sm_model *m) {
    static float p[SM_MAX_PATH * 4];
    static double g[2 * 400]; static float sp[400];
    sm_request *r = NULL; sm_ctx *c = NULL; sm_motion *base = NULL, *w = NULL, *x = NULL;
    CHECK(sm_request_create(m, &r) == SM_OK && sm_ctx_create(m, &c) == SM_OK);
    CHECK(sm_motion_create(m, &base) == SM_OK && sm_motion_create(m, &w) == SM_OK && sm_motion_create(m, &x) == SM_OK);
    sm_request_set_text(r, "a person keeps walking forward", -1), sm_request_set_float(r, SM_P_SECONDS, 6);
    sm_request_set_int(r, SM_P_SEED, 1);
    CHECK(sm_ctx_generate(c, r, &base, 1) == SM_OK);
    const uint32_t T = sm_motion_frames(base); const double L = travel(base);
    double worst = 0, sm[3];
    for (int shape = 0; shape < 3; shape++) {   /* on the path */
        const uint32_t n = shape_points(shape, L, p);
        CHECK(sm_request_set_path(r, p, n) == SM_OK && sm_ctx_generate(c, r, &w, 1) == SM_OK);
        worst = fmax(worst, off_path(w, base, p, n, 0)), sm[shape] = off_path(w, base, p, n, 1);
        CHECK(!memcmp(sm_motion_codes(w, NULL), sm_motion_codes(base, NULL), sizeof(int16_t) * 6 * T / 2));
    }
    const float timed[3 * 4] = {NAN, 0, 0, NAN, 3, 0, (float)(0.55 * L), NAN, NAN, 0, (float)L, NAN};
    CHECK(sm_request_set_path(r, timed, 3) == SM_OK && sm_ctx_generate(c, r, &w, 1) == SM_OK);
    const double at_time = off_path(w, base, timed, 3, 0);
    sm_root_smooth(sm_motion_joints(base, NULL), T, 20, g, sp);   /* its own path, a point per frame */
    uint32_t n = 0;
    for (uint32_t t = 0; t < T; t++, n++) p[4 * n] = NAN, p[4 * n + 1] = (float)g[2 * t], p[4 * n + 2] = (float)g[2 * t + 1], p[4 * n + 3] = NAN;
    CHECK(sm_request_set_path(r, p, n) == SM_OK && sm_ctx_generate(c, r, &w, 1) == SM_OK);
    const double own = max_diff(w, base);
    printf("path: on the path within %.2f mm (straight, S, r 4 m), smoothed root %.0f / %.0f / %.0f mm; a timed point %.2f mm; "
           "its own path moves joints %.2f mm\n", worst * 1e3, sm[0] * 1e3, sm[1] * 1e3, sm[2] * 1e3, at_time * 1e3, own * 1e3);
    CHECK(worst <= 0.02 && at_time <= 0.02 && own <= 1e-3 && sm[0] <= 0.02 && sm[1] <= 0.04 && sm[2] <= 0.04);

    const uint32_t ns = shape_points(1, L, p);   /* the warp off: the motion of no path, the same error */
    CHECK(sm_request_set_path(r, p, ns) == SM_OK && sm_ctx_generate(c, r, &w, 1) == SM_OK);
    CHECK(sm_request_set_int(r, SM_P_PATH_WARP, 0) == SM_OK && sm_ctx_generate(c, r, &x, 1) == SM_OK);
    CHECK(hash(x) == hash(base) && sm_motion_path_error(x) == sm_motion_path_error(w) && sm_motion_path_error(w) > 0.1f);
    CHECK(sm_request_set_int(r, SM_P_PATH_WARP, 1) == SM_OK);
    CHECK(sm_ctx_decode_codes(c, r, sm_motion_codes(w, NULL), T / 2, x) == SM_OK && hash(x) == hash(w));   /* decode only */
    sm_motion *o[2] = {x, base}; uint32_t d, tot; sm_status s;   /* step API, variants */
    CHECK(sm_ctx_begin(c, r, o, 2) == SM_OK);
    while ((s = sm_ctx_step(c, &d, &tot)) == SM_OK) CHECK(sm_ctx_preview(c) == SM_OK);
    CHECK(s == SM_DONE && sm_ctx_finish(c) == SM_OK && hash(x) == hash(w));

    const char *texts[2] = {"a person keeps walking forward", "a person walks forward slowly"};   /* timeline time */
    const double sec[2] = {3, 3}; const int32_t seeds[2] = {1, 2}; uint32_t first[3];
    CHECK(sm_request_set_path(r, NULL, 0) == SM_OK && sm_ctx_generate_blocks(c, r, 2, texts, NULL, sec, seeds, NULL, first, w) == SM_OK);
    const uint32_t nb = shape_points(2, travel(w), p);
    CHECK(sm_request_set_path(r, p, nb) == SM_OK && sm_ctx_generate_blocks(c, r, 2, texts, NULL, sec, seeds, NULL, first, x) == SM_OK);
    const int16_t *keep[2] = {sm_motion_codes(x, NULL), sm_motion_codes(x, NULL) + first[1] / 2 * 6};
    CHECK(off_path(x, w, p, nb, 0) < 0.02 && !memcmp(sm_motion_codes(x, NULL), sm_motion_codes(w, NULL), sizeof(int16_t) * 6 * first[2] / 2));
    CHECK(sm_motion_copy(base, x) == SM_OK && sm_ctx_generate_blocks(c, r, 2, NULL, NULL, sec, seeds, keep, NULL, x) == SM_OK);
    CHECK(hash(x) == hash(base) && sm_motion_path_error(x) == sm_motion_path_error(base));

    CHECK(sm_request_set_prefix(r, x) == SM_OK);   /* a prefix: the path in the previous clip's frame */
    const float *end = sm_motion_root(x, NULL) + 3 * (sm_motion_frames(x) - 1);
    const float pre[2 * 4] = {NAN, end[0], end[2], NAN, NAN, end[0] + 3, end[2] + 3, NAN};
    CHECK(sm_request_set_path(r, NULL, 0) == SM_OK && sm_request_set_float(r, SM_P_SECONDS, 3) == SM_OK);
    CHECK(sm_ctx_generate(c, r, &base, 1) == SM_OK);
    CHECK(sm_request_set_path(r, pre, 2) == SM_OK && sm_ctx_generate(c, r, &w, 1) == SM_OK);
    CHECK(finite(w) && off_path(w, base, pre, 2, 0) < 0.02);
    CHECK(sm_request_set_prefix(r, NULL) == SM_OK && sm_request_set_int(r, SM_P_LOOP, 1) == SM_OK);   /* a closed path */
    CHECK(sm_request_set_path(r, NULL, 0) == SM_OK && sm_request_set_text(r, "a person walks in a circle", -1) == SM_OK);
    CHECK(sm_ctx_generate(c, r, &base, 1) == SM_OK && sm_motion_copy(w, base) == SM_OK);
    const double C = travel(w); n = 0;
    for (int i = 0; i <= 32; i++, n++) {
        const double a = 2 * 3.14159265358979 * i / 32, rad = C / (2 * 3.14159265358979);
        p[4 * n] = NAN, p[4 * n + 1] = i < 32 ? (float)(rad - rad * cos(a)) : 0, p[4 * n + 2] = i < 32 ? (float)(rad * sin(a)) : 0, p[4 * n + 3] = NAN;
    }
    CHECK(sm_request_set_path(r, p, n) == SM_OK && sm_ctx_generate(c, r, &w, 1) == SM_OK && finite(w));
    CHECK(off_path(w, base, p, n, 0) < 0.02);
    sm_motion_free(base); sm_motion_free(w); sm_motion_free(x); sm_ctx_free(c); sm_request_free(r);
}

static const int FEET[4] = {7, 10, 8, 11};

/* mean horizontal speed of ankles (toe 0) or toes over the frames where the unwarped clip has that joint low (5 cm of
   its lowest) and slow (0.6 m/s), cm/s */
static double slide(const float *base, const float *j, uint32_t T, int toe) {
    double s = 0, lo[4]; int n = 0;
    for (int k = 0; k < 4; k++) { lo[k] = 1e9; for (uint32_t t = 0; t < T; t++) lo[k] = fmin(lo[k], base[(t * 22 + (uint32_t)FEET[k]) * 3 + 1]); }
    for (uint32_t t = 0; t + 2 < T; t++)
        for (int k = toe; k < 4; k += 2) {
            int planted = 1;
            for (uint32_t u = t; u <= t + 1; u++) {
                const float *a = base + (u * 22 + (uint32_t)FEET[k]) * 3, *b = a + 66;
                planted &= a[1] < lo[k] + 0.05 && hypot(b[0] - a[0], b[2] - a[2]) * 20 < 0.6;
            }
            if (!planted) continue;
            const float *a = j + (t * 22 + (uint32_t)FEET[k]) * 3, *b = a + 66;
            s += hypot(b[0] - a[0], b[2] - a[2]) * 2000, n++;
        }
    return n ? s / n : 0;
}


static void jerk(const float *j, uint32_t T, double *med, double *p95) {   /* world joints, mean over joints, m/s^3 */
    static double v[4000];
    for (uint32_t t = 0; t + 3 < T; t++) {
        double s = 0;
        for (uint32_t i = 0; i < 66; i += 3) {
            double d2 = 0;
            for (uint32_t k = i; k < i + 3; k++) {
                const double d = j[(t + 3) * 66 + k] - 3.0 * j[(t + 2) * 66 + k] + 3.0 * j[(t + 1) * 66 + k] - j[t * 66 + k];
                d2 += d * d;
            }
            s += sqrt(d2);
        }
        v[t] = s / 22 * 8000;
    }
    qsort(v, T - 3, sizeof(double), cmp_double);
    *med = v[(T - 4) / 2], *p95 = v[(size_t)(0.95 * (T - 4))];
}

/* the foot gate: 4 seeds, 6 s, at the prompt's own speed; ankle and toe slide and jerk median and p95 each within
   +20 % of the unwarped clip, both foot-locked */
static void test_gate(const sm_model *m) {
    enum { SEEDS = 4 };
    static float p[SM_MAX_PATH * 4];
    sm_request *r = NULL; sm_ctx *c = NULL; sm_motion *base[SEEDS] = {0}, *w = NULL;
    CHECK(sm_request_create(m, &r) == SM_OK && sm_ctx_create(m, &c) == SM_OK && sm_motion_create(m, &w) == SM_OK);
    for (int i = 0; i < SEEDS; i++) CHECK(sm_motion_create(m, &base[i]) == SM_OK);
    sm_request_set_text(r, "a person keeps walking forward", -1), sm_request_set_float(r, SM_P_SECONDS, 6);
    sm_request_set_int(r, SM_P_SEED, 1);
    CHECK(sm_ctx_generate(c, r, base, SEEDS) == SM_OK);
    static const char *const names[3] = {"straight", "gentle S", "circle r 4 m"};
    for (int shape = 0; shape < 3; shape++) {
        double v[2][4] = {{0}};   /* unwarped, warped: ankle, toe, jerk median, p95 */
        for (int i = 0; i < SEEDS; i++) {
            const uint32_t n = shape_points(shape, travel(base[i]), p), T = sm_motion_frames(base[i]);
            CHECK(sm_request_set_int(r, SM_P_SEED, 1 + i) == SM_OK && sm_request_set_path(r, p, n) == SM_OK);
            CHECK(sm_ctx_generate(c, r, &w, 1) == SM_OK);
            const float *b = sm_motion_joints(base[i], NULL), *j = sm_motion_joints(w, NULL);
            for (int k = 0; k < 2; k++) {
                const float *x = k ? j : b; double med, p95;
                jerk(x, T, &med, &p95);
                v[k][0] += slide(b, x, T, 0), v[k][1] += slide(b, x, T, 1), v[k][2] += med, v[k][3] += p95;
            }
        }
        printf("gate %-12s ankle %+3.0f %%, toe %+3.0f %%, jerk median %+3.0f %%, p95 %+3.0f %%\n", names[shape],
               100 * (v[1][0] / v[0][0] - 1), 100 * (v[1][1] / v[0][1] - 1), 100 * (v[1][2] / v[0][2] - 1), 100 * (v[1][3] / v[0][3] - 1));
        for (int k = 0; k < 4; k++) CHECK(v[1][k] <= 1.2 * v[0][k]);
        CHECK(sm_request_set_path(r, NULL, 0) == SM_OK);
    }
    for (int i = 0; i < SEEDS; i++) sm_motion_free(base[i]);
    sm_motion_free(w); sm_ctx_free(c); sm_request_free(r);
}

int main(void) {
    for (int i = 0; i < 60 * 6; i++) codes[i] = (int16_t)(i % 6 == 5 ? -1 : (i * 37 + i / 6 * 11) % 512);
    char path[512], err[128]; sm_model *m = NULL;
    CHECK(sm_model_load_file("tests/data/test.gguf", NULL, &m, err, sizeof err) == SM_OK);
    if (!m) return CHECK_DONE();
    test_points(m); test_golden(m); test_fuzz(m);
    sm_model_free(m);
    const char *dir = getenv("SM_MODEL_DIR");
    snprintf(path, sizeof path, "%s/SprongMotion-14M-v1.0-Q8_0.gguf", dir ? dir : "tests/data");
    if (!dir || sm_model_load_file(path, NULL, &m, err, sizeof err) != SM_OK) {
        printf("test_path: no model in $SM_MODEL_DIR, warp tests skipped\n");
        return CHECK_DONE();
    }
    test_warp(m);
    sm_model_free(m);
    const char *gate = getenv("SM_GATE_MODEL");   /* the model the gate was measured on */
    if (!gate || sm_model_load_file(gate, NULL, &m, err, sizeof err) != SM_OK) printf("test_path: no $SM_GATE_MODEL, gate skipped\n");
    else test_gate(m), sm_model_free(m);
    return CHECK_DONE();
}
