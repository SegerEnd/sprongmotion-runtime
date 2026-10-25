/* Root paths: checks, fuzzing, unchanged output without one; with a model the warp, with $SM_GATE_MODEL the gate. */
#include "../src/sm_internal.h"
#include "check.h"
#include <stdlib.h>

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

/* joints, rotations, root, contacts, offsets */
static uint64_t hash(sm_motion mo) {
    const size_t T = (size_t)mo.frame_count;
    uint64_t h = fnv(0xcbf29ce484222325ull, mo.joints, T * 66 * 4);
    h = fnv(h, mo.rotations, T * 88 * 4);
    h = fnv(h, mo.root, T * 3 * 4);
    h = fnv(h, mo.contacts, T * 4);
    return fnv(h, mo.offsets, 66 * 4);
}

static int finite(sm_motion mo) {
    for (int i = 0; i < mo.frame_count * 66; i++) if (!isfinite(mo.joints[i])) return 0;
    for (int i = 0; i < mo.frame_count * 88; i++) if (!isfinite(mo.rotations[i])) return 0;
    return mo.frame_count > 0 && isfinite(mo.path_error) && mo.path_error >= 0;
}

static sm_motion made(sm_motion mo) {
    if (!sm_is_motion_valid(mo)) fprintf(stderr, "  %s\n", sm_get_error());
    CHECK(sm_is_motion_valid(mo));
    return mo;
}

static int failed_with(sm_motion mo, const char *msg) {
    if (!sm_is_motion_valid(mo) && !strcmp(sm_get_error(), msg)) return 1;
    fprintf(stderr, "  got \"%s\", want \"%s\"\n", sm_get_error(), msg);
    sm_unload_motion(mo);
    return 0;
}

static sm_options with_path(sm_options o, const float *p, int n) {
    o.path = (const sm_path_point *)p;
    o.path_count = n;
    return o;
}

static int16_t codes[60 * 6];

/* rejected points name themselves */
static void test_points(sm_model m) {
    float p[1025 * 4];
    for (int i = 0; i < 1025; i++) p[4 * i] = NAN, p[4 * i + 1] = 0.01f * (float)i, p[4 * i + 2] = 0.1f * (float)i, p[4 * i + 3] = NAN;
    const struct { int i, k; float v; int n; const char *msg; } bad[] = {
        {0, 0, 0, 1, "path_count = 1 is out of range 2 .. 1024"}, {0, 0, 0, 1025, "path_count = 1025 is out of range 2 .. 1024"},
        {1, 1, NAN, 3, "path[1].x = nan is out of range -1000 .. 1000"}, {2, 2, 1e6f, 3, "path[2].z = 1000000 is out of range -1000 .. 1000"},
        {1, 3, INFINITY, 3, "path[1].heading = inf is out of range -1000 .. 1000"}, {0, 0, -1, 3, "path[0].time = -1 is out of range 0 .. 600"},
        {2, 0, 601, 3, "path[2].time = 601 is out of range 0 .. 600"}, {1, 0, 0, 3, "path[1].time = 0 is not after the time before it"},
    };
    const sm_options o = sm_default_options(m);
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        float q[1025 * 4];
        memcpy(q, p, sizeof q);
        if (bad[i].n == 3) q[4 * bad[i].i + bad[i].k] = bad[i].v;
        CHECK(failed_with(sm_decode_codes(m, codes, 60, with_path(o, q, bad[i].n)), bad[i].msg));
    }
    p[4] = 2, p[8] = 2;
    CHECK(failed_with(sm_decode_codes(m, codes, 60, with_path(o, p, 3)), "path[2].time = 2 is not after the time before it"));
    p[8] = 7;
    CHECK(failed_with(sm_decode_codes(m, codes, 60, with_path(o, NULL, 2)), "path is NULL"));
    CHECK(failed_with(sm_decode_codes(m, codes, 60, with_path(o, p, 3)), "path[2].time = 7 is out of range 0 .. 6"));
    p[8] = 5;
    sm_motion mo = made(sm_decode_codes(m, codes, 60, with_path(o, p, 1024)));
    sm_unload_motion(mo);
    CHECK(!m.follows_path && o.path_warp);
}

/* without a path the output matches the stored hashes; a path with the warp off changes nothing */
static void test_golden(sm_model m) {
    static const uint64_t golden[4] = {0x95ecc839c38c7862ull, 0x82a6180c9827e58bull, 0xf78669154625cef1ull,
                                       0x827c89241ade4e14ull};   /* foot lock 0/1 x in place 0/1 */
    const float p[3 * 4] = {NAN, 0, 0, NAN, NAN, 1, 2, NAN, NAN, -1, 5, NAN};
    for (int k = 0; k < 4; k++) {
        sm_options o = sm_default_options(m);
        o.foot_lock = k & 1;
        o.in_place = k >> 1;
        sm_motion a = made(sm_decode_codes(m, codes, 60, o));
        CHECK(hash(a) == golden[k] && a.path_error == -1);
        o = with_path(o, p, 3);
        o.path_warp = false;
        sm_motion b = made(sm_decode_codes(m, codes, 60, o));
        CHECK(hash(b) == golden[k] && b.path_error > 0);
        o.path_warp = true;
        sm_motion c = made(sm_decode_codes(m, codes, 60, o));
        CHECK(hash(c) != golden[k] && a.speed && a.frame_count == 120);
        sm_unload_motion(a);
        sm_unload_motion(b);
        sm_unload_motion(c);
    }
}

/* random, duplicate, collinear, closed, far and timed points: always a finite motion */
static void test_fuzz(sm_model m) {
    static float p[SM_MAX_PATH * 4];
    unsigned s = 7;
    int bad = 0, runs = 0;
    static const uint32_t sizes[] = {2, 2, 3, 4, 5, 9, 40, 1024};
    for (int kind = 0; kind < 8; kind++)
        for (size_t z = 0; z < sizeof sizes / sizeof sizes[0]; z++, runs++) {
            const uint32_t n = sizes[z];
            double t = 0;
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
            sm_options o = with_path(sm_default_options(m), p, (int)n);
            o.loop = kind == 3;
            sm_motion mo = sm_decode_codes(m, codes, 60, o);
            if (!sm_is_motion_valid(mo)) fprintf(stderr, "fuzz %d/%u: %s\n", kind, n, sm_get_error());
            bad += !sm_is_motion_valid(mo) || !finite(mo);
            sm_unload_motion(mo);
        }
    printf("fuzzed paths: %d decodes, %d failed or not finite\n", runs, bad);
    CHECK(bad == 0);
}

/* the smoothed root's arc length */
static double travel(sm_motion mo) {
    double s = 0;
    for (int t = 0; t < mo.frame_count; t++) s += mo.speed[t] / 20;
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
static double off_path(sm_motion w, sm_motion plain, const float *p, uint32_t n, int smoothed) {
    static double g[2 * 4000], P[2 * 4000]; static float sp[4000];
    const uint32_t T = (uint32_t)w.frame_count; double worst = 0;
    const float *a = plain.joints, *b = w.joints;
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

static double max_diff(sm_motion a, sm_motion b) {
    double d = 0;
    for (int i = 0; i < a.frame_count * 66; i++) d = fmax(d, fabs((double)a.joints[i] - b.joints[i]));
    return d;
}

static void test_warp(sm_model m) {
    static float p[SM_MAX_PATH * 4];
    static double g[2 * 400]; static float sp[400];
    sm_options o = sm_default_options(m);
    o.seconds = 6;
    o.seed = 1;
    const char *walk = "a person keeps walking forward";
    sm_motion base = made(sm_generate_motion_ex(m, walk, o));
    const uint32_t T = (uint32_t)base.frame_count;
    const double L = travel(base);
    double worst = 0, sm[3];
    for (int shape = 0; shape < 3; shape++) {   /* on the path */
        const uint32_t n = shape_points(shape, L, p);
        sm_motion w = made(sm_generate_motion_ex(m, walk, with_path(o, p, (int)n)));
        worst = fmax(worst, off_path(w, base, p, n, 0)), sm[shape] = off_path(w, base, p, n, 1);
        CHECK(!memcmp(w.codes, base.codes, sizeof(int16_t) * 6 * T / 2));
        sm_unload_motion(w);
    }
    const float timed[3 * 4] = {NAN, 0, 0, NAN, 3, 0, (float)(0.55 * L), NAN, NAN, 0, (float)L, NAN};
    sm_motion w = made(sm_generate_motion_ex(m, walk, with_path(o, timed, 3)));
    const double at_time = off_path(w, base, timed, 3, 0);
    sm_unload_motion(w);
    sm_root_smooth(base.joints, T, 20, g, sp);   /* its own path, a point per frame */
    uint32_t n = 0;
    for (uint32_t t = 0; t < T; t++, n++) p[4 * n] = NAN, p[4 * n + 1] = (float)g[2 * t], p[4 * n + 2] = (float)g[2 * t + 1], p[4 * n + 3] = NAN;
    w = made(sm_generate_motion_ex(m, walk, with_path(o, p, (int)n)));
    const double own = max_diff(w, base);
    sm_unload_motion(w);
    printf("path: on the path within %.2f mm (straight, S, r 4 m), smoothed root %.0f / %.0f / %.0f mm; a timed point %.2f mm; "
           "its own path moves joints %.2f mm\n", worst * 1e3, sm[0] * 1e3, sm[1] * 1e3, sm[2] * 1e3, at_time * 1e3, own * 1e3);
    CHECK(worst <= 0.02 && at_time <= 0.02 && own <= 1e-3 && sm[0] <= 0.02 && sm[1] <= 0.04 && sm[2] <= 0.04);

    const uint32_t ns = shape_points(1, L, p);   /* the warp off: the motion of no path, the same error */
    sm_options po = with_path(o, p, (int)ns);
    w = made(sm_generate_motion_ex(m, walk, po));
    po.path_warp = false;
    sm_motion x = made(sm_generate_motion_ex(m, walk, po));
    CHECK(hash(x) == hash(base) && x.path_error == w.path_error && w.path_error > 0.1f);
    sm_unload_motion(x);
    po.path_warp = true;
    x = made(sm_decode_codes(m, w.codes, (int)T / 2, po));   /* decode only */
    CHECK(hash(x) == hash(w));
    sm_unload_motion(x);
    sm_unload_motion(w);

    const sm_block blocks[2] = {{walk, NULL, 3, 1, NULL}, {"a person walks forward slowly", NULL, 3, 2, NULL}};   /* timeline time */
    w = made(sm_generate_timeline(m, blocks, 2, o));
    const uint32_t nb = shape_points(2, travel(w), p);
    x = made(sm_generate_timeline(m, blocks, 2, with_path(o, p, (int)nb)));
    CHECK(off_path(x, w, p, nb, 0) < 0.02 && !memcmp(x.codes, w.codes, sizeof(int16_t) * 6 * (size_t)x.block_start_frames[2] / 2));
    const sm_block kept[2] = {{NULL, NULL, 3, 1, x.codes}, {NULL, NULL, 3, 2, x.codes + x.block_start_frames[1] / 2 * 6}};
    sm_motion y = made(sm_generate_timeline(m, kept, 2, with_path(o, p, (int)nb)));
    CHECK(hash(y) == hash(x) && y.path_error == x.path_error);
    sm_unload_motion(y);
    sm_unload_motion(w);

    o.previous = &x;   /* a previous motion: the path in its frame */
    o.seconds = 3;
    const float *end = x.root + 3 * (x.frame_count - 1);
    const float pre[2 * 4] = {NAN, end[0], end[2], NAN, NAN, end[0] + 3, end[2] + 3, NAN};
    sm_unload_motion(base);
    base = made(sm_generate_motion_ex(m, walk, o));
    w = made(sm_generate_motion_ex(m, walk, with_path(o, pre, 2)));
    CHECK(finite(w) && off_path(w, base, pre, 2, 0) < 0.02);
    sm_unload_motion(w);
    sm_unload_motion(base);
    sm_unload_motion(x);

    o.previous = NULL;   /* a closed path */
    o.loop = true;
    base = made(sm_generate_motion_ex(m, "a person walks in a circle", o));
    const double C = travel(base);
    n = 0;
    for (int i = 0; i <= 32; i++, n++) {
        const double a = 2 * 3.14159265358979 * i / 32, rad = C / (2 * 3.14159265358979);
        p[4 * n] = NAN, p[4 * n + 1] = i < 32 ? (float)(rad - rad * cos(a)) : 0, p[4 * n + 2] = i < 32 ? (float)(rad * sin(a)) : 0, p[4 * n + 3] = NAN;
    }
    w = made(sm_generate_motion_ex(m, "a person walks in a circle", with_path(o, p, (int)n)));
    CHECK(finite(w) && off_path(w, base, p, n, 0) < 0.02);
    sm_unload_motion(w);
    sm_unload_motion(base);
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
static void test_gate(sm_model m) {
    enum { SEEDS = 4 };
    static float p[SM_MAX_PATH * 4];
    const char *walk = "a person keeps walking forward";
    sm_options o = sm_default_options(m);
    o.seconds = 6;
    sm_motion base[SEEDS];
    for (int i = 0; i < SEEDS; i++) {
        o.seed = 1 + i;
        base[i] = made(sm_generate_motion_ex(m, walk, o));
    }
    static const char *const names[3] = {"straight", "gentle S", "circle r 4 m"};
    for (int shape = 0; shape < 3; shape++) {
        double v[2][4] = {{0}};   /* unwarped, warped: ankle, toe, jerk median, p95 */
        for (int i = 0; i < SEEDS; i++) {
            const uint32_t n = shape_points(shape, travel(base[i]), p), T = (uint32_t)base[i].frame_count;
            o.seed = 1 + i;
            sm_motion w = made(sm_generate_motion_ex(m, walk, with_path(o, p, (int)n)));
            for (int k = 0; k < 2; k++) {
                const float *x = k ? w.joints : base[i].joints;
                double med, p95;
                jerk(x, T, &med, &p95);
                v[k][0] += slide(base[i].joints, x, T, 0), v[k][1] += slide(base[i].joints, x, T, 1);
                v[k][2] += med, v[k][3] += p95;
            }
            sm_unload_motion(w);
        }
        printf("gate %-12s ankle %+3.0f %%, toe %+3.0f %%, jerk median %+3.0f %%, p95 %+3.0f %%\n", names[shape],
               100 * (v[1][0] / v[0][0] - 1), 100 * (v[1][1] / v[0][1] - 1), 100 * (v[1][2] / v[0][2] - 1), 100 * (v[1][3] / v[0][3] - 1));
        for (int k = 0; k < 4; k++) CHECK(v[1][k] <= 1.2 * v[0][k]);
    }
    for (int i = 0; i < SEEDS; i++) sm_unload_motion(base[i]);
}

int main(void) {
    for (int i = 0; i < 60 * 6; i++) codes[i] = (int16_t)(i % 6 == 5 ? -1 : (i * 37 + i / 6 * 11) % 512);
    sm_model m = sm_load_model("tests/data/test.gguf");
    CHECK(sm_is_model_valid(m));
    if (!sm_is_model_valid(m)) return CHECK_DONE();
    test_points(m);
    test_golden(m);
    test_fuzz(m);
    sm_unload_model(m);
    const char *dir = getenv("SM_MODEL_DIR");
    char path[512];
    snprintf(path, sizeof path, "%s/SprongMotion-14M-v1.0-Q8_0.gguf", dir ? dir : "tests/data");
    m = dir ? sm_load_model(path) : (sm_model){0};
    if (!sm_is_model_valid(m)) {
        printf("test_path: no model in $SM_MODEL_DIR, warp tests skipped\n");
        return CHECK_DONE();
    }
    test_warp(m);
    sm_unload_model(m);
    const char *gate = getenv("SM_GATE_MODEL");   /* the model the gate was measured on */
    m = gate ? sm_load_model(gate) : (sm_model){0};
    if (!sm_is_model_valid(m)) {
        printf("test_path: no $SM_GATE_MODEL, gate skipped\n");
        return CHECK_DONE();
    }
    test_gate(m);
    sm_unload_model(m);
    return CHECK_DONE();
}
