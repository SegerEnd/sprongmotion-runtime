/* Motion editing: the encoder vs torch, round trips, pinned codes, loops, timelines and their seams. */
#include "../src/sm_internal.h"
#include "check.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { double *v; size_t n, cap; } list;
static void add(list *l, double x) {
    if (l->n == l->cap) l->v = realloc(l->v, sizeof(double) * (l->cap = l->cap ? 2 * l->cap : 256));
    l->v[l->n++] = x;
}
static double pct(list *l, double p) { qsort(l->v, l->n, sizeof(double), cmp_double); return l->n ? l->v[(size_t)(p * (double)(l->n - 1))] : 0; }
static double mean(const list *l) { double s = 0; for (size_t i = 0; i < l->n; i++) s += l->v[i]; return l->n ? s / (double)l->n : 0; }

/* seam vs inside: jerk of the local pose (the features' joints relative to the root in its heading frame, m/s^3, mean
   over joints) and foot slide (cm/s of a foot on the ground, horizontal) */
typedef struct { list jerk[2], slide[2]; } seams;
static double local_jerk(const float *f0, const float *f1, const float *f2, const float *f3) {
    double s = 0;
    for (int j = 0; j < 21; j++) {
        double d2 = 0;
        for (int k = 0; k < 3; k++) { const int i = 4 + 3 * j + k; const double d = f3[i] - 3.0 * f2[i] + 3.0 * f1[i] - f0[i]; d2 += d * d; }
        s += sqrt(d2);
    }
    return s / 21 * 8000;
}
static void measure(seams *st, sm_motion mo, const uint32_t *junctions, int nj) {
    const uint32_t T = (uint32_t)mo.frame_count;
    const float *f = mo.features, *jt = mo.joints;
    const uint8_t *ct = mo.contacts;
    static const uint32_t feet[4] = {7, 10, 8, 11};
    for (uint32_t t = 0; t + 3 < T; t++) {
        int at = 0, near = 0;
        for (int i = 0; i < nj; i++) {
            const int64_t J = junctions[i], d = (int64_t)t - J;
            at |= d >= -3 && d <= 2;
            near |= d >= -9 && d <= 6;
        }
        if (!at && near) continue;
        add(&st->jerk[at], local_jerk(f + t * SM_F, f + (t + 1) * SM_F, f + (t + 2) * SM_F, f + (t + 3) * SM_F));
        for (uint32_t k = 0; k < 4; k++) {
            if (!ct[t * 4 + k]) continue;
            const float *a = jt + (t * SM_J + feet[k]) * 3, *b = a + SM_J * 3;
            add(&st->slide[at], hypot(b[0] - a[0], b[2] - a[2]) * 2000);
        }
    }
}
static void report(const char *name, seams *s) {
    const double jm[2] = {pct(&s->jerk[0], 0.5), pct(&s->jerk[1], 0.5)}, sl[2] = {mean(&s->slide[0]), mean(&s->slide[1])};
    printf("%s: jerk median %.1f at seams vs %.1f inside (%+.0f %%), p95 %.1f vs %.1f; foot slide %.2f vs %.2f cm/s (%+.0f %%)"
           " -> gate %s\n", name, jm[1], jm[0], 100 * (jm[1] / jm[0] - 1), pct(&s->jerk[1], 0.95), pct(&s->jerk[0], 0.95),
           sl[1], sl[0], 100 * (sl[1] / sl[0] - 1), jm[1] <= 1.2 * jm[0] && sl[1] <= 1.2 * sl[0] ? "pass" : "FAIL");
    for (int i = 0; i < 2; i++) free(s->jerk[i].v), free(s->slide[i].v);
}

/* mean joint distance, mm; rel: both relative to their root (the pose without the root's drift) */
static double mpjpe(const float *a, const float *b, uint32_t T, int rel) {
    double s = 0;
    for (size_t t = 0; t < T; t++)
        for (size_t j = 0; j < SM_J; j++) {
            double d2 = 0;
            for (size_t k = 0; k < 3; k++) {
                const size_t i = (t * SM_J + j) * 3 + k, r = t * SM_J * 3 + k;
                const double d = (a[i] - (rel ? a[r] : 0)) - (b[i] - (rel ? b[r] : 0));
                d2 += d * d;
            }
            s += sqrt(d2);
        }
    return s / ((double)T * (double)SM_J) * 1000;
}

typedef struct { uint32_t T; float *joints, *feat, *codes; } clip;
static clip clips[16]; static uint32_t n_clips;

static void load_golden(const char *path) {
    FILE *f = fopen(path, "rb"); CHECK(f != NULL); if (!f) return;
    CHECK(fread(&n_clips, 4, 1, f) == 1 && n_clips <= 16);
    for (uint32_t i = 0; i < n_clips; i++) {
        clip *c = &clips[i]; CHECK(fread(&c->T, 4, 1, f) == 1);
        const size_t nj = (size_t)c->T * SM_J * 3, nf = (size_t)(c->T - 1) * SM_F, nc = (size_t)(c->T - 1) / 2 * 6;
        c->joints = malloc(4 * nj), c->feat = malloc(4 * nf), c->codes = malloc(4 * nc);
        CHECK(fread(c->joints, 4, nj, f) == nj && fread(c->feat, 4, nf, f) == nf && fread(c->codes, 4, nc, f) == nc);
    }
    fclose(f);
}

static void test_features(void) {   /* the recipe vs torch, per feature group */
    static const int group[6] = {0, 4, 67, 193, 259, 263};
    double worst[5] = {0}; int contact_diff = 0;
    for (uint32_t i = 0; i < n_clips; i++) {
        const clip *c = &clips[i];
        double *tmp = malloc(sizeof(double) * sm_feat_tmp_doubles(c->T)); float *f = malloc(sizeof(float) * (c->T - 1) * SM_F);
        sm_joints_to_features(c->joints, c->T, tmp, f);
        for (size_t t = 0; t + 1 < c->T; t++)
            for (int g = 0; g < 5; g++)
                for (size_t k = (size_t)group[g]; k < (size_t)group[g + 1]; k++) {
                    const double d = fabs(f[t * SM_F + k] - c->feat[t * SM_F + k]);
                    if (g == 4) contact_diff += d > 0; else worst[g] = fmax(worst[g], d);
                }
        free(tmp); free(f);
    }
    printf("features vs torch, max abs diff: root %.1e, local joints %.1e m, 6D rotations %.1e, velocities %.1e m; "
           "%d contact flags differ\n", worst[0], worst[1], worst[2], worst[3], contact_diff);
    NEAR(worst[0], 0, 3e-5); NEAR(worst[1], 0, 1e-5); NEAR(worst[2], 0, 1e-4); NEAR(worst[3], 0, 1e-5); CHECK(contact_diff <= 2);
}

/* the motion's canonical frame (on the floor, at the origin facing +Z, on the target skeleton): the reference a round
   trip is measured against */
static float *canonical(const float *joints, uint32_t T) {
    double *tmp = malloc(sizeof(double) * sm_feat_tmp_doubles(T)); float *f = malloc(sizeof(float) * (T - 1) * SM_F);
    float *j = malloc(sizeof(float) * (T - 1) * SM_J * 3); sm_track tr = {0};
    sm_joints_to_features(joints, T, tmp, f); sm_features_to_joints(f, T - 1, &tr, j);
    free(tmp); free(f); return j;
}

static sm_motion made(sm_motion mo) {
    if (!sm_is_motion_valid(mo)) fprintf(stderr, "  %s\n", sm_get_error());
    CHECK(sm_is_motion_valid(mo));
    return mo;
}

static void test_encoder(sm_model m, sm_options o) {
    uint32_t same[6] = {0}, tokens = 0, frames = 0;
    double rt[2] = {0}, secs = 0;
    for (uint32_t i = 0; i < n_clips; i++) {
        const clip *k = &clips[i];
        const clock_t t0 = clock();
        sm_motion mo = made(sm_encode_motion(m, k->joints, (int)k->T, o));
        secs += (double)(clock() - t0) / CLOCKS_PER_SEC, frames += k->T;
        const uint32_t n = (uint32_t)mo.token_count * 6;
        CHECK(n == (k->T - 1) / 2 * 6);
        for (uint32_t j = 0; j < n; j++) same[j % 6] += mo.codes[j] == (int16_t)k->codes[j];
        tokens += n / 6;
        float *ref = canonical(k->joints, k->T);
        for (int rel = 0; rel < 2; rel++) rt[rel] += mpjpe(mo.joints, ref, (uint32_t)mo.frame_count, rel) / n_clips;
        free(ref);
        sm_unload_motion(mo);
    }
    printf("encoder vs torch: codes equal per layer %.1f %.1f %.1f %.1f %.1f %.1f %% of %u tokens; round trip %.1f mm "
           "(%.1f mm root-relative, %u clips); encode + decode %.1f ms per 10 s\n", 100.0 * same[0] / tokens,
           100.0 * same[1] / tokens, 100.0 * same[2] / tokens, 100.0 * same[3] / tokens, 100.0 * same[4] / tokens,
           100.0 * same[5] / tokens, tokens, rt[0], rt[1], n_clips, secs / frames * 200 * 1000);
    CHECK(same[0] >= tokens * 99 / 100 && same[5] >= tokens * 9 / 10);
    CHECK(rt[1] < 40);
    static const char *texts[4] = {"a person walks forward", "a person jumps", "a person dances", "a person kicks with the left leg"};
    double own[2] = {0}, input = 0;
    uint32_t agree = 0, total = 0;
    for (int i = 0; i < 4; i++) {   /* generated clips: the codes come back and so do the joints */
        o.seed = 40 + i;
        sm_motion g = made(sm_generate_motion_ex(m, texts[i], o));
        sm_motion mo = made(sm_encode_motion(m, g.joints, g.frame_count, o));
        for (int k = 0; k < mo.frame_count / 2; k++) agree += g.codes[6 * k] == mo.codes[6 * k], total++;
        float *ref = canonical(g.joints, (uint32_t)g.frame_count);
        for (int rel = 0; rel < 2; rel++) own[rel] += mpjpe(mo.joints, ref, (uint32_t)mo.frame_count, rel) / 4;
        input += mpjpe(mo.joints, g.joints, (uint32_t)mo.frame_count, 1) / 4;
        free(ref);
        sm_unload_motion(mo);
        sm_unload_motion(g);
    }
    printf("generated clips: round trip %.1f mm (%.1f root-relative) on the target skeleton, %.1f root-relative vs the "
           "clip's own joints; base codes back %.0f %%\n", own[0], own[1], input, 100.0 * agree / total);
    CHECK(own[1] < 40);
    const float zero[3 * SM_J * 3] = {0}, nan_[3 * SM_J * 3] = {NAN};
    CHECK(!sm_is_motion_valid(sm_encode_motion(m, zero, 3, o)) && strstr(sm_get_error(), "no pose"));
    CHECK(!sm_is_motion_valid(sm_encode_motion(m, nan_, 3, o)) && !sm_is_motion_valid(sm_encode_motion(m, zero, 2, o)));
}

static const char *const PAIRS[8][2] = {
    {"a person walks forward", "a person jumps up and down"}, {"a person waves with the right hand", "a person claps"},
    {"a person runs forward", "a person runs forward"}, {"a person sits down", "a person stretches both arms"},
    {"a person walks in a circle", "a person turns around"}, {"a person dances", "a person dances"},
    {"a person kicks with the left leg", "a person punches forward"}, {"a person walks forward", "a person walks forward"},
};

static void test_pins(sm_model m, sm_options o) {
    int16_t pins[50 * 6];
    seams in = {0}, lock = {0};
    int exact = 1;
    o.seconds = 5;
    for (int i = 0; i < 8; i++) {
        o.pinned_token_count = 0;
        o.seed = i;
        sm_motion a = made(sm_generate_motion_ex(m, PAIRS[i][0], o));
        memcpy(pins, a.codes, sizeof pins);
        sm_unload_motion(a);
        for (int t = 20; t < 30; t++) pins[6 * t] = -1;   /* regenerate 2 s in the middle, both ends pinned */
        o.seed = 100 + i;
        o.pinned_codes = pins;
        o.pinned_token_count = 50;
        sm_motion b = made(sm_generate_motion_ex(m, PAIRS[i][1], o));
        for (int k = 0; k < 300; k++) exact &= pins[k - k % 6] < 0 || b.codes[k] == pins[k];
        measure(&in, b, (const uint32_t[]){40, 60}, 2);
        sm_unload_motion(b);
        o.pinned_token_count = 20;   /* lock 2 s */
        b = made(sm_generate_motion_ex(m, PAIRS[i][1], o));
        exact &= !memcmp(b.codes, pins, sizeof(int16_t) * 120);
        measure(&lock, b, (const uint32_t[]){40}, 1);
        sm_unload_motion(b);
    }
    CHECK(exact);
    printf("pinned tokens come back exactly: %s\n", exact ? "yes" : "NO");
    report("in-painting (2 s, both ends pinned)", &in);
    report("a locked 2 s block, then free", &lock);
    const int16_t bad[6] = {512, 0, 0, 0, 0, 0};
    o.pinned_codes = bad;
    o.pinned_token_count = 1;
    CHECK(!sm_is_motion_valid(sm_generate_motion_ex(m, "a", o)) && strstr(sm_get_error(), "a pinned code = 512"));
    o.pinned_codes = NULL;
    CHECK(!sm_is_motion_valid(sm_generate_motion_ex(m, "a", o)) && strstr(sm_get_error(), "pinned_codes is NULL"));
    int16_t many[6 * 6] = {0};
    o.seconds = 0.5f;
    o.pinned_codes = many;
    o.pinned_token_count = 6;   /* 6 pins, 5 tokens */
    CHECK(!sm_is_motion_valid(sm_generate_motion_ex(m, "a", o)) && strstr(sm_get_error(), "pinned tokens = 6"));
}

static double pose_dist(const float *f0, const float *f1) {   /* local pose distance, cm, mean over joints */
    double s = 0;
    for (int j = 0; j < 21; j++) s += sqrt(pow(f1[4 + 3 * j] - f0[4 + 3 * j], 2) + pow(f1[5 + 3 * j] - f0[5 + 3 * j], 2) + pow(f1[6 + 3 * j] - f0[6 + 3 * j], 2));
    return s / 21 * 100;
}

static void test_loop(sm_model m, sm_options o) {
    list wrap[2] = {{0}}, step = {0}, jerk[2] = {{0}}, inside = {0};
    o.seconds = 4;
    for (int i = 0; i < 8; i++)
        for (int loop = 0; loop < 2; loop++) {
            o.seed = i;
            o.loop = loop;
            sm_motion a = made(sm_generate_motion_ex(m, PAIRS[i][0], o));
            CHECK(a.frame_count == 80);
            const uint32_t T = (uint32_t)a.frame_count;
            const float *f = a.features;
            add(&wrap[loop], pose_dist(f + (T - 1) * SM_F, f));
            const float *w[6];
            for (uint32_t k = 0; k < 6; k++) w[k] = f + ((T - 3 + k) % T) * SM_F;   /* across the wrap */
            for (int k = 0; k < 3; k++) add(&jerk[loop], local_jerk(w[k], w[k + 1], w[k + 2], w[k + 3]));
            if (loop) {
                for (uint32_t t = 10; t + 10 < T; t++) {
                    add(&step, pose_dist(f + t * SM_F, f + (t + 1) * SM_F));
                    add(&inside, local_jerk(f + t * SM_F, f + (t + 1) * SM_F, f + (t + 2) * SM_F, f + (t + 3) * SM_F));
                }
                int16_t rot[40 * 6];
                for (int k = 0; k < 40 * 6; k++) rot[k] = a.codes[(k + 7 * 6) % (40 * 6)];   /* the cycle from token 7 */
                sm_motion b = made(sm_decode_codes(m, rot, 40, o));
                CHECK(!memcmp(b.features, f + 14 * SM_F, sizeof(float) * 66 * SM_F));   /* circular */
                CHECK(!memcmp(b.features + 66 * SM_F, f, sizeof(float) * 14 * SM_F));
                sm_unload_motion(b);
            }
            sm_unload_motion(a);
        }
    printf("loops (4 s, 8 prompts): last to first frame %.1f cm (median) vs %.1f cm without the loop, %.1f cm between "
           "frames inside; jerk at the wrap %.1f vs %.1f without, %.1f inside\n", pct(&wrap[1], 0.5), pct(&wrap[0], 0.5),
           pct(&step, 0.5), pct(&jerk[1], 0.5), pct(&jerk[0], 0.5), pct(&inside, 0.5));
    CHECK(pct(&wrap[1], 0.5) < pct(&wrap[0], 0.5));
    for (int i = 0; i < 2; i++) free(wrap[i].v), free(jerk[i].v);
    free(step.v), free(inside.v);
}

static void test_timeline(sm_model m, sm_options o) {
    seams st = {0};
    for (int i = 0; i < 8; i++) {
        sm_block b[3] = {{PAIRS[i][0], NULL, 3, i, NULL}, {PAIRS[i][1], NULL, 2, 50 + i, NULL},
                         {PAIRS[(i + 1) % 8][0], NULL, 3.5f, 90 + i, NULL}};
        sm_motion a = made(sm_generate_timeline(m, b, 3, o));
        CHECK(a.frame_count == 170 && a.block_start_frames[1] == 60 && a.block_start_frames[2] == 100);
        CHECK(a.block_start_frames[3] == 170);
        measure(&st, a, (const uint32_t[]){60, 100}, 2);
        if (i) {
            sm_unload_motion(a);
            continue;
        }
        sm_motion again = made(sm_generate_timeline(m, b, 3, o));   /* the same motion */
        CHECK(!memcmp(a.joints, again.joints, sizeof(float) * 170 * SM_J * 3));
        sm_unload_motion(again);
        b[0].kept_codes = a.codes;
        b[2].kept_codes = a.codes + 50 * 6;
        b[1].seed = 7, b[1].seconds = 2.5f;   /* edit the middle block: the others stay */
        sm_motion e = made(sm_generate_timeline(m, b, 3, o));
        CHECK(e.frame_count == 180 && !memcmp(e.codes, a.codes, sizeof(int16_t) * 30 * 6) && e.block_start_frames[2] == 110);
        CHECK(!memcmp(e.codes + 55 * 6, a.codes + 50 * 6, sizeof(int16_t) * 35 * 6));
        CHECK(!memcmp(e.features, a.features, sizeof(float) * 26 * SM_F));
        sm_text t = sm_encode_text(m, b[1].prompt);   /* an encoded text, the same codes */
        b[1].prompt = NULL;
        b[1].text = &t;
        sm_motion f = made(sm_generate_timeline(m, b, 3, o));
        CHECK(!memcmp(f.codes, e.codes, sizeof(int16_t) * 90 * 6));
        sm_unload_text(t);
        sm_unload_motion(f);
        sm_unload_motion(e);
        sm_unload_motion(a);
    }
    report("timeline (3 blocks, 8 sets)", &st);
}

int main(void) {
    load_golden("tests/data/enc_golden.bin");
    test_features();
    const char *dir = getenv("SM_MODEL_DIR");
    char path[512];
    snprintf(path, sizeof path, "%s/SprongMotion-14M-v1.0-Q8_0.gguf", dir ? dir : "tests/data");
    sm_model m = sm_load_model(path);
    if (!m.has_encoder) {
        printf("test_edit: no model with the encoder at %s, model tests skipped\n", path);
    } else {
        sm_options o = sm_default_options(m);
        o.foot_lock = false;   /* the model's own feet, for the slide numbers */
        test_encoder(m, o);
        test_pins(m, o);
        test_loop(m, o);
        test_timeline(m, o);
    }
    sm_unload_model(m);
    for (uint32_t i = 0; i < n_clips; i++) free(clips[i].joints), free(clips[i].feat), free(clips[i].codes);
    return CHECK_DONE();
}
