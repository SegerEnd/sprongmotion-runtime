/* The public API through the shared library: misuse, options, decoding, BVH; with $SM_MODEL_DIR generation. */
#define _POSIX_C_SOURCE 200809L   /* posix_memalign */
#include "sprongmotion.h"
#include "check.h"
#include <pthread.h>
#include <stdatomic.h>

static atomic_long live_allocs;
static void *count_alloc(size_t size, size_t align) {
    void *p = NULL;
    if (posix_memalign(&p, align, size ? size : 1)) return NULL;
    atomic_fetch_add(&live_allocs, 1);
    return p;
}
static void count_free(void *p) {
    atomic_fetch_sub(&live_allocs, 1);
    free(p);
}

static sm_model load(const char *dir, const char *file) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, file);
    sm_model m = sm_load_model(path);
    if (!sm_is_model_valid(m)) fprintf(stderr, "%s: %s\n", path, sm_get_error());
    return m;
}

static int failed_with(const char *msg) {
    if (strstr(sm_get_error(), msg)) return 1;
    fprintf(stderr, "  error \"%s\", want \"%s\"\n", sm_get_error(), msg);
    return 0;
}

static int empty(sm_motion mo) { return !sm_is_motion_valid(mo) && mo.frame_count == 0 && !mo.joints; }

/* empty inputs and values out of range fail with a message; empty values unload as no-ops */
static void test_misuse(sm_model m) {
    const int16_t codes[12] = {1, 2, 3, 4, 5, 6, 512, 0, 0, 0, 0, 0};
    const float joints[4 * 66] = {0}, vec[3] = {0};
    const sm_path_point point = {0, 0, 0, 0};
    sm_options o = sm_default_options(m);
    CHECK(empty(sm_decode_codes(m, NULL, 2, o)) && failed_with("codes is NULL"));
    CHECK(empty(sm_decode_codes(m, codes, 0, o)) && failed_with("token_count = 0 is out of range 1 .. 6000"));
    CHECK(empty(sm_decode_codes(m, codes, 2, o)) && failed_with("a base code = 512 is out of range 0 .. 511"));
    CHECK(empty(sm_encode_motion(m, NULL, 4, o)) && empty(sm_encode_motion(m, joints, 4, o)));
    CHECK(failed_with("no motion encoder"));
    CHECK(empty(sm_generate_timeline(m, NULL, 1, o)) && failed_with("blocks is NULL"));
    CHECK(!sm_is_text_valid(sm_text_from_vector(m, vec, 3)) && failed_with("no text encoder"));
    CHECK(!sm_is_text_valid(sm_encode_text(m, "a")) && failed_with("no text encoder"));
    CHECK(empty(sm_generate_motion(m, "a", 2, 0)) && failed_with("no generator"));
    CHECK(empty(sm_generate_motion_ex(m, NULL, o)) && !sm_export_motion_bvh_text((sm_motion){0}, 100));
    for (int i = 0; i < 6; i++) {
        static const char *const msg[6] = {
            "steps = 0 is out of range 1 .. 64", "seed = -1 is out of range 0 .. 2147483647", "seconds = 0.4",
            "text_guidance = nan is out of range 0 .. 20", "path_count = 1 is out of range 2 .. 1024",
            "pinned_token_count = 513 is out of range 0 .. 512",
        };
        sm_options x = sm_default_options(m);
        if (i == 0) x.steps = 0;
        if (i == 1) x.seed = -1;
        if (i == 2) x.seconds = 0.4f;
        if (i == 3) x.text_guidance = NAN;
        if (i == 4) x.path = &point, x.path_count = 1;
        if (i == 5) x.pinned_codes = codes, x.pinned_token_count = 513;
        CHECK(empty(sm_decode_codes(m, codes + 6, 1, x)) && failed_with(msg[i]));
    }
    const sm_model none = {0};
    CHECK(!sm_is_model_valid(none) && !sm_get_model_meta(none, "general.name") && !sm_get_model_word(none, 0));
    CHECK(!sm_get_model_meta(m, "nope") && !sm_get_model_word(m, -1) && sm_default_options(none).steps == 0);
    sm_unload_motion((sm_motion){0});
    sm_unload_text((sm_text){0});
    sm_unload_model(none);
    sm_free(NULL);
}

/* the defaults are the model file's; every option at its limits is accepted */
static void test_options(sm_model m) {
    sm_options o = sm_default_options(m);
    CHECK(o.seconds == 4 && o.steps == 10 && o.residual_layers == 5 && o.text_guidance == 3 && o.temperature == 1);
    CHECK(o.top_k == 52 && o.foot_lock && !o.in_place && !o.loop && o.path_warp && o.seed == 0);
    const int16_t codes[6] = {0, 1, 2, 3, 4, 5};
    o.seconds = m.max_seconds, o.seed = 2147483647, o.steps = 64, o.residual_layers = 0, o.text_guidance = 20;
    o.residual_guidance = 0, o.temperature = 0.01f, o.top_k = 512;
    sm_motion mo = sm_decode_codes(m, codes, 1, o);
    CHECK(sm_is_motion_valid(mo) && mo.frame_count == 2);
    sm_unload_motion(mo);
    o.temperature = 10.5f;
    CHECK(empty(sm_decode_codes(m, codes, 1, o)) && failed_with("temperature = 10.5 is out of range 0.01 .. 10"));
}

static void qmul(const double a[4], const double b[4], double o[4]) {   /* w x y z */
    o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

/* FK of local rotations q (x y z w) on the offsets from root positions vs the joints: mean and max distance */
static void fk_error(sm_model m, const float *q, const float *off, const float *root, const float *j, int T,
                     double *mean, double *worst) {
    double sum = 0;
    *worst = 0;
    for (int t = 0; t < T; t++) {
        double G[22][4], P[22][3];
        for (int k = 0; k < 22; k++) {
            const float *l = q + ((size_t)t * 22 + (size_t)k) * 4;
            const int p = m.joint_parents[k];
            const double b[4] = {l[3], l[0], l[1], l[2]}, id[4] = {1, 0, 0, 0};
            qmul(p < 0 ? id : G[p], b, G[k]);
            if (p < 0) {
                for (int i = 0; i < 3; i++) P[k][i] = root[3 * t + i];
                continue;
            }
            const double *g = G[p], v[4] = {0, off[3 * k], off[3 * k + 1], off[3 * k + 2]};
            const double gi[4] = {g[0], -g[1], -g[2], -g[3]};
            double gv[4], r[4];
            qmul(g, v, gv);
            qmul(gv, gi, r);
            for (int i = 0; i < 3; i++) P[k][i] = P[p][i] + r[i + 1];
        }
        for (int k = 0; k < 22; k++) {
            const float *w = j + ((size_t)t * 22 + (size_t)k) * 3;
            const double e = sqrt(pow(P[k][0] - w[0], 2) + pow(P[k][1] - w[1], 2) + pow(P[k][2] - w[2], 2));
            sum += e;
            *worst = fmax(*worst, e);
        }
    }
    *mean = sum / (22.0 * T);
}

/* the BVH text read back (scale 100): offsets and root in metres, rotations x y z w from the ZYX Euler angles */
static int read_bvh(sm_model m, const char *bvh, float *off, float *root, float *q, int max_t) {
    int order[22], n = 0, T = 0;
    char name[32];
    const char *p = bvh;
    for (; (p = strpbrk(p, "RJEOM")) && *p != 'M'; p++) {
        if ((!strncmp(p, "ROOT ", 5) || !strncmp(p, "JOINT ", 6)) && sscanf(strchr(p, ' ') + 1, "%31s", name) == 1)
            for (int k = 0; k < 22; k++) if (!strcmp(name, m.joint_names[k])) order[n] = k;
        if (!strncmp(p, "End Site", 8)) p = strstr(p, "OFFSET") + 6;
        if (!strncmp(p, "OFFSET", 6)) { float *o = off + 3 * order[n++]; sscanf(p + 6, "%f %f %f", o, o + 1, o + 2); }
    }
    for (int k = 0; k < 66; k++) off[k] /= 100;
    p = strstr(p, "Frame Time:");
    p = strchr(p, '\n') + 1;
    for (char *e; T < max_t && *p; T++, p = e) {
        for (int k = 0; k < 3; k++) root[3 * T + k] = strtof(p, &e) / 100, p = e;
        for (int i = 0; i < 22; i++) {
            double a[3], h[3][4] = {{0}}, zy[4], r[4];
            for (int k = 0; k < 3; k++) a[k] = strtod(p, &e) * 3.14159265358979323846 / 360, p = e;
            for (int k = 0; k < 3; k++) h[k][0] = cos(a[k]), h[k][3 - k] = sin(a[k]);   /* about z, y, x */
            qmul(h[0], h[1], zy);
            qmul(zy, h[2], r);
            float *o = q + ((size_t)T * 22 + (size_t)order[i]) * 4;
            o[0] = (float)r[1], o[1] = (float)r[2], o[2] = (float)r[3], o[3] = (float)r[0];
        }
        while (*e == ' ' || *e == '\n') e++;
    }
    return T;
}

static void test_decode(sm_model m) {
    int16_t codes[40 * 6];
    for (int i = 0; i < 40 * 6; i++) codes[i] = (int16_t)(i % 6 == 5 ? -1 : (i * 37) % 512);
    sm_motion mo = sm_decode_codes(m, codes, 40, sm_default_options(m));
    CHECK(sm_is_motion_valid(mo) && mo.frame_count == 80 && mo.fps == 20 && mo.token_count == 40 && mo.seed == -1);
    CHECK(mo.confidence[0] == 1 && !memcmp(mo.codes, codes, sizeof codes) && mo.path_error == -1);
    CHECK(mo.offsets && mo.features && mo.contacts && !mo.block_start_frames && mo.block_count == 0);
    char *bvh = sm_export_motion_bvh_text(mo, 100);
    CHECK(bvh && !strncmp(bvh, "HIERARCHY\nROOT pelvis", 21) && strstr(bvh, "Frames: 80\nFrame Time: 0.05\n"));
    int lines = 0, open = 0;
    for (const char *p = bvh; p && *p; p++) lines += *p == '\n', open += (*p == '{') - (*p == '}');
    CHECK(open == 0 && strstr(bvh, "End Site") && !strchr(bvh, ','));
    printf("bvh: %zu bytes, %d lines\n", strlen(bvh), lines);
    CHECK(!sm_export_motion_bvh_text(mo, 0) && failed_with("scale"));
    CHECK(!sm_export_motion_bvh(mo, "/nonexistent/dir/x.bvh") && failed_with("cannot write"));
    sm_set_memory_functions(NULL, NULL);  // the text still goes back to the functions it was made with
    sm_free(bvh);
    sm_set_memory_functions(count_alloc, count_free);
    sm_unload_motion(mo);
}

/* without a generator: kept blocks still decode as one timeline; the rest says what is missing */
static void test_timeline_kept(sm_model m) {
    int16_t codes[2][30 * 6];
    for (int i = 0; i < 30 * 6; i++) codes[0][i] = (int16_t)((i * 37) % 512), codes[1][i] = (int16_t)((i * 11) % 512);
    sm_options o = sm_default_options(m);
    sm_block b[2] = {{NULL, NULL, 3, 0, codes[0]}, {NULL, NULL, 2.5f, 0, codes[1]}};
    sm_motion mo = sm_generate_timeline(m, b, 2, o);
    CHECK(sm_is_motion_valid(mo) && mo.frame_count == 110 && mo.seed == -1 && mo.block_count == 2);
    CHECK(mo.block_start_frames[1] == 60 && mo.block_start_frames[2] == 110);
    CHECK(!memcmp(mo.codes + 30 * 6, codes[1], 25 * 6 * sizeof(int16_t)));
    sm_unload_motion(mo);
    b[1].kept_codes = NULL, b[1].prompt = "a person waves";
    CHECK(empty(sm_generate_timeline(m, b, 2, o)) && failed_with("no generator"));
    b[1].kept_codes = codes[1], b[1].seconds = 11;
    CHECK(empty(sm_generate_timeline(m, b, 2, o)) && failed_with("block 1 seconds = 11"));
    b[1].seconds = 2.5f, codes[1][6] = -1;
    CHECK(empty(sm_generate_timeline(m, b, 2, o)) && failed_with("block 1: a kept code"));
    CHECK(empty(sm_generate_timeline(m, b, 0, o)) && failed_with("block_count = 0"));
    o.loop = true;
    mo = sm_decode_codes(m, codes[0], 30, o);
    CHECK(sm_is_motion_valid(mo));
    sm_unload_motion(mo);
}

/* a failed call's empty result keeps that call's error through the next ones */
static void test_errors(void) {
    sm_model m = sm_load_model("/nonexistent/m.gguf");
    CHECK(!sm_is_model_valid(m) && failed_with("cannot open the model file"));
    sm_motion mo = sm_generate_motion(m, "a", 2, 0);
    CHECK(!sm_export_motion_bvh(mo, "x.bvh") && failed_with("cannot open the model file"));
    CHECK(!sm_is_model_valid(sm_load_model(NULL)) && failed_with("file_name is NULL"));
    CHECK(!sm_is_model_valid(sm_load_model_from_memory("GGUF", 4)) && failed_with("not a GGUF v3 file"));
    CHECK(!strcmp(sm_get_version(), "1.1.0") && !strncmp(sm_get_build_info(), "1.1.0 ", 6));
    sm_set_memory_functions(count_alloc, NULL);
    CHECK(failed_with("set both alloc and free"));
}

static int has_word(sm_model m, const char *w) {
    for (int i = 0; sm_get_model_word(m, i); i++) if (!strcmp(sm_get_model_word(m, i), w)) return 1;
    return 0;
}

/* the vocabulary: sorted, lowercase, NULL past the end; with a text encoder every word is known */
static void test_vocab(sm_model m) {
    int n = 0, bad = 0, unknown = 0;
    const char *w, *prev = "";
    for (; (w = sm_get_model_word(m, n)); prev = w, n++) {
        bad += !*w || strcmp(prev, w) >= 0;
        for (const char *p = w; *p; p++) bad += (*p < 'a' || *p > 'z') && (unsigned char)*p < 0x80;
    }
    CHECK(!bad && !sm_get_model_word(m, n) && !sm_get_model_word(m, 1 << 30));
    if (!m.has_generator) return;
    for (int i = 0; i < n; i += 37) {
        sm_text t = sm_encode_text(m, sm_get_model_word(m, i));
        CHECK(sm_is_text_valid(t));
        unknown += t.unknown_word_count;
        sm_unload_text(t);
    }
    CHECK(n > 1000 && !unknown && has_word(m, "walk") && has_word(m, "jumps") && !has_word(m, "kaleidoscope"));
    printf("vocabulary: %d words (%s .. %s)\n", n, sm_get_model_word(m, 0), sm_get_model_word(m, n - 1));
}

static bool stop_at_3(void *user, int done, int total) {
    (void)total;
    *(int *)user = done;
    return done < 3;
}

typedef struct { sm_model m; int seed; sm_motion out; } job;

static void *generate_on_thread(void *arg) {
    job *j = arg;
    sm_options o = sm_default_options(j->m);
    o.seconds = 3;
    o.seed = j->seed;
    j->out = sm_generate_motion_ex(j->m, "a person walks forward and waves", o);
    return NULL;
}

/* the same codes from a prompt, its encoding and other threads sharing the model */
static void test_generate(sm_model m) {
    sm_options o = sm_default_options(m);
    o.seconds = 3;
    o.seed = 4;
    sm_motion a = sm_generate_motion_ex(m, "a person walks forward and waves", o);
    CHECK(sm_is_motion_valid(a) && a.frame_count == 60 && a.seed == 4 && a.token_count == 30 && a.confidence[0] > 0);
    sm_text t = sm_encode_text(m, "a person walks forward and waves");
    CHECK(sm_is_text_valid(t) && t.vector_size == m.text_vector_size && t.unknown_word_count == 0 && !t.unknown_words[0]);
    o.text = &t;
    sm_motion b = sm_generate_motion_ex(m, NULL, o);
    CHECK(sm_is_motion_valid(b) && !memcmp(a.codes, b.codes, 180 * sizeof(int16_t)));
    CHECK(!memcmp(a.joints, b.joints, 60 * 66 * sizeof(float)));
    sm_unload_motion(b);

    sm_text v = sm_text_from_vector(m, t.vector, t.vector_size);   /* a vector, no words */
    o.text = &v;
    b = sm_generate_motion_ex(m, NULL, o);
    CHECK(sm_is_text_valid(v) && sm_is_motion_valid(b) && b.frame_count == 60);
    sm_unload_motion(b);
    sm_unload_text(v);
    sm_unload_text(t);

    t = sm_encode_text(m, "a person does photosynthesis by a kaleidoscope");
    CHECK(t.unknown_word_count == 2 && !strcmp(t.unknown_words, "photosynthesis\nkaleidoscope"));
    sm_unload_text(t);

    job jobs[4];
    pthread_t threads[4];
    for (int i = 0; i < 4; i++) {
        jobs[i] = (job){m, 3 + i % 2, {0}};
        pthread_create(&threads[i], NULL, generate_on_thread, &jobs[i]);
    }
    int same = 0;
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
        same += jobs[i].seed == 4 && !memcmp(jobs[i].out.codes, a.codes, 180 * sizeof(int16_t));
        same += jobs[i].seed == 3 && sm_is_motion_valid(jobs[i].out);
        sm_unload_motion(jobs[i].out);
    }
    CHECK(same == 4);

    int passes = 0;
    o.text = NULL;
    o.progress = stop_at_3;
    o.user = &passes;
    CHECK(empty(sm_generate_motion_ex(m, "a person walks", o)) && passes == 3 && failed_with("progress callback"));
    sm_unload_motion(a);
    printf("generate: the same codes from a prompt, its encoding and 4 threads at once; cancel after %d passes\n", passes);
}

/* foot lock: ankles still while in contact, the leg bones keep their lengths; in place: the root stays at x = z = 0 */
static void test_post(sm_model m) {
    double slide[2] = {0}, stretch = 0, low = 1;
    int cnt = 0;
    sm_options o = sm_default_options(m);
    o.seconds = 6;
    sm_motion on = sm_generate_motion_ex(m, "a person walks forward", o);
    o.foot_lock = false;
    sm_motion off = sm_generate_motion_ex(m, "a person walks forward", o);
    CHECK(sm_is_motion_valid(on) && sm_is_motion_valid(off));
    const uint8_t *ct = on.contacts;
    const int T = on.frame_count;
    for (int k = 0; k < 2; k++) {
        const float *j = (k ? off : on).joints;
        for (int t = 1; t < T; t++)
            for (int s = 0; s < 2; s++) {
                const int ank = 7 + s, col = 2 * s;
                if (!ct[4 * t + col] || !ct[4 * (t - 1) + col]) continue;
                const float *p = j + ((size_t)t * 22 + (size_t)ank) * 3, *q = p - 66;
                slide[k] += hypot(p[0] - q[0], p[2] - q[2]) * 20;
                cnt += !k;
            }
        for (int t = 0; !k && t < T; t++)
            for (int b = 4; b <= 8; b++) {   /* knees and ankles: the bone to their parent */
                if (b == 6) continue;
                const float *p = j + ((size_t)t * 22 + (size_t)b) * 3;
                const float *q = j + ((size_t)t * 22 + (size_t)m.joint_parents[b]) * 3;
                const float *p0 = off.joints + (p - j), *q0 = off.joints + (q - j);
                stretch = fmax(stretch, fabs(hypot(hypot(p[0] - q[0], p[1] - q[1]), p[2] - q[2]) -
                                             hypot(hypot(p0[0] - q0[0], p0[1] - q0[1]), p0[2] - q0[2])));
            }
        for (int t = 0; !k && t < T; t++)
            for (int f = 7; f <= 11; f += f == 8 ? 2 : 1) low = fmin(low, j[((size_t)t * 22 + (size_t)f) * 3 + 1]);
    }
    printf("foot lock: ankle slide in contact %.3f m/s (off: %.3f), bone length change %.1e m, lowest foot %.3f m\n",
           slide[0] / cnt, slide[1] / cnt, stretch, low);
    CHECK(slide[0] < 0.5 * slide[1] && stretch < 1e-4 && low >= -1e-4);

    double mean, worst, bmean, bworst;
    float off_b[66];
    static float root_b[200 * 3], q_b[200 * 88];
    fk_error(m, off.rotations, m.rest_offsets, off.root, off.joints, T, &mean, &worst);
    char *bvh = sm_export_motion_bvh_text(off, 100);
    CHECK(bvh && read_bvh(m, bvh, off_b, root_b, q_b, 200) == T);
    fk_error(m, q_b, off_b, root_b, off.joints, T, &bmean, &bworst);
    printf("rotations: FK vs the joints on the reference skeleton: mean %.1f mm, max %.1f mm; BVH (the clip's own): "
           "mean %.1f mm, max %.1f mm\n", mean * 1e3, worst * 1e3, bmean * 1e3, bworst * 1e3);
    CHECK(mean < 0.1 && bmean < 0.01 && bworst < 0.06);   /* the joints' bones change length within a clip */
    sm_free(bvh);

    o.in_place = true;
    sm_unload_motion(on);
    on = sm_generate_motion_ex(m, "a person walks forward", o);
    CHECK(!memcmp(on.root, off.root, (size_t)T * 3 * sizeof(float)));
    const float *j = on.joints;
    CHECK(j[(T - 1) * 66] == 0 && j[(T - 1) * 66 + 2] == 0 && j[(T - 1) * 66 + 1] == off.root[(T - 1) * 3 + 1]);
    sm_motion dec = sm_decode_codes(m, on.codes, on.token_count, o);
    CHECK(sm_is_motion_valid(dec) && !memcmp(dec.joints, j, (size_t)T * 66 * sizeof(float)));
    sm_unload_motion(dec);
    dec = sm_decode_codes(m, on.codes, on.token_count, sm_default_options(m));
    CHECK(dec.joints[(T - 1) * 66] != 0);
    sm_unload_motion(dec);
    sm_unload_motion(on);
    sm_unload_motion(off);
}

/* chained clips (2 s, then 1 s segments, each from the last 0.3 s of the one before): acceleration at the seams vs
   inside, for the segments' own joints played one after another and for all their codes decoded at once */
static void test_seams(sm_model m) {
    static const char *prompts[3] = {"a person walks forward", "a person keeps walking forward at a steady pace",
                                     "a person jogs in a circle"};
    enum { SEGS = 4, T_MAX = 40 + 20 * (SEGS - 1), N = 3 * 3 };
    static float J[T_MAX * 66];
    static int16_t codes[T_MAX * 3];
    static double acc[2][2][N * T_MAX];
    size_t idx[2][2] = {{0}};
    for (int clip = 0; clip < N; clip++) {
        int T = 0, L = 0, cuts[SEGS];
        sm_options o = sm_default_options(m);
        o.seed = clip % 3;
        sm_motion prev = {0};
        for (int s = 0; s < SEGS; s++) {
            o.seconds = s ? 1 : 2;
            o.previous = s ? &prev : NULL;
            sm_motion seg = sm_generate_motion_ex(m, prompts[clip / 3], o);
            CHECK(sm_is_motion_valid(seg));
            cuts[s] = T;
            memcpy(J + (size_t)T * 66, seg.joints, sizeof(float) * 66 * (size_t)seg.frame_count);
            memcpy(codes + L * 6, seg.codes, sizeof(int16_t) * 6 * (size_t)seg.token_count);
            T += seg.frame_count;
            L += seg.token_count;
            sm_unload_motion(prev);
            prev = seg;
        }
        sm_unload_motion(prev);
        sm_motion all = sm_decode_codes(m, codes, L, sm_default_options(m));
        CHECK(all.frame_count == T);
        for (int way = 0; way < 2; way++)
            for (int t = 1; t + 1 < T; t++) {
                const float *p = (way ? all.joints : J) + (size_t)t * 66;
                double a = 0;
                int at = 0;
                for (int k = 0; k < 66; k += 3)
                    a += sqrt(pow(p[k + 66] - 2 * p[k] + p[k - 66], 2) + pow(p[k + 67] - 2 * p[k + 1] + p[k - 65], 2) +
                              pow(p[k + 68] - 2 * p[k + 2] + p[k - 64], 2));
                for (int s = 1; s < SEGS; s++) at |= t + 2 >= cuts[s] && t <= cuts[s] + 1;
                acc[way][at][idx[way][at]++] = a * 400 / 22;   /* m/s^2 at 20 fps, mean over joints */
            }
        sm_unload_motion(all);
    }
    double med[2][2];
    for (int w = 0; w < 2; w++)
        for (int at = 0; at < 2; at++) {
            qsort(acc[w][at], idx[w][at], sizeof(double), cmp_double);
            med[w][at] = acc[w][at][idx[w][at] / 2];
        }
    printf("seams (1 s segments, median acceleration, m/s^2): one decode %.2f at seams vs %.2f inside; own joints %.2f "
           "vs %.2f\n", med[1][1], med[1][0], med[0][1], med[0][0]);
    CHECK(med[1][1] < 1.25 * med[1][0] && med[0][1] < 2 * med[0][0]);
}

int main(void) {
    test_errors();
    sm_set_memory_functions(count_alloc, count_free);
    sm_model m = load("tests/data", "test.gguf");
    CHECK(sm_is_model_valid(m));
    if (!sm_is_model_valid(m)) return CHECK_DONE();
    CHECK(m.fps == 20 && m.joint_count == 22 && m.code_layers == 6 && m.feature_size == 263 && !m.has_encoder);
    CHECK(!strcmp(m.joint_names[0], "pelvis") && m.joint_parents[0] == -1 && m.joint_parents[4] == 1);
    CHECK(!strcmp(sm_get_model_meta(m, "general.architecture"), "sprongmotion") && !m.follows_path);
    test_misuse(m);
    test_options(m);
    test_decode(m);
    test_timeline_kept(m);
    test_vocab(m);
    sm_unload_model(m);
    const char *dir = getenv("SM_MODEL_DIR");
    if (dir) m = load(dir, "SprongMotion-14M-v1.0-Q8_0.gguf");
    if (dir && sm_is_model_valid(m)) {
        printf("model: %s\n", m.id);
        CHECK(m.has_generator && m.text_vector_size == 384 && m.max_seconds == 10);
        test_generate(m);
        test_post(m);
        test_seams(m);
        test_vocab(m);
        sm_unload_model(m);
    } else {
        printf("test_api: no model in $SM_MODEL_DIR, generation skipped\n");
    }
    printf("allocations still live: %ld\n", atomic_load(&live_allocs));
    CHECK(atomic_load(&live_allocs) == 0);
    return CHECK_DONE();
}
