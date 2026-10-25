/* Contexts: scratch memory, text encoding, runs one pass at a time, decoding. */
#include "sm_internal.h"
#include <stdatomic.h>

#define NO_RUN "no run: call sm_ctx_begin"
#define FADE 8   /* frames over which a prefix clip's end fades into the new one */

sm_status sm_ctx_fail(sm_ctx *c, sm_status s, const char *msg) { return sm_fail(c->err, sizeof c->err, s, msg); }
sm_status sm_ctx_fail_range(sm_ctx *c, const char *name, double v, double lo, double hi) {
    return sm_fail_range(c->err, sizeof c->err, SM_E_ARG, name, v, lo, hi);
}
static int no_ctx(sm_ctx *c) { return c ? (c->err[0] = 0) : 1; }   /* clears the last error */

static sm_status grow(const sm_model *m, void **p, size_t *have, size_t need) {
    if (need <= *have) return SM_OK;
    void *q = sm_alloc(&m->h, need);
    if (!q) return SM_E_NOMEM;
    sm_free(&m->h, *p);
    *p = q, *have = need;
    return SM_OK;
}

/* room for generating L tokens and decoding Lw */
sm_status sm_ctx_fit(sm_ctx *c, uint32_t L, uint32_t Lw) {
    const sm_model *m = c->m;
    const size_t T = 2 * (size_t)Lw, gen = m->gen.layers && L ? sm_gen_scratch_floats(&m->gen, m->tok.max_len, L) : 0;
    const size_t text = m->gen.layers ? sm_text_scratch_floats(&m->text) : 0;
    const size_t dec = sizeof(float) * (sm_vq_scratch_floats(Lw) + T * SM_F) + sizeof(double) * SM_POST_TMP(T) +
                       sizeof(int16_t) * Lw * SM_VQ_LAYERS;
    sm_status s = grow(m, &c->scratch, &c->n_scratch, sizeof(float) * (gen > text ? gen : text));
    if (s == SM_OK) s = grow(m, &c->dec, &c->n_dec, dec);
    if (s != SM_OK) return sm_ctx_fail(c, s, "out of memory");
    return SM_OK;
}

sm_status sm_ctx_create(const sm_model *m, sm_ctx **out) {
    sm_ctx *c = m && out ? sm_alloc(&m->h, sizeof *c) : NULL;
    if (!c) return m && out ? SM_E_NOMEM : SM_E_ARG;
    memset(c, 0, sizeof *c);
    c->m = m;
    atomic_init(&c->cancel, 0);
    c->tw = sm_alloc(&m->h, sizeof *c->tw);
    if (!c->tw || sm_text_create(m, &c->text) != SM_OK) {
        sm_ctx_free(c);
        return SM_E_NOMEM;
    }
    *out = c;
    return SM_OK;
}

void sm_ctx_free(sm_ctx *c) {
    if (!c) return;
    sm_text_free(c->text);
    void *owned[] = {c->scratch, c->dec, c->tw, c};
    for (size_t i = 0; i < sizeof owned / sizeof owned[0]; i++) sm_free(&c->m->h, owned[i]);
}

uint32_t sm_tokens_of(const sm_model *m, double seconds) {
    return (uint32_t)(seconds * m->fps / m->frames_per_token + 0.5);
}

sm_status sm_ctx_reserve(sm_ctx *c, double max_seconds, uint32_t max_variants) {
    if (no_ctx(c)) return SM_E_ARG;
    const double *sec = c->m->param[SM_P_SECONDS];
    if (!(max_seconds >= sec[0] && max_seconds <= sec[1]))
        return sm_ctx_fail_range(c, "max_seconds", max_seconds, sec[0], sec[1]);
    if (!max_variants || max_variants > SM_MAX_VARIANTS)
        return sm_ctx_fail_range(c, "max_variants", max_variants, 1, SM_MAX_VARIANTS);
    const uint32_t most = c->m->max_frames / c->m->frames_per_token, L = sm_tokens_of(c->m, max_seconds);
    return sm_ctx_fit(c, L + SM_TIE < most ? L + SM_TIE : most, L + 2 * SM_WRAP);
}

size_t sm_ctx_memory_bytes(const sm_ctx *c) { return c ? sizeof *c + sizeof *c->tw + c->n_scratch + c->n_dec : 0; }
const char *sm_ctx_last_error(const sm_ctx *c) { return c ? c->err : "context is NULL"; }
void sm_ctx_cancel(sm_ctx *c) { if (c) atomic_store(&c->cancel, 1); }
void sm_ctx_clear_cancel(sm_ctx *c) { if (c) atomic_store(&c->cancel, 0); }
void sm_ctx_set_progress(sm_ctx *c, sm_progress_fn fn, void *user) {
    if (!c) return;
    c->progress = fn;
    c->user = user;
}

sm_status sm_ctx_encode(sm_ctx *c, const char *utf8, size_t len, sm_text *t) {
    const sm_model *m = c->m;
    int32_t ids[64];
    const int32_t n = sm_tok_encode(&m->tok, utf8, len, c->tw, ids);
    if (n < 0) return sm_ctx_fail(c, SM_E_ARG, "text is not valid UTF-8");
    const sm_status s = sm_ctx_fit(c, 0, 0);
    if (s != SM_OK) return s;
    sm_text_encode(&m->text, ids, (uint32_t)n, t->vec, t->words, c->scratch);
    t->W = (uint32_t)n;
    t->ready = 1;
    sm_tok_unknown(&m->tok, c->tw, ids, n, t->unknown, sizeof t->unknown, &t->n_unknown);
    return SM_OK;
}

sm_status sm_ctx_encode_text(sm_ctx *c, const char *utf8, int32_t len, sm_text *out) {
    if (no_ctx(c)) return SM_E_ARG;
    size_t n;
    if (!out || out->m != c->m) return sm_ctx_fail(c, SM_E_ARG, "out is NULL or from another model");
    if (!c->m->gen.layers) return sm_ctx_fail(c, SM_E_UNSUPPORTED, "this model file has no text encoder");
    if (c->active) return sm_ctx_fail(c, SM_E_STATE, SM_ACTIVE);
    if (sm_check_text(utf8, len, &n, c->err, sizeof c->err) != SM_OK) return SM_E_ARG;
    return sm_ctx_encode(c, utf8, n, out);
}

/* the decoder's window: K context tokens (a prefix's, or a loop's end wrapped around), the codes, a loop's start */
static void fill_window(const int16_t *codes, uint32_t n, const sm_prefix *pre, uint32_t K, uint32_t Lw,
                        int16_t *window) {
    for (uint32_t i = 0; i < Lw; i++) {
        const uint32_t at = (i + n * SM_WRAP - K) % n;
        const int16_t *src = pre && i < K ? pre->codes + i * SM_VQ_LAYERS : codes + at * SM_VQ_LAYERS;
        memcpy(window + i * SM_VQ_LAYERS, src, sizeof(int16_t) * SM_VQ_LAYERS);
    }
}

/* fade out prev's own pose offset from its last frame as this decode sees it */
static void fade_from_prefix(sm_motion *mo, const sm_prefix *pre, const float *seen) {
    for (uint32_t t = 0; t < FADE && t < mo->frames; t++) {
        const float w = 1 - (float)(t + 1) / (FADE + 1);
        for (size_t i = 3; i < SM_F; i++) mo->feat[t * SM_F + i] += (pre->last[i] - seen[i]) * w;
    }
}

/* a prefix clip's last tokens give the decoder its left context and the root its start; a loop's window wraps */
sm_status sm_ctx_decode(sm_ctx *c, const int16_t *codes, uint32_t n, const sm_prefix *pre, const double *v,
                        const sm_path *path, sm_motion *mo) {
    if (pre && !pre->n) pre = NULL;
    const uint32_t K = pre ? pre->n : v[SM_P_LOOP] != 0 ? SM_WRAP : 0, Lw = K + n + (pre ? 0 : K);
    sm_status s = sm_ctx_fit(c, 0, Lw);
    if (s != SM_OK) return s;
    s = sm_motion_fit(mo, n);
    if (s != SM_OK) return sm_ctx_fail(c, s, "out of memory");

    float *vq_scratch = c->dec, *feat = vq_scratch + sm_vq_scratch_floats(Lw);
    double *tmp = (double *)(void *)(feat + 2 * (size_t)Lw * SM_F);
    int16_t *window = (int16_t *)(void *)(tmp + SM_POST_TMP(2 * Lw));
    fill_window(codes, n, pre, K, Lw, window);
    sm_vq_decode(&c->m->vq, window, Lw, feat, vq_scratch);

    if (codes != mo->codes) memcpy(mo->codes, codes, sizeof(int16_t) * n * SM_VQ_LAYERS);
    memcpy(mo->feat, feat + 2 * (size_t)K * SM_F, sizeof(float) * 2 * n * SM_F);
    mo->tokens = n;
    mo->frames = 2 * n;
    sm_track track = {0};
    if (pre) {
        const float *seen = feat + (2 * (size_t)K - 1) * SM_F;
        fade_from_prefix(mo, pre, seen);
        track = pre->track;
        memcpy(track.vel, seen, sizeof track.vel);   /* into the first frame: prev's own end velocity is poor */
    }
    sm_motion_post(mo, &track, v, path, tmp);
    return SM_OK;
}

static void params_or_defaults(const sm_ctx *c, const sm_request *r, double *v) {
    if (r) memcpy(v, r->v, sizeof(double) * SM_N_PARAMS);
    else sm_param_defaults(c->m, v);
}

/* codes not generated in this call: certain, and no seed */
static sm_status decode_plain(sm_ctx *c, const sm_request *r, const int16_t *codes, uint32_t n, sm_motion *out) {
    double v[SM_N_PARAMS];
    params_or_defaults(c, r, v);
    const sm_status s = sm_ctx_decode(c, codes, n, NULL, v, r ? &r->path : NULL, out);
    if (s != SM_OK) return s;
    for (uint32_t i = 0; i < n; i++) out->conf[i] = 1;
    out->seed = -1;
    return SM_OK;
}

sm_status sm_ctx_decode_codes(sm_ctx *c, const sm_request *r, const int16_t *codes, uint32_t n_tokens, sm_motion *out) {
    if (no_ctx(c)) return SM_E_ARG;
    if (!codes || !out || out->m != c->m || (r && r->m != c->m))
        return sm_ctx_fail(c, SM_E_ARG, "codes or out NULL, or out or r from another model");
    if (!n_tokens || n_tokens > SM_MAX_CODES) return sm_ctx_fail_range(c, "n_tokens", n_tokens, 1, SM_MAX_CODES);
    for (size_t i = 0; i < (size_t)n_tokens * SM_VQ_LAYERS; i++) {
        const int lo = i % SM_VQ_LAYERS ? -1 : 0;
        if (codes[i] < lo || codes[i] >= SM_VQ_N)
            return sm_ctx_fail_range(c, lo ? "a residual code" : "a base code", codes[i], lo, SM_VQ_N - 1);
    }
    if (r && sm_path_check(c->m, &r->path, n_tokens, c->err, sizeof c->err) != SM_OK) return SM_E_ARG;
    return decode_plain(c, r, codes, n_tokens, out);
}

sm_status sm_ctx_encode_motion(sm_ctx *c, const sm_request *r, const float *joints, uint32_t n_frames,
                               sm_motion *out) {
    if (no_ctx(c)) return SM_E_ARG;
    if (!joints || !out || out->m != c->m || (r && r->m != c->m))
        return sm_ctx_fail(c, SM_E_ARG, "joints or out NULL, or out or r from another model");
    if (!c->m->vq.has_enc) return sm_ctx_fail(c, SM_E_UNSUPPORTED, "this model file has no motion encoder");
    if (n_frames < 3 || (n_frames - 1) / 2 > SM_MAX_CODES)
        return sm_ctx_fail_range(c, "n_frames", n_frames, 3, 2 * SM_MAX_CODES + 1);
    for (size_t i = 0; i < (size_t)n_frames * SM_J * 3; i++)
        if (!isfinite(joints[i])) return sm_ctx_fail(c, SM_E_ARG, "joints hold NaN or Inf");
    const uint32_t T = n_frames - 1, L = T / 2;   /* features are steps between frames; 2 per token */
    if (r && sm_path_check(c->m, &r->path, L, c->err, sizeof c->err) != SM_OK) return SM_E_ARG;
    const size_t tmp = sizeof(double) * sm_feat_tmp_doubles(n_frames);
    const size_t floats = (size_t)T * SM_F + sm_vq_enc_scratch_floats(T);
    sm_status s = grow(c->m, &c->dec, &c->n_dec, tmp + sizeof(float) * floats);
    if (s == SM_OK) s = sm_motion_fit(out, L);
    if (s != SM_OK) return sm_ctx_fail(c, s, "out of memory");
    out->tokens = out->frames = 0;
    float *feat = (float *)(void *)((char *)c->dec + tmp);
    sm_joints_to_features(joints, n_frames, c->dec, feat);
    for (size_t i = 0; i < (size_t)T * SM_F; i++)
        if (!isfinite(feat[i]))
            return sm_ctx_fail(c, SM_E_NUMERIC, "the joints have no pose to read (zero-length bones?)");
    sm_vq_encode(&c->m->vq, feat, T, out->codes, feat + (size_t)T * SM_F);
    return decode_plain(c, r, out->codes, L, out);
}

#define SEED_RANGE 2147483648   /* 2^31 */

static int32_t seed_of(const sm_ctx *c, uint32_t i) { return (int32_t)(((int64_t)c->v[SM_P_SEED] + i) % SEED_RANGE); }

sm_status sm_ctx_next_pass(sm_ctx *c) {
    if (atomic_load(&c->cancel)) return sm_ctx_fail(c, SM_E_CANCELLED, SM_CANCELLED_MSG);
    return sm_gen_pass(&c->m->gen, &c->run, c->scratch);
}

sm_status sm_ctx_report(sm_ctx *c, uint32_t done, uint32_t total) {
    if (c->progress && c->progress(c->user, done, total))
        return sm_ctx_fail(c, SM_E_CANCELLED, "stopped by the progress callback");
    return SM_OK;
}

sm_status sm_ctx_start_run(sm_ctx *c, const double *v, uint32_t L, uint32_t tie, int32_t seed) {
    const sm_sampling s = {(uint32_t)v[SM_P_STEPS], (uint32_t)v[SM_P_RES_LAYERS], (uint32_t)v[SM_P_TOP_K],
                           (float)v[SM_P_CFG], (float)v[SM_P_RES_CFG], (float)v[SM_P_TEMP]};
    c->run = (sm_run){c->text->W, L, 0, 0, s, (uint64_t)seed, c->codes, c->conf, c->pins, tie};
    const sm_status st = sm_gen_start(&c->m->gen, &c->run, c->text->vec, c->text->words, c->scratch);
    return st == SM_OK ? SM_OK : sm_ctx_fail(c, st, "bad sampling settings");
}

static sm_status start_variant(sm_ctx *c) { return sm_ctx_start_run(c, c->v, c->L, c->tie, seed_of(c, c->at)); }

static void emit(sm_ctx *c, uint32_t i) {   /* the current codes of variant i, without the prefix tokens */
    sm_motion *mo = c->outs[i];
    const uint32_t skip = c->skip;
    memcpy(mo->codes, c->codes + skip * SM_VQ_LAYERS, sizeof(int16_t) * c->fresh * SM_VQ_LAYERS);
    memcpy(mo->conf, c->conf + skip, sizeof(float) * c->fresh);
    mo->tokens = c->fresh;
    mo->frames = 0;
    mo->seed = seed_of(c, i);
}

static sm_status check_run(sm_ctx *c, const sm_request *r, sm_motion *const *outs, uint32_t n) {
    if (c->active) return sm_ctx_fail(c, SM_E_STATE, SM_ACTIVE);
    if (!r || !outs || r->m != c->m) return sm_ctx_fail(c, SM_E_ARG, "r or outs NULL, or r from another model");
    if (!n || n > SM_MAX_VARIANTS) return sm_ctx_fail_range(c, "n_variants", n, 1, SM_MAX_VARIANTS);
    for (uint32_t i = 0; i < n; i++)
        if (!outs[i] || outs[i]->m != c->m)
            return sm_ctx_fail(c, SM_E_ARG, "an output motion is NULL or from another model");
    if (!c->m->gen.layers) return sm_ctx_fail(c, SM_E_UNSUPPORTED, "this model file has no generator");
    if (!r->has_text) return sm_ctx_fail(c, SM_E_ARG, "no text: call sm_request_set_text");
    if (atomic_load(&c->cancel)) return sm_ctx_fail(c, SM_E_CANCELLED, SM_CANCELLED_MSG);
    const int loop = r->v[SM_P_LOOP] != 0;
    if (loop && r->pre.n) return sm_ctx_fail(c, SM_E_ARG, "a loop with a prefix: a loop has no start to continue from");
    const uint32_t fresh = sm_tokens_of(c->m, r->v[SM_P_SECONDS]), skip = r->pre.n ? SM_OVERLAP : 0;
    const uint32_t tie = loop ? SM_TIE : 0, L = skip + fresh + tie;
    const uint32_t cap = c->m->gen.max_tokens < SM_MAX_RUN ? c->m->gen.max_tokens : SM_MAX_RUN;
    const double lo = c->m->param[SM_P_SECONDS][0];
    const double hi = (double)(cap - (L - fresh)) * c->m->frames_per_token / c->m->fps;
    if (L > cap)
        return sm_ctx_fail_range(c, skip ? "seconds with a prefix" : loop ? "seconds of a loop" : "seconds",
                                 r->v[SM_P_SECONDS],
                      lo, hi);
    if (r->n_pins > fresh) return sm_ctx_fail_range(c, "pinned tokens", r->n_pins, 0, fresh);
    const sm_status s = sm_path_check(c->m, &r->path, fresh, c->err, sizeof c->err);
    if (s != SM_OK) return s;
    c->fresh = fresh;
    c->L = L;
    c->skip = skip;
    c->tie = tie;
    return SM_OK;
}

/* the prefix's base codes, then the request's pinned tokens */
static void pin_run(sm_ctx *c, const sm_request *r) {
    for (uint32_t j = 0; j < c->L; j++) {
        int16_t *pin = c->pins + j * SM_VQ_LAYERS;
        const uint32_t fresh = j - c->skip;
        pin[0] = j < c->skip ? c->pre.codes[(c->pre.n - SM_OVERLAP + j) * SM_VQ_LAYERS] : -1;
        for (int q = 1; q < SM_VQ_LAYERS; q++) pin[q] = SM_FREE;
        if (j >= c->skip && fresh < r->n_pins && r->pins[fresh * SM_VQ_LAYERS] >= 0)
            memcpy(pin, r->pins + fresh * SM_VQ_LAYERS, sizeof(int16_t) * SM_VQ_LAYERS);
    }
}

sm_status sm_ctx_begin(sm_ctx *c, const sm_request *r, sm_motion *const *outs, uint32_t n) {
    if (no_ctx(c)) return SM_E_ARG;
    sm_status s = check_run(c, r, outs, n);
    if (s == SM_OK) s = sm_ctx_fit(c, c->L, c->L + r->pre.n);
    for (uint32_t i = 0; s == SM_OK && i < n; i++) {
        s = sm_motion_fit(outs[i], c->fresh);
        if (s != SM_OK) sm_ctx_fail(c, s, "out of memory");
    }
    if (s == SM_OK && r->has_text == 2) sm_text_copy(c->text, r->enc);
    else if (s == SM_OK) s = sm_ctx_encode(c, r->text, r->text_len, c->text);
    if (s != SM_OK) return s;

    memcpy(c->v, r->v, sizeof c->v);
    c->pre = r->pre;
    c->path = r->path;
    pin_run(c, r);
    for (uint32_t i = 0; i < n; i++) {
        c->outs[i] = outs[i];
        outs[i]->tokens = outs[i]->frames = 0;
    }
    c->n = n;
    c->at = 0;
    c->active = 1;
    s = start_variant(c);
    if (s != SM_OK) sm_ctx_abort(c);
    return s;
}

sm_status sm_ctx_step(sm_ctx *c, uint32_t *done, uint32_t *total) {
    if (no_ctx(c)) return SM_E_ARG;
    if (!c->active || c->at == c->n)
        return sm_ctx_fail(c, SM_E_STATE, c->active ? "the run is done: call sm_ctx_finish" : NO_RUN);
    sm_status s = sm_ctx_next_pass(c);
    if (s == SM_DONE) {
        emit(c, c->at);
        s = ++c->at < c->n ? start_variant(c) : SM_DONE;
    }
    const uint32_t per = c->run.s.steps + c->run.s.res_layers, d = c->at * per + (c->at < c->n ? c->run.pass : 0);
    if (s >= 0 && sm_ctx_report(c, d, c->n * per) != SM_OK) s = SM_E_CANCELLED;
    if (s < 0) {
        if (!c->err[0]) sm_ctx_fail(c, s, sm_status_message(s));
        sm_ctx_abort(c);
        return s;
    }
    if (done) *done = d;
    if (total) *total = c->n * per;
    return s;
}

sm_status sm_ctx_preview(sm_ctx *c) {
    if (no_ctx(c)) return SM_E_ARG;
    if (!c->active) return sm_ctx_fail(c, SM_E_STATE, NO_RUN);
    if (c->at < c->n && c->run.pass) emit(c, c->at);
    sm_status s = SM_OK;
    for (uint32_t i = 0; s == SM_OK && i < c->n; i++)
        if (c->outs[i]->tokens) s = sm_ctx_decode(c, c->outs[i]->codes, c->fresh, &c->pre, c->v, &c->path, c->outs[i]);
    return s;
}

sm_status sm_ctx_finish(sm_ctx *c) {
    if (no_ctx(c)) return SM_E_ARG;
    if (!c->active || c->at < c->n)
        return sm_ctx_fail(c, SM_E_STATE, "the run is not done: call sm_ctx_step until SM_DONE");
    sm_status s = SM_OK;
    for (uint32_t i = 0; s == SM_OK && i < c->n; i++)
        s = sm_ctx_decode(c, c->outs[i]->codes, c->fresh, &c->pre, c->v, &c->path, c->outs[i]);
    if (s != SM_OK) sm_ctx_abort(c);
    c->active = 0;
    return s;
}

void sm_ctx_abort(sm_ctx *c) {
    for (uint32_t i = 0; c && c->active && i < c->n; i++) c->outs[i]->tokens = c->outs[i]->frames = 0;
    if (c) c->active = 0;
}

sm_status sm_ctx_generate(sm_ctx *c, const sm_request *r, sm_motion *const *outs, uint32_t n_variants) {
    sm_status s = sm_ctx_begin(c, r, outs, n_variants);
    while (s == SM_OK) s = sm_ctx_step(c, NULL, NULL);
    return s == SM_DONE ? sm_ctx_finish(c) : s;
}

sm_status sm_generate_simple(const sm_model *m, const char *utf8, double seconds, int32_t seed, sm_motion *out,
                             char *err, size_t err_len) {
    sm_ctx *c = NULL;
    sm_request *r = NULL;
    if (err && err_len) err[0] = 0;
    sm_status s = !m || !out ? sm_fail(err, err_len, SM_E_ARG, "m and out must not be NULL") : sm_ctx_create(m, &c);
    if (s == SM_OK) s = sm_request_create(m, &r);
    if (s == SM_OK) s = sm_request_set_text(r, utf8, -1);
    if (s == SM_OK) s = sm_request_set_float(r, SM_P_SECONDS, seconds);
    if (s == SM_OK) s = sm_request_set_int(r, SM_P_SEED, seed);
    if (s == SM_OK) s = sm_ctx_generate(c, r, &out, 1);
    if (s != SM_OK && err && err_len && !err[0])
        sm_fail(err, err_len, s, r && r->err[0] ? r->err : c && c->err[0] ? c->err : sm_status_message(s));
    sm_request_free(r);
    sm_ctx_free(c);
    return s;
}

