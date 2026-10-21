/* Contexts: scratch memory, text encoding, generation one pass at a time, decoding into motions. */
#include "sm_internal.h"
#include <stdatomic.h>

#define MAX_VARIANTS 16
#define MAX_RUN 512   /* tokens per run */
#define FADE 8        /* frames over which a prefix clip's end fades into the new one */
#define TIE 12        /* tokens a loop makes past its end as its start again, then drops: 1.2 s of the wrap seen */
#define WRAP 16       /* tokens of a loop's decode window past each end: more than the decoder's reach (15) */
#define ACTIVE "a run is active: finish or abort it first"

struct sm_ctx {
    const sm_model *m;
    atomic_int cancel;
    sm_progress_fn progress;
    void *user;
    void *scratch, *dec;                /* text encoder or generator; decoder */
    size_t n_scratch, n_dec;            /* bytes */
    sm_tok_work *tw;
    sm_text *text;
    int active;
    uint32_t n, at, L, fresh, skip, tie;   /* variants, the one running, tokens made, kept, the prefix's, the wrap's */
    double v[SM_N_PARAMS];
    sm_prefix pre;
    sm_path path;
    sm_run run;
    sm_motion *outs[MAX_VARIANTS];
    int16_t codes[MAX_RUN * SM_VQ_LAYERS];
    float conf[MAX_RUN];
    int16_t pins[MAX_RUN * SM_VQ_LAYERS];
    char err[128];
};

static sm_status cfail(sm_ctx *c, sm_status s, const char *msg) { return sm_fail(c->err, sizeof c->err, s, msg); }
static sm_status crange(sm_ctx *c, const char *name, double v, double lo, double hi) {
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
static sm_status fit(sm_ctx *c, uint32_t L, uint32_t Lw) {
    const sm_model *m = c->m;
    const size_t T = 2 * (size_t)Lw, gen = m->gen.layers && L ? sm_gen_scratch_floats(&m->gen, m->tok.max_len, L) : 0;
    const size_t text = m->gen.layers ? sm_text_scratch_floats(&m->text) : 0;
    const size_t dec = sizeof(float) * (sm_vq_scratch_floats(Lw) + T * SM_F) + sizeof(double) * SM_POST_TMP(T) +
                       sizeof(int16_t) * Lw * SM_VQ_LAYERS;
    sm_status s = grow(m, &c->scratch, &c->n_scratch, sizeof(float) * (gen > text ? gen : text));
    if (s == SM_OK) s = grow(m, &c->dec, &c->n_dec, dec);
    return s == SM_OK ? SM_OK : cfail(c, s, "out of memory");
}

sm_status sm_ctx_create(const sm_model *m, sm_ctx **out) {
    sm_ctx *c = m && out ? sm_alloc(&m->h, sizeof *c) : NULL;
    if (!c) return m && out ? SM_E_NOMEM : SM_E_ARG;
    memset(c, 0, sizeof *c);
    c->m = m;
    atomic_init(&c->cancel, 0);
    c->tw = sm_alloc(&m->h, sizeof *c->tw);
    if (!c->tw || sm_text_create(m, &c->text) != SM_OK) return sm_ctx_free(c), SM_E_NOMEM;
    *out = c;
    return SM_OK;
}

void sm_ctx_free(sm_ctx *c) {
    if (!c) return;
    sm_text_free(c->text);
    void *owned[] = {c->scratch, c->dec, c->tw, c};
    for (size_t i = 0; i < sizeof owned / sizeof owned[0]; i++) sm_free(&c->m->h, owned[i]);
}

static uint32_t tokens_of(const sm_model *m, double seconds) {
    return (uint32_t)(seconds * m->fps / m->frames_per_token + 0.5);
}

sm_status sm_ctx_reserve(sm_ctx *c, double max_seconds, uint32_t max_variants) {
    if (no_ctx(c)) return SM_E_ARG;
    const double *sec = c->m->param[SM_P_SECONDS];
    if (!(max_seconds >= sec[0] && max_seconds <= sec[1])) return crange(c, "max_seconds", max_seconds, sec[0], sec[1]);
    if (!max_variants || max_variants > MAX_VARIANTS) return crange(c, "max_variants", max_variants, 1, MAX_VARIANTS);
    const uint32_t most = c->m->max_frames / c->m->frames_per_token, L = tokens_of(c->m, max_seconds);
    return fit(c, L + TIE < most ? L + TIE : most, L + 2 * WRAP);
}

size_t sm_ctx_memory_bytes(const sm_ctx *c) { return c ? sizeof *c + sizeof *c->tw + c->n_scratch + c->n_dec : 0; }
const char *sm_ctx_last_error(const sm_ctx *c) { return c ? c->err : "context is NULL"; }
void sm_ctx_cancel(sm_ctx *c) { if (c) atomic_store(&c->cancel, 1); }
void sm_ctx_clear_cancel(sm_ctx *c) { if (c) atomic_store(&c->cancel, 0); }
void sm_ctx_set_progress(sm_ctx *c, sm_progress_fn fn, void *user) { if (c) c->progress = fn, c->user = user; }

static sm_status encode(sm_ctx *c, const char *utf8, size_t len, sm_text *t) {
    const sm_model *m = c->m;
    int32_t ids[64];
    const int32_t n = sm_tok_encode(&m->tok, utf8, len, c->tw, ids);
    if (n < 0) return cfail(c, SM_E_ARG, "text is not valid UTF-8");
    const sm_status s = fit(c, 0, 0);
    if (s != SM_OK) return s;
    sm_text_encode(&m->text, ids, (uint32_t)n, t->vec, t->words, c->scratch);
    t->W = (uint32_t)n, t->ready = 1;
    sm_tok_unknown(&m->tok, c->tw, ids, n, t->unknown, sizeof t->unknown, &t->n_unknown);
    return SM_OK;
}

sm_status sm_ctx_encode_text(sm_ctx *c, const char *utf8, int32_t len, sm_text *out) {
    if (no_ctx(c)) return SM_E_ARG;
    size_t n;
    if (!out || out->m != c->m) return cfail(c, SM_E_ARG, "out is NULL or from another model");
    if (!c->m->gen.layers) return cfail(c, SM_E_UNSUPPORTED, "this model file has no text encoder");
    if (c->active) return cfail(c, SM_E_STATE, ACTIVE);
    if (sm_check_text(utf8, len, &n, c->err, sizeof c->err) != SM_OK) return SM_E_ARG;
    return encode(c, utf8, n, out);
}

/* a prefix clip's last tokens give the decoder its left context and the root its start; a loop's window wraps */
static sm_status decode(sm_ctx *c, const int16_t *codes, uint32_t n, const sm_prefix *pre, const double *v,
                        const sm_path *path, sm_motion *mo) {
    if (pre && !pre->n) pre = NULL;
    const uint32_t K = pre ? pre->n : v[SM_P_LOOP] != 0 ? WRAP : 0, Lw = K + n + (pre ? 0 : K);
    sm_status s = fit(c, 0, Lw);
    if (s == SM_OK && (s = sm_motion_fit(mo, n)) != SM_OK) cfail(c, s, "out of memory");
    if (s != SM_OK) return s;
    float *vs = c->dec, *feat = vs + sm_vq_scratch_floats(Lw);
    double *tmp = (double *)(void *)(feat + 2 * (size_t)Lw * SM_F);
    int16_t *win = (int16_t *)(void *)(tmp + SM_POST_TMP(2 * Lw));
    for (uint32_t i = 0; i < Lw; i++) {
        const uint32_t at = (i + n * WRAP - K) % n;   /* a loop's window wraps */
        const int16_t *src = pre && i < K ? pre->codes + i * SM_VQ_LAYERS : codes + at * SM_VQ_LAYERS;
        memcpy(win + i * SM_VQ_LAYERS, src, sizeof(int16_t) * SM_VQ_LAYERS);
    }
    sm_vq_decode(&c->m->vq, win, Lw, feat, vs);
    if (codes != mo->codes) memcpy(mo->codes, codes, sizeof(int16_t) * n * SM_VQ_LAYERS);
    memcpy(mo->feat, feat + 2 * (size_t)K * SM_F, sizeof(float) * 2 * n * SM_F);
    const float *seen = pre ? feat + (2 * (size_t)K - 1) * SM_F : feat;   /* prev's last frame as this decode sees it */
    for (uint32_t t = 0; pre && t < FADE && t < 2 * n; t++) {     /* fade out prev's own pose offset from it */
        const float w = 1 - (float)(t + 1) / (FADE + 1);
        for (size_t i = 3; i < SM_F; i++) mo->feat[t * SM_F + i] += (pre->last[i] - seen[i]) * w;
    }
    sm_track tr = pre ? pre->track : (sm_track){0};
    if (pre) memcpy(tr.vel, seen, sizeof tr.vel);   /* the step into the first frame: prev's own end velocity is poor */
    mo->tokens = n, mo->frames = 2 * n;
    sm_motion_post(mo, &tr, v, path, tmp);
    return SM_OK;
}

sm_status sm_ctx_decode_codes(sm_ctx *c, const sm_request *r, const int16_t *codes, uint32_t n_tokens, sm_motion *out) {
    if (no_ctx(c)) return SM_E_ARG;
    if (!codes || !out || out->m != c->m || (r && r->m != c->m))
        return cfail(c, SM_E_ARG, "codes or out NULL, or out or r from another model");
    if (!n_tokens || n_tokens > SM_MAX_CODES) return crange(c, "n_tokens", n_tokens, 1, SM_MAX_CODES);
    for (size_t i = 0; i < (size_t)n_tokens * SM_VQ_LAYERS; i++) {
        const int lo = i % SM_VQ_LAYERS ? -1 : 0;
        if (codes[i] < lo || codes[i] >= SM_VQ_N)
            return crange(c, lo ? "a residual code" : "a base code", codes[i], lo, SM_VQ_N - 1);
    }
    double v[SM_N_PARAMS];
    if (r) memcpy(v, r->v, sizeof v);
    else sm_param_defaults(c->m, v);
    sm_status s = r ? sm_path_check(c->m, &r->path, n_tokens, c->err, sizeof c->err) : SM_OK;
    if (s == SM_OK) s = decode(c, codes, n_tokens, NULL, v, r ? &r->path : NULL, out);
    for (uint32_t i = 0; s == SM_OK && i < n_tokens; i++) out->conf[i] = 1;
    if (s == SM_OK) out->seed = -1;
    return s;
}

sm_status sm_ctx_encode_motion(sm_ctx *c, const sm_request *r, const float *joints, uint32_t n_frames,
                               sm_motion *out) {
    if (no_ctx(c)) return SM_E_ARG;
    if (!joints || !out || out->m != c->m || (r && r->m != c->m))
        return cfail(c, SM_E_ARG, "joints or out NULL, or out or r from another model");
    if (!c->m->vq.has_enc) return cfail(c, SM_E_UNSUPPORTED, "this model file has no motion encoder");
    if (n_frames < 3 || (n_frames - 1) / 2 > SM_MAX_CODES)
        return crange(c, "n_frames", n_frames, 3, 2 * SM_MAX_CODES + 1);
    for (size_t i = 0; i < (size_t)n_frames * SM_J * 3; i++)
        if (!isfinite(joints[i])) return cfail(c, SM_E_ARG, "joints hold NaN or Inf");
    const uint32_t T = n_frames - 1, L = T / 2;   /* features are steps between frames; 2 per token */
    if (r && sm_path_check(c->m, &r->path, L, c->err, sizeof c->err) != SM_OK) return SM_E_ARG;
    const size_t tmp = sizeof(double) * sm_feat_tmp_doubles(n_frames);
    const size_t floats = (size_t)T * SM_F + sm_vq_enc_scratch_floats(T);
    sm_status s = grow(c->m, &c->dec, &c->n_dec, tmp + sizeof(float) * floats);
    if (s == SM_OK) s = sm_motion_fit(out, L);
    if (s != SM_OK) return cfail(c, s, "out of memory");
    out->tokens = out->frames = 0;
    float *feat = (float *)(void *)((char *)c->dec + tmp);
    sm_joints_to_features(joints, n_frames, c->dec, feat);
    for (size_t i = 0; i < (size_t)T * SM_F; i++)
        if (!isfinite(feat[i])) return cfail(c, SM_E_NUMERIC, "the joints have no pose to read (zero-length bones?)");
    sm_vq_encode(&c->m->vq, feat, T, out->codes, feat + (size_t)T * SM_F);
    double v[SM_N_PARAMS];
    if (r) memcpy(v, r->v, sizeof v);
    else sm_param_defaults(c->m, v);
    if ((s = decode(c, out->codes, L, NULL, v, r ? &r->path : NULL, out)) != SM_OK) return s;
    for (uint32_t i = 0; i < L; i++) out->conf[i] = 1;
    out->seed = -1;
    return SM_OK;
}

static int32_t seed_of(const sm_ctx *c, uint32_t i) { return (int32_t)(((int64_t)c->v[SM_P_SEED] + i) % 2147483648); }

static sm_status start(sm_ctx *c, const double *v, uint32_t L, uint32_t tie, int32_t seed) {
    const sm_sampling s = {(uint32_t)v[SM_P_STEPS], (uint32_t)v[SM_P_RES_LAYERS], (uint32_t)v[SM_P_TOP_K],
                           (float)v[SM_P_CFG], (float)v[SM_P_RES_CFG], (float)v[SM_P_TEMP]};
    c->run = (sm_run){c->text->W, L, 0, 0, s, (uint64_t)seed, c->codes, c->conf, c->pins, tie};
    const sm_status st = sm_gen_start(&c->m->gen, &c->run, c->text->vec, c->text->words, c->scratch);
    return st == SM_OK ? SM_OK : cfail(c, st, "bad sampling settings");
}

static sm_status start_variant(sm_ctx *c) { return start(c, c->v, c->L, c->tie, seed_of(c, c->at)); }

static void emit(sm_ctx *c, uint32_t i) {   /* the current codes of variant i, without the prefix tokens */
    sm_motion *mo = c->outs[i];
    const uint32_t skip = c->skip;
    memcpy(mo->codes, c->codes + skip * SM_VQ_LAYERS, sizeof(int16_t) * c->fresh * SM_VQ_LAYERS);
    memcpy(mo->conf, c->conf + skip, sizeof(float) * c->fresh);
    mo->tokens = c->fresh, mo->frames = 0, mo->seed = seed_of(c, i);
}

static sm_status check_run(sm_ctx *c, const sm_request *r, sm_motion *const *outs, uint32_t n) {
    if (c->active) return cfail(c, SM_E_STATE, ACTIVE);
    if (!r || !outs || r->m != c->m) return cfail(c, SM_E_ARG, "r or outs NULL, or r from another model");
    if (!n || n > MAX_VARIANTS) return crange(c, "n_variants", n, 1, MAX_VARIANTS);
    for (uint32_t i = 0; i < n; i++)
        if (!outs[i] || outs[i]->m != c->m) return cfail(c, SM_E_ARG, "an output motion is NULL or from another model");
    if (!c->m->gen.layers) return cfail(c, SM_E_UNSUPPORTED, "this model file has no generator");
    if (!r->has_text) return cfail(c, SM_E_ARG, "no text: call sm_request_set_text");
    if (atomic_load(&c->cancel)) return cfail(c, SM_E_CANCELLED, "cancel is set: call sm_ctx_clear_cancel");
    const int loop = r->v[SM_P_LOOP] != 0;
    if (loop && r->pre.n) return cfail(c, SM_E_ARG, "a loop with a prefix: a loop has no start to continue from");
    const uint32_t fresh = tokens_of(c->m, r->v[SM_P_SECONDS]), skip = r->pre.n ? SM_OVERLAP : 0;
    const uint32_t tie = loop ? TIE : 0, L = skip + fresh + tie;
    const uint32_t cap = c->m->gen.max_tokens < MAX_RUN ? c->m->gen.max_tokens : MAX_RUN;
    const double lo = c->m->param[SM_P_SECONDS][0];
    const double hi = (double)(cap - (L - fresh)) * c->m->frames_per_token / c->m->fps;
    if (L > cap)
        return crange(c, skip ? "seconds with a prefix" : loop ? "seconds of a loop" : "seconds", r->v[SM_P_SECONDS],
                      lo, hi);
    if (r->n_pins > fresh) return crange(c, "pinned tokens", r->n_pins, 0, fresh);
    const sm_status s = sm_path_check(c->m, &r->path, fresh, c->err, sizeof c->err);
    if (s != SM_OK) return s;
    c->fresh = fresh, c->L = L, c->skip = skip, c->tie = tie;
    return SM_OK;
}

sm_status sm_ctx_begin(sm_ctx *c, const sm_request *r, sm_motion *const *outs, uint32_t n) {
    if (no_ctx(c)) return SM_E_ARG;
    sm_status s = check_run(c, r, outs, n);
    if (s == SM_OK) s = fit(c, c->L, c->L + r->pre.n);
    for (uint32_t i = 0; s == SM_OK && i < n; i++)
        if ((s = sm_motion_fit(outs[i], c->fresh)) != SM_OK) cfail(c, s, "out of memory");
    if (s == SM_OK && r->has_text == 2) sm_text_copy(c->text, r->enc);
    else if (s == SM_OK) s = encode(c, r->text, r->text_len, c->text);
    if (s != SM_OK) return s;
    memcpy(c->v, r->v, sizeof c->v);
    c->pre = r->pre, c->path = r->path;
    for (uint32_t j = 0; j < c->L; j++) {   /* the prefix's base codes, then the request's pinned tokens */
        int16_t *p = c->pins + j * SM_VQ_LAYERS;
        const uint32_t f = j - c->skip;
        p[0] = j < c->skip ? c->pre.codes[(c->pre.n - SM_OVERLAP + j) * SM_VQ_LAYERS] : -1;
        for (int q = 1; q < SM_VQ_LAYERS; q++) p[q] = SM_FREE;
        if (j >= c->skip && f < r->n_pins && r->pins[f * SM_VQ_LAYERS] >= 0)
            memcpy(p, r->pins + f * SM_VQ_LAYERS, sizeof(int16_t) * SM_VQ_LAYERS);
    }
    for (uint32_t i = 0; i < n; i++) c->outs[i] = outs[i], outs[i]->tokens = outs[i]->frames = 0;
    c->n = n, c->at = 0, c->active = 1;
    if ((s = start_variant(c)) != SM_OK) sm_ctx_abort(c);
    return s;
}

sm_status sm_ctx_step(sm_ctx *c, uint32_t *done, uint32_t *total) {
    if (no_ctx(c)) return SM_E_ARG;
    if (!c->active || c->at == c->n)
        return cfail(c, SM_E_STATE, c->active ? "the run is done: call sm_ctx_finish" : "no run: call sm_ctx_begin");
    sm_status s = atomic_load(&c->cancel) ? cfail(c, SM_E_CANCELLED, "cancel is set: call sm_ctx_clear_cancel")
                                          : sm_gen_pass(&c->m->gen, &c->run, c->scratch);
    if (s == SM_DONE) {
        emit(c, c->at);
        s = ++c->at < c->n ? start_variant(c) : SM_DONE;
    }
    const uint32_t per = c->run.s.steps + c->run.s.res_layers, d = c->at * per + (c->at < c->n ? c->run.pass : 0);
    if (s >= 0 && c->progress && c->progress(c->user, d, c->n * per))
        s = cfail(c, SM_E_CANCELLED, "stopped by the progress callback");
    if (s < 0) {
        if (!c->err[0]) cfail(c, s, sm_status_message(s));
        sm_ctx_abort(c);
        return s;
    }
    if (done) *done = d;
    if (total) *total = c->n * per;
    return s;
}

sm_status sm_ctx_preview(sm_ctx *c) {
    if (no_ctx(c)) return SM_E_ARG;
    if (!c->active) return cfail(c, SM_E_STATE, "no run: call sm_ctx_begin");
    if (c->at < c->n && c->run.pass) emit(c, c->at);
    sm_status s = SM_OK;
    for (uint32_t i = 0; s == SM_OK && i < c->n; i++)
        if (c->outs[i]->tokens) s = decode(c, c->outs[i]->codes, c->fresh, &c->pre, c->v, &c->path, c->outs[i]);
    return s;
}

sm_status sm_ctx_finish(sm_ctx *c) {
    if (no_ctx(c)) return SM_E_ARG;
    if (!c->active || c->at < c->n) return cfail(c, SM_E_STATE, "the run is not done: call sm_ctx_step until SM_DONE");
    sm_status s = SM_OK;
    for (uint32_t i = 0; s == SM_OK && i < c->n; i++)
        s = decode(c, c->outs[i]->codes, c->fresh, &c->pre, c->v, &c->path, c->outs[i]);
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

#define MAX_BLOCKS 256
#define AHEAD 8   /* tokens made past a block whose next one is made later, then dropped: its end is not a clip's end */

typedef struct { const char *text; const sm_text *enc; double seconds; int32_t seed; const int16_t *keep; } block;

/* the neighbour of block i on side dir whose codes are final when i is made (kept, or made before it), else n */
static uint32_t side(const block *b, uint32_t n, int loop, uint32_t i, int dir) {
    const uint32_t j = dir < 0 ? (i ? i - 1 : loop ? n - 1 : n) : (i + 1 < n ? i + 1 : loop ? 0 : n);
    return j < n && j != i && (b[j].keep || j < i) ? j : n;
}

/* a made block's run: the pinned seam tokens of final neighbours l and q, free tokens ahead, a lone loop's wrap */
typedef struct { uint32_t l, q, left, right, ahead, tie; } edges;
static edges edges_of(const block *b, uint32_t n, int loop, uint32_t i) {
    edges e = {side(b, n, loop, i, -1), side(b, n, loop, i, 1), 0, 0, 0, loop && n == 1 ? TIE : 0};
    e.left = e.l < n ? SM_OVERLAP : 0, e.right = e.q < n ? SM_OVERLAP : 0;
    e.ahead = !e.right && i + 1 < n ? AHEAD : 0;
    return e;
}

static sm_status check_block(sm_ctx *c, const block *b, uint32_t i, uint32_t *tokens) {
    const double *sec = c->m->param[SM_P_SECONDS];
    char name[48];
    size_t len;
    snprintf(name, sizeof name, "block %u seconds", i);
    if (!(b->seconds >= sec[0] && b->seconds <= sec[1])) return crange(c, name, b->seconds, sec[0], sec[1]);
    *tokens = tokens_of(c->m, b->seconds);
    for (size_t k = 0; b->keep && k < (size_t)*tokens * SM_VQ_LAYERS; k++)
        if (b->keep[k] < (k % SM_VQ_LAYERS ? -1 : 0) || b->keep[k] >= SM_VQ_N)
            return snprintf(c->err, sizeof c->err, "block %u: a kept code is out of range", i), SM_E_ARG;
    if (b->keep) return SM_OK;
    if (!c->m->gen.layers) return cfail(c, SM_E_UNSUPPORTED, "this model file has no generator");
    if (b->text) return sm_check_text(b->text, -1, &len, c->err, sizeof c->err);
    if (b->enc && b->enc->m == c->m && b->enc->ready) return SM_OK;
    return snprintf(c->err, sizeof c->err, "block %u: no text, or enc not encoded or from another model", i), SM_E_ARG;
}

/* every block valid, the timeline within one decode and each made block within one run; *L: the longest run */
static sm_status check_blocks(sm_ctx *c, const sm_request *r, const block *b, uint32_t n, const sm_motion *out,
                              uint32_t *tok, uint32_t *L) {
    if (c->active) return cfail(c, SM_E_STATE, ACTIVE);
    if (!r || !out || r->m != c->m || out->m != c->m) return cfail(c, SM_E_ARG, "r or out NULL or from another model");
    if (atomic_load(&c->cancel)) return cfail(c, SM_E_CANCELLED, "cancel is set: call sm_ctx_clear_cancel");
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; i++) {
        const sm_status s = check_block(c, &b[i], i, &tok[i]);
        if (s != SM_OK) return s;
        total += tok[i];
    }
    if (total > SM_MAX_CODES) return crange(c, "timeline tokens", total, 1, SM_MAX_CODES);
    const sm_status s = sm_path_check(c->m, &r->path, total, c->err, sizeof c->err);
    if (s != SM_OK) return s;
    const uint32_t cap = c->m->gen.max_tokens < MAX_RUN ? c->m->gen.max_tokens : MAX_RUN;
    for (uint32_t i = *L = 0; i < n; i++) {
        const edges e = edges_of(b, n, r->v[SM_P_LOOP] != 0, i);
        const uint32_t run = e.left + tok[i] + e.right + e.ahead + e.tie;
        if (!b[i].keep && run > cap)
            return snprintf(c->err, sizeof c->err, "block %u: %u tokens with seams, over %u", i, run, cap), SM_E_ARG;
        if (!b[i].keep && run > *L) *L = run;
    }
    return SM_OK;
}

typedef struct { const double *v; int16_t *codes; float *conf; const uint32_t *first; uint32_t done, total; } timeline;

/* block i's run with the seams pinned on all layers; its own tokens go into the timeline's codes */
static sm_status gen_block(sm_ctx *c, timeline *t, const block *b, uint32_t i, edges e) {
    const uint32_t n = t->first[i + 1] - t->first[i], L = e.left + n + e.right + e.ahead + e.tie;
    const int16_t *lt = t->codes + (e.left ? t->first[e.l + 1] - e.left : 0) * SM_VQ_LAYERS;
    const int16_t *rt = t->codes + (e.right ? t->first[e.q] : 0) * SM_VQ_LAYERS;
    sm_status s = b->text ? encode(c, b->text, strlen(b->text), c->text) : (sm_text_copy(c->text, b->enc), SM_OK);
    for (uint32_t j = 0; j < L; j++) {
        const uint32_t k = j - e.left - n;
        const int16_t *src = j < e.left ? lt + j * SM_VQ_LAYERS : k < e.right ? rt + k * SM_VQ_LAYERS : NULL;
        if (src) memcpy(c->pins + j * SM_VQ_LAYERS, src, sizeof(int16_t) * SM_VQ_LAYERS);
        else c->pins[j * SM_VQ_LAYERS] = -1;
    }
    if (s == SM_OK) s = start(c, t->v, L, e.tie, b->seed);
    while (s == SM_OK) {
        s = atomic_load(&c->cancel) ? cfail(c, SM_E_CANCELLED, "cancel is set: call sm_ctx_clear_cancel")
                                    : sm_gen_pass(&c->m->gen, &c->run, c->scratch);
        if (s >= 0 && c->progress && c->progress(c->user, ++t->done, t->total))
            s = cfail(c, SM_E_CANCELLED, "stopped by the progress callback");
    }
    if (s != SM_DONE) return s;
    memcpy(t->codes + t->first[i] * SM_VQ_LAYERS, c->codes + e.left * SM_VQ_LAYERS, sizeof(int16_t) * n * SM_VQ_LAYERS);
    memcpy(t->conf + t->first[i], c->conf + e.left, sizeof(float) * n);
    return SM_OK;
}

sm_status sm_ctx_generate_blocks(sm_ctx *c, const sm_request *r, uint32_t n_blocks, const char *const *texts,
                                 const sm_text *const *encs, const double *seconds, const int32_t *seeds,
                                 const int16_t *const *keep, uint32_t *first_frames, sm_motion *out) {
    if (no_ctx(c)) return SM_E_ARG;
    block b[MAX_BLOCKS];
    uint32_t tok[MAX_BLOCKS], first[MAX_BLOCKS + 1], L = 0;
    sm_status s = !n_blocks || n_blocks > MAX_BLOCKS ? crange(c, "n_blocks", n_blocks, 1, MAX_BLOCKS)
                  : !seconds || !seeds ? cfail(c, SM_E_ARG, "seconds or seeds NULL") : SM_OK;
    for (uint32_t i = 0; s == SM_OK && i < n_blocks; i++)
        b[i] = (block){texts ? texts[i] : NULL, encs ? encs[i] : NULL, seconds[i], seeds[i], keep ? keep[i] : NULL};
    if (s == SM_OK) s = check_blocks(c, r, b, n_blocks, out, tok, &L);
    if (s != SM_OK && out && out->m == c->m) out->tokens = out->frames = 0;
    for (uint32_t i = first[0] = 0; s == SM_OK && i < n_blocks; i++) first[i + 1] = first[i] + tok[i];
    if (s == SM_OK) s = fit(c, L, first[n_blocks]);   /* the decode's scratch holds the kept codes until out fits */
    const size_t row = sizeof(int16_t) * SM_VQ_LAYERS;
    int16_t *kept = s == SM_OK ? c->dec : NULL;
    for (uint32_t i = 0; kept && i < n_blocks; i++)
        if (b[i].keep) memcpy(kept + first[i] * SM_VQ_LAYERS, b[i].keep, row * tok[i]);
    if (s == SM_OK && (s = sm_motion_fit(out, first[n_blocks])) != SM_OK) cfail(c, s, "out of memory");
    if (s != SM_OK) return s;
    timeline t = {r->v, out->codes, out->conf, first, 0, 0};
    for (uint32_t i = 0; i < n_blocks; i++) {
        t.total += b[i].keep ? 0 : (uint32_t)(r->v[SM_P_STEPS] + r->v[SM_P_RES_LAYERS]);
        if (b[i].keep) memcpy(out->codes + first[i] * SM_VQ_LAYERS, kept + first[i] * SM_VQ_LAYERS, row * tok[i]);
        for (uint32_t j = first[i]; b[i].keep && j < first[i + 1]; j++) out->conf[j] = 1;
    }
    out->tokens = out->frames = 0;
    for (uint32_t i = 0; s == SM_OK && i < n_blocks; i++)
        if (!b[i].keep) s = gen_block(c, &t, &b[i], i, edges_of(b, n_blocks, r->v[SM_P_LOOP] != 0, i));
    if (s == SM_OK) s = decode(c, out->codes, first[n_blocks], NULL, r->v, &r->path, out);
    for (uint32_t i = 0; s == SM_OK && first_frames && i <= n_blocks; i++) first_frames[i] = 2 * first[i];
    if (s != SM_OK && !c->err[0]) cfail(c, s, sm_status_message(s));
    if (s != SM_OK) out->tokens = out->frames = 0;
    else out->seed = -1;
    return s;
}
