/* Timelines: prompt blocks back to back, each made with its final neighbours' seams pinned, decoded in one pass. */
#include "sm_internal.h"
#include <stdatomic.h>

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
    edges e = {side(b, n, loop, i, -1), side(b, n, loop, i, 1), 0, 0, 0, loop && n == 1 ? SM_TIE : 0};
    e.left = e.l < n ? SM_OVERLAP : 0;
    e.right = e.q < n ? SM_OVERLAP : 0;
    e.ahead = !e.right && i + 1 < n ? AHEAD : 0;
    return e;
}

static sm_status block_fail(sm_ctx *c, uint32_t i, const char *msg) {
    snprintf(c->err, sizeof c->err, "block %u: %s", i, msg);
    return SM_E_ARG;
}

static sm_status check_block(sm_ctx *c, const block *b, uint32_t i, uint32_t *tokens) {
    const double *sec = c->m->param[SM_P_SECONDS];
    char name[48];
    size_t len;
    snprintf(name, sizeof name, "block %u seconds", i);
    if (!(b->seconds >= sec[0] && b->seconds <= sec[1])) return sm_ctx_fail_range(c, name, b->seconds, sec[0], sec[1]);
    *tokens = sm_tokens_of(c->m, b->seconds);
    for (size_t k = 0; b->keep && k < (size_t)*tokens * SM_VQ_LAYERS; k++)
        if (b->keep[k] < (k % SM_VQ_LAYERS ? -1 : 0) || b->keep[k] >= SM_VQ_N)
            return block_fail(c, i, "a kept code is out of range");
    if (b->keep) return SM_OK;
    if (!c->m->gen.layers) return sm_ctx_fail(c, SM_E_UNSUPPORTED, "this model file has no generator");
    if (b->text) return sm_check_text(b->text, -1, &len, c->err, sizeof c->err);
    if (b->enc && b->enc->m == c->m && b->enc->ready) return SM_OK;
    return block_fail(c, i, "no text, or enc not encoded or from another model");
}

/* every block valid, the timeline within one decode and each made block within one run; *L: the longest run */
static sm_status check_blocks(sm_ctx *c, const sm_request *r, const block *b, uint32_t n, const sm_motion *out,
                              uint32_t *tok, uint32_t *L) {
    if (c->active) return sm_ctx_fail(c, SM_E_STATE, SM_ACTIVE);
    if (!r || !out || r->m != c->m || out->m != c->m) return sm_ctx_fail(c, SM_E_ARG, "r or out NULL or from another model");
    if (atomic_load(&c->cancel)) return sm_ctx_fail(c, SM_E_CANCELLED, SM_CANCELLED_MSG);
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; i++) {
        const sm_status s = check_block(c, &b[i], i, &tok[i]);
        if (s != SM_OK) return s;
        total += tok[i];
    }
    if (total > SM_MAX_CODES) return sm_ctx_fail_range(c, "timeline tokens", total, 1, SM_MAX_CODES);
    const sm_status s = sm_path_check(c->m, &r->path, total, c->err, sizeof c->err);
    if (s != SM_OK) return s;
    const uint32_t cap = c->m->gen.max_tokens < SM_MAX_RUN ? c->m->gen.max_tokens : SM_MAX_RUN;
    for (uint32_t i = *L = 0; i < n; i++) {
        const edges e = edges_of(b, n, r->v[SM_P_LOOP] != 0, i);
        const uint32_t run = e.left + tok[i] + e.right + e.ahead + e.tie;
        if (!b[i].keep && run > cap) {
            snprintf(c->err, sizeof c->err, "block %u: %u tokens with seams, over %u", i, run, cap);
            return SM_E_ARG;
        }
        if (!b[i].keep && run > *L) *L = run;
    }
    return SM_OK;
}

typedef struct { const double *v; int16_t *codes; float *conf; const uint32_t *first; uint32_t done, total; } timeline;

/* the run's seam tokens pinned to the final neighbours' codes on all layers, the rest free */
static void pin_seams(sm_ctx *c, const timeline *t, uint32_t n, uint32_t L, edges e) {
    const int16_t *left = t->codes + (e.left ? t->first[e.l + 1] - e.left : 0) * SM_VQ_LAYERS;
    const int16_t *right = t->codes + (e.right ? t->first[e.q] : 0) * SM_VQ_LAYERS;
    for (uint32_t j = 0; j < L; j++) {
        const uint32_t k = j - e.left - n;
        const int16_t *src = j < e.left ? left + j * SM_VQ_LAYERS : k < e.right ? right + k * SM_VQ_LAYERS : NULL;
        if (src) memcpy(c->pins + j * SM_VQ_LAYERS, src, sizeof(int16_t) * SM_VQ_LAYERS);
        else c->pins[j * SM_VQ_LAYERS] = -1;
    }
}

/* block i's own tokens go into the timeline's codes */
static sm_status gen_block(sm_ctx *c, timeline *t, const block *b, uint32_t i, edges e) {
    const uint32_t n = t->first[i + 1] - t->first[i], L = e.left + n + e.right + e.ahead + e.tie;
    sm_status s = SM_OK;
    if (b->text) s = sm_ctx_encode(c, b->text, strlen(b->text), c->text);
    else sm_text_copy(c->text, b->enc);
    pin_seams(c, t, n, L, e);
    if (s == SM_OK) s = sm_ctx_start_run(c, t->v, L, e.tie, b->seed);
    while (s == SM_OK) {
        s = sm_ctx_next_pass(c);
        if (s >= 0 && sm_ctx_report(c, ++t->done, t->total) != SM_OK) s = SM_E_CANCELLED;
    }
    if (s != SM_DONE) return s;

    memcpy(t->codes + t->first[i] * SM_VQ_LAYERS, c->codes + e.left * SM_VQ_LAYERS, sizeof(int16_t) * n * SM_VQ_LAYERS);
    memcpy(t->conf + t->first[i], c->conf + e.left, sizeof(float) * n);
    return SM_OK;
}

/* kept codes go through the decode scratch first, as keep may point into out's own codes */
static sm_status place_kept(sm_ctx *c, const block *b, uint32_t n, const uint32_t *first, const uint32_t *tok,
                            sm_motion *out) {
    const size_t row = sizeof(int16_t) * SM_VQ_LAYERS;
    int16_t *kept = c->dec;
    for (uint32_t i = 0; i < n; i++)
        if (b[i].keep) memcpy(kept + first[i] * SM_VQ_LAYERS, b[i].keep, row * tok[i]);
    const sm_status s = sm_motion_fit(out, first[n]);
    if (s != SM_OK) return sm_ctx_fail(c, s, "out of memory");

    for (uint32_t i = 0; i < n; i++) {
        if (!b[i].keep) continue;
        memcpy(out->codes + first[i] * SM_VQ_LAYERS, kept + first[i] * SM_VQ_LAYERS, row * tok[i]);
        for (uint32_t j = first[i]; j < first[i + 1]; j++) out->conf[j] = 1;
    }
    out->tokens = out->frames = 0;
    return SM_OK;
}

static sm_status make_blocks(sm_ctx *c, const sm_request *r, const block *b, uint32_t n, const uint32_t *first,
                             sm_motion *out) {
    const int loop = r->v[SM_P_LOOP] != 0;
    timeline t = {r->v, out->codes, out->conf, first, 0, 0};
    for (uint32_t i = 0; i < n; i++)
        if (!b[i].keep) t.total += (uint32_t)(r->v[SM_P_STEPS] + r->v[SM_P_RES_LAYERS]);
    for (uint32_t i = 0; i < n; i++) {
        if (b[i].keep) continue;
        const sm_status s = gen_block(c, &t, &b[i], i, edges_of(b, n, loop, i));
        if (s != SM_OK) return s;
    }
    return SM_OK;
}

static sm_status generate_blocks(sm_ctx *c, const sm_request *r, uint32_t n, const char *const *texts,
                                 const sm_text *const *encs, const double *seconds, const int32_t *seeds,
                                 const int16_t *const *keep, uint32_t *first_frames, sm_motion *out) {
    if (!n || n > MAX_BLOCKS) return sm_ctx_fail_range(c, "n_blocks", n, 1, MAX_BLOCKS);
    if (!seconds || !seeds) return sm_ctx_fail(c, SM_E_ARG, "seconds or seeds NULL");
    block b[MAX_BLOCKS];
    uint32_t tok[MAX_BLOCKS], first[MAX_BLOCKS + 1], longest = 0;
    for (uint32_t i = 0; i < n; i++)
        b[i] = (block){texts ? texts[i] : NULL, encs ? encs[i] : NULL, seconds[i], seeds[i], keep ? keep[i] : NULL};
    sm_status s = check_blocks(c, r, b, n, out, tok, &longest);
    if (s != SM_OK) return s;

    first[0] = 0;
    for (uint32_t i = 0; i < n; i++) first[i + 1] = first[i] + tok[i];
    s = sm_ctx_fit(c, longest, first[n]);
    if (s == SM_OK) s = place_kept(c, b, n, first, tok, out);
    if (s == SM_OK) s = make_blocks(c, r, b, n, first, out);
    if (s == SM_OK) s = sm_ctx_decode(c, out->codes, first[n], NULL, r->v, &r->path, out);
    if (s != SM_OK) return s;

    for (uint32_t i = 0; first_frames && i <= n; i++) first_frames[i] = 2 * first[i];
    out->seed = -1;
    return SM_OK;
}

sm_status sm_ctx_generate_blocks(sm_ctx *c, const sm_request *r, uint32_t n_blocks, const char *const *texts,
                                 const sm_text *const *encs, const double *seconds, const int32_t *seeds,
                                 const int16_t *const *keep, uint32_t *first_frames, sm_motion *out) {
    if (!c) return SM_E_ARG;
    c->err[0] = 0;
    const sm_status s = generate_blocks(c, r, n_blocks, texts, encs, seconds, seeds, keep, first_frames, out);
    if (s == SM_OK) return s;
    if (!c->err[0]) sm_ctx_fail(c, s, sm_status_message(s));
    if (out && out->m == c->m) out->tokens = out->frames = 0;
    return s;
}
