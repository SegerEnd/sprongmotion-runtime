/* The text encoder and the token generator: a masked base model and a residual model on one post-norm
   trunk, both with classifier-free guidance, and the sampling loop (MoMask, Guo et al. 2024). */
#include "sm_internal.h"

enum { N_CODES = 512, MASK_ID = 512, RES_N = 513 };   /* RES_N: the codes and the pad code */

static int has(const sm_model *m, const char *name) {
    uint64_t d[2] = {0, 0};
    return sm_gguf_tensor(&m->g, name, SM_T_Q8_0, 2, d) || sm_gguf_tensor(&m->g, name, SM_T_F32, 2, d);
}

static const float *vec(const sm_model *m, const char *name, uint64_t n, char *err, size_t en) {
    uint64_t d[1] = {n};
    return sm_need(m, name, SM_T_F32, 1, d, err, en);
}

/* [n][k] rows, Q8_0 or F32 (k or n 0: any). kind 0: a table; 1: a GEMM weight (n % 16 == 0); 2: one with a bias, the
   names then being name.w and name.b */
static int lin(const sm_model *m, sm_lin *l, const char *name, int kind, uint64_t k, uint64_t n, char *err, size_t en) {
    char w[64], b[64];
    uint64_t d[2] = {k, n};
    snprintf(w, sizeof w, kind == 2 ? "%s.w" : "%s", name);
    snprintf(b, sizeof b, "%s.b", name);
    l->type = SM_T_Q8_0;
    if (!(l->w = sm_gguf_tensor(&m->g, w, SM_T_Q8_0, 2, d))) l->type = SM_T_F32;
    if (!l->w && !(l->w = sm_need(m, w, SM_T_F32, 2, d, err, en))) return 0;
    if (d[0] % 32 || d[0] > 0xFFFF || d[1] > 0xFFFF || (kind && d[1] % 16))
        return !sm_fail(err, en, SM_E_FORMAT, "bad matrix size");
    l->k = (uint32_t)d[0], l->n = (uint32_t)d[1], l->b = NULL;
    return kind < 2 || (l->b = vec(m, b, l->n, err, en));
}

static int norm(const sm_model *m, sm_norm *n, const char *name, uint32_t d, char *err, size_t en) {
    char w[64], b[64];
    snprintf(w, sizeof w, "%s.w", name), snprintf(b, sizeof b, "%s.b", name);
    return (n->w = vec(m, w, d, err, en)) && (n->b = vec(m, b, d, err, en));
}

/* layers by probing pre.L0, pre.L1, ...; heads = max(4, d / 64) */
static int stack(const sm_model *m, sm_layer *L, const char *pre, uint32_t d, uint32_t *n, char *err, size_t en) {
    for (*n = 0; *n < SM_MAX_LAYERS; ++*n) {
        char s[7][48];
        const char *part[7] = {"attn.qkv", "attn.out", "ff1", "ff2", "ln1", "ln2", "attn.qkv.w"};
        for (int i = 0; i < 7; i++) snprintf(s[i], sizeof s[i], "%s.L%u.%s", pre, *n, part[i]);
        if (!has(m, s[6])) break;
        sm_layer *l = &L[*n];
        if (!lin(m, &l->qkv, s[0], 2, d, 3 * d, err, en) || !lin(m, &l->out, s[1], 2, d, d, err, en) ||
            !lin(m, &l->ff1, s[2], 2, d, 0, err, en) || !lin(m, &l->ff2, s[3], 2, l->ff1.n, d, err, en) ||
            !norm(m, &l->ln1, s[4], d, err, en) || !norm(m, &l->ln2, s[5], d, err, en))
            return 0;
    }
    return *n > 0 && d % (d / 64 > 4 ? d / 64 : 4) == 0 ? 1 : !sm_fail(err, en, SM_E_FORMAT, "bad layer stack");
}

static int bind_text(const sm_model *m, sm_textenc *t, char *err, size_t en) {
    uint64_t dp[2] = {0, 0};
    if (!lin(m, &t->emb, "text.emb", 0, 0, 0, err, en) || !lin(m, &t->emb_up, "text.emb_up", 1, t->emb.k, 0, err, en))
        return 0;
    dp[0] = t->d = t->emb_up.n;
    if (!(t->pos = sm_need(m, "text.pos", SM_T_F32, 2, dp, err, en))) return 0;
    t->max_len = (uint32_t)(dp[1] < 0xFFFF ? dp[1] : 0);
    t->heads = t->d / 64 > 4 ? t->d / 64 : 4;
    if (!stack(m, t->L, "text", t->d, &t->layers, err, en) || !norm(m, &t->ln, "text.ln", t->d, err, en) ||
        !lin(m, &t->out, "text.out", 2, 2 * t->d, 0, err, en))
        return 0;
    return m->tok.max_len <= t->max_len && t->emb.n > m->tok.pad
               ? 1 : !sm_fail(err, en, SM_E_FORMAT, "text encoder smaller than the tokenizer");
}

static int bind_heads(const sm_model *m, sm_gen *g, uint32_t text_dim, char *err, size_t en) {
    const uint32_t d = g->d;
    uint64_t dq[2] = {SM_VQ_LAYERS, d};
    const struct { sm_lin *l; const char *name; int kind; uint32_t k, n; } lins[] = {
        {&g->mask.cond, "mask.cond", 2, text_dim, d}, {&g->mask.word, "mask.word", 2, g->word_dim, d},
        {&g->mask.dense, "mask.head.dense", 2, d, d}, {&g->mask.out, "mask.head.out", 2, d, N_CODES},
        {&g->tok, "mask.tok", 0, d, 0}, {&g->res.cond, "res.cond", 2, text_dim, d},
        {&g->res.dense, "res.head.dense", 2, d, d},
    };
    for (size_t i = 0; i < sizeof lins / sizeof lins[0]; i++)
        if (!lin(m, lins[i].l, lins[i].name, lins[i].kind, lins[i].k, lins[i].n, err, en)) return 0;
    if (!norm(m, &g->mask.norm, "mask.head.norm", d, err, en) || !norm(m, &g->res.norm, "res.head.norm", d, err, en) ||
        !(g->word_type = vec(m, "mask.word_type", d, err, en)) || !(g->in_b = vec(m, "res.in_b", d, err, en)) ||
        !(g->quant_b = vec(m, "res.quant.b", d, err, en)) ||
        !(g->quant_w = sm_need(m, "res.quant.w", SM_T_F32, 2, dq, err, en)))
        return 0;
    for (uint32_t q = 1; q < SM_VQ_LAYERS; q++) {
        char in[16], out[16];
        snprintf(in, sizeof in, "res.in%u", q), snprintf(out, sizeof out, "res.out%u", q);
        if (!lin(m, &g->in[q], in, 0, d, 0, err, en) || !lin(m, &g->res_out[q], out, 2, d, 0, err, en)) return 0;
        if (g->in[q].n < RES_N || g->res_out[q].n < RES_N || g->res_out[q].n != g->res_out[1].n)
            return !sm_fail(err, en, SM_E_FORMAT, "bad residual tables");
    }
    return g->tok.n > MASK_ID ? 1 : !sm_fail(err, en, SM_E_FORMAT, "bad mask.tok");
}

sm_status sm_bind_gen(sm_model *m, char *err, size_t en) {
    if (!has(m, "text.emb")) return SM_OK;
    sm_textenc *t = &m->text;
    sm_gen *g = &m->gen;
    uint64_t dp[2] = {0, 0};
    if (!bind_text(m, t, err, en)) return SM_E_FORMAT;
    if (!(g->pe = sm_need(m, "pe", SM_T_F32, 2, dp, err, en))) return SM_E_FORMAT;
    if (dp[0] % 32 || dp[0] > 0xFFFF || !dp[1] || dp[1] > 0xFFFF) return sm_fail(err, en, SM_E_FORMAT, "bad pe");
    g->d = (uint32_t)dp[0], g->max_tokens = (uint32_t)dp[1];
    g->heads = g->d / 64 > 4 ? g->d / 64 : 4;
    g->word_dim = t->d;
    if (!stack(m, g->mask.L, "trunk", g->d, &g->layers, err, en) || !bind_heads(m, g, t->out.n, err, en))
        return SM_E_FORMAT;
    t->par = g->par = &m->par;
    sm_lin *all[2 * 4 * SM_MAX_LAYERS + 16];
    size_t n = 0, floats = 0;
    for (uint32_t l = 0; l < t->layers; l++)
        all[n++] = &t->L[l].qkv, all[n++] = &t->L[l].out, all[n++] = &t->L[l].ff1, all[n++] = &t->L[l].ff2;
    for (uint32_t l = 0; l < g->layers; l++)
        all[n++] = &g->mask.L[l].qkv, all[n++] = &g->mask.L[l].out, all[n++] = &g->mask.L[l].ff1, all[n++] = &g->mask.L[l].ff2;
    sm_lin *heads[] = {&t->emb_up, &t->out, &g->mask.cond, &g->mask.word, &g->mask.dense, &g->mask.out, &g->res.cond,
                       &g->res.dense};
    for (size_t i = 0; i < sizeof heads / sizeof heads[0]; i++) all[n++] = heads[i];
    for (uint32_t q = 1; q < SM_VQ_LAYERS; q++) all[n++] = &g->res_out[q];
    for (size_t i = 0; i < n; i++) floats += (size_t)all[i]->k * all[i]->n;
    if (!(m->packed = sm_alloc(&m->h, sizeof(float) * floats))) return sm_fail(err, en, SM_E_NOMEM, "out of memory");
    for (size_t i = 0, at = 0; i < n; at += (size_t)all[i]->k * all[i]->n, i++) sm_pack(all[i], m->packed + at);
    memcpy(g->res.L, g->mask.L, sizeof g->res.L);
    return SM_OK;
}

/* text encoder: pre-norm layers, final LN, vec = out(concat(x[W-1], mean(x))) */
size_t sm_text_scratch_floats(const sm_textenc *t) {
    const size_t M = t->max_len;
    return M * t->emb.k + M * t->d + 2 * (size_t)t->d +
           sm_layer_scratch_floats(t->max_len, t->d, t->L[0].ff1.n, t->heads);
}

void sm_text_encode(const sm_textenc *t, const int32_t *ids, uint32_t W, float *vec, float *words, float *scratch) {
    const uint32_t d = t->d, M = t->max_len;
    float *e = scratch, *x = e + (size_t)M * t->emb.k, *cat = x + (size_t)M * d, *s = cat + 2 * d;
    for (uint32_t i = 0; i < W; i++) sm_row(&t->emb, (uint32_t)ids[i], e + (size_t)i * t->emb.k);
    sm_linear(t->par, &t->emb_up, e, W, x, SM_EPI_ADD, t->pos);
    const sm_seg seg = {0, W};
    const sm_batch b = {&seg, 1, W};
    for (uint32_t l = 0; l < t->layers; l++) sm_layer_pre(t->par, &t->L[l], &b, t->heads, x, s);
    sm_layernorm(t->par, x, W, d, t->ln, 1e-5, x);
    memcpy(cat, x + (size_t)(W - 1) * d, sizeof(float) * d);
    for (uint32_t i = 0; i < d; i++) {
        float sum = 0;
        for (uint32_t j = 0; j < W; j++) sum += x[(size_t)j * d + i];
        cat[d + i] = sum / (float)W;
    }
    sm_linear(t->par, &t->out, cat, 1, vec, SM_EPI_NONE, NULL);
    for (size_t i = 0; i < (size_t)W * d; i++) words[i] = sm_round_half(x[i]);   /* the generator expects fp16-rounded words */
}

typedef struct {
    float *words, *cm, *cr, *seq, *layer, *h, *h2, *raw, *lg, *hist, *row, *score, *prob;
    int32_t *ids, *pick;
    uint8_t *mask;
} work;

/* one layout for the size and the pointers; s NULL: only the size */
static size_t carve(const sm_gen *g, uint32_t W, uint32_t L, float *s, work *w) {
    const size_t d = g->d, n = g->res_out[1].n, M = 2 * (size_t)L + (W + 2 > 4 ? W + 2 : 4);
    const size_t len[] = {W * d, d, d, M * d, sm_layer_scratch_floats((uint32_t)M, g->d, g->mask.L[0].ff1.n, g->heads),
                          2 * L * d, 2 * L * d, 2 * L * n, L * n, L * d, d, L, L, L, L, (L + 3) / 4};
    float *p[sizeof len / sizeof len[0]];
    size_t at = 0;
    for (size_t i = 0; i < sizeof len / sizeof len[0]; at += len[i++]) p[i] = s ? s + at : NULL;
    *w = (work){p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12],
                (int32_t *)p[13], (int32_t *)p[14], (uint8_t *)p[15]};
    return at;
}

size_t sm_gen_scratch_floats(const sm_gen *g, uint32_t W, uint32_t L) { return carve(g, W, L, NULL, &(work){0}); }

void sm_gen_prepare(const sm_gen *g, const float *vec, const float *words, uint32_t W, uint32_t L, float *scratch) {
    work w;
    carve(g, W, L, scratch, &w);
    sm_linear(g->par, &g->mask.cond, vec, 1, w.cm, SM_EPI_NONE, NULL);
    sm_linear(g->par, &g->res.cond, vec, 1, w.cr, SM_EPI_NONE, NULL);
    sm_linear(g->par, &g->mask.word, words, W, w.words, SM_EPI_NONE, NULL);
    for (uint32_t j = 0; j < W; j++)
        for (uint32_t i = 0; i < g->d; i++) w.words[(size_t)j * g->d + i] += g->word_type[i];
}

/* the trunk on both sequences, then dense, GELU, LN (eps 1e-12) and out on the L motion rows of each: raw [2L][n] */
static void trunk_head(const sm_gen *g, const sm_head *hd, const sm_lin *out, const work *w, const sm_seg seg[2],
                       uint32_t L) {
    const size_t d = g->d;
    const sm_batch b = {seg, 2, seg[1].start + seg[1].len};
    for (uint32_t l = 0; l < g->layers; l++) sm_layer_post(g->par, &hd->L[l], &b, g->heads, w->seq, w->layer);
    for (size_t s = 0; s < 2; s++)
        memcpy(w->h + s * L * d, w->seq + (seg[s].start + seg[s].len - L) * d, sizeof(float) * L * d);
    sm_linear(g->par, &hd->dense, w->h, 2 * L, w->h2, SM_EPI_GELU, NULL);
    sm_layernorm(g->par, w->h2, 2 * L, g->d, hd->norm, 1e-12, w->h2);
    sm_linear(g->par, out, w->h2, 2 * L, w->raw, SM_EPI_NONE, NULL);
}

/* [cond, words, T_mask[ids] + pe] and [cond_emb(0), T_mask[ids] + pe]: the unconditional pass has no words */
const float *sm_gen_base(const sm_gen *g, uint32_t W, uint32_t L, const int32_t *ids, float *scratch) {
    work w;
    carve(g, W, L, scratch, &w);
    const size_t d = g->d, c0 = 1 + W, u0 = 1 + W + L;
    memcpy(w.seq, w.cm, sizeof(float) * d);
    memcpy(w.seq + d, w.words, sizeof(float) * W * d);
    memcpy(w.seq + u0 * d, g->mask.cond.b, sizeof(float) * d);
    for (uint32_t j = 0; j < L; j++) {
        float *x = w.seq + (c0 + j) * d;
        sm_row(&g->tok, (uint32_t)ids[j], x);
        for (size_t i = 0; i < d; i++) x[i] += g->pe[j * d + i];
        memcpy(w.seq + (u0 + 1 + j) * d, x, sizeof(float) * d);
    }
    const sm_seg seg[2] = {{0, 1 + W + L}, {(uint32_t)u0, 1 + L}};
    trunk_head(g, &g->mask, &g->mask.out, &w, seg, L);
    return w.raw;
}

/* [cond, quant_emb(onehot(q)), hist + b_in + pe], conditional and with cond_emb(0) */
const float *sm_gen_res(const sm_gen *g, uint32_t W, uint32_t L, uint32_t q, const float *hist, float *scratch) {
    work w;
    carve(g, W, L, scratch, &w);
    const size_t d = g->d, u0 = 2 + L;
    memcpy(w.seq, w.cr, sizeof(float) * d);
    memcpy(w.seq + u0 * d, g->res.cond.b, sizeof(float) * d);
    for (size_t i = 0; i < d; i++)
        w.seq[d + i] = w.seq[(u0 + 1) * d + i] = g->quant_w[i * SM_VQ_LAYERS + q] + g->quant_b[i];
    for (size_t j = 0; j < L; j++) {
        float *x = w.seq + (2 + j) * d;
        for (size_t i = 0; i < d; i++) x[i] = hist[j * d + i] + g->in_b[i] + g->pe[j * d + i];
        memcpy(w.seq + (u0 + 2 + j) * d, x, sizeof(float) * d);
    }
    const sm_seg seg[2] = {{0, 2 + L}, {(uint32_t)u0, 2 + L}};
    trunk_head(g, &g->res, &g->res_out[q], &w, seg, L);
    return w.raw;
}

/* lg [L][used] = u + (c - u) * cfg, from raw [2L][n] */
static sm_status guide(const float *raw, uint32_t L, uint32_t n, uint32_t used, float cfg, float *lg) {
    for (size_t j = 0; j < L; j++)
        for (size_t c = 0; c < used; c++) {
            const float cv = raw[j * n + c], uv = raw[(L + j) * n + c], v = uv + (cv - uv) * cfg;
            if (!isfinite(v)) return SM_E_NUMERIC;
            lg[j * used + c] = v;
        }
    return SM_OK;
}

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/* h = splitmix64(seed ^ splitmix64(stream << 48 | step << 32 | pos << 12 | code)), u = ((h >> 11) + 0.5) 2^-53,
   g = -log(-log(u)); stream 0: base steps, 1: residual layers */
float sm_hash_gumbel(uint64_t seed, uint32_t stream, uint32_t step, uint32_t pos, uint32_t code) {
    const uint64_t key = (uint64_t)stream << 48 | (uint64_t)step << 32 | (uint64_t)pos << 12 | code;
    const double u = ((double)(splitmix64(seed ^ splitmix64(key)) >> 11) + 0.5) * 0x1p-53;
    return (float)-sm_log(-sm_log(u));
}

typedef struct { uint64_t seed; uint32_t stream; } noise_key;
static float code_noise(void *k, uint32_t step, uint32_t pos, uint32_t code) {
    return sm_hash_gumbel(((noise_key *)k)->seed, ((noise_key *)k)->stream, step, pos, code);
}

static float kth_largest(float *v, uint32_t n, uint32_t k) {  /* quickselect with a 3-way split: [> p | == p | < p] */
    uint32_t lo = 0, hi = n;
    const uint32_t want = k - 1;
    for (;;) {
        const float p = v[lo + (hi - lo) / 2];
        uint32_t a = lo, i = lo, b = hi;
        while (i < b) {
            const float x = v[i];
            if (x > p) { v[i++] = v[a]; v[a++] = x; }
            else if (x < p) { v[i] = v[--b]; v[b] = x; }
            else i++;
        }
        if (want < a) hi = a;
        else if (want < b) return p;
        else lo = b;
    }
}

typedef struct {
    const float *lg; uint32_t L, N; const uint8_t *rows; float temp; uint32_t k; sm_noise_fn noise; void *user;
    uint32_t step; int32_t *pick; float *prob;
} pick_job;

static void pick_row(const pick_job *p, uint32_t j, float *tmp) {
    const uint32_t N = p->N, k = p->k;
    const float *x = p->lg + (size_t)j * N;
    memcpy(tmp, x, sizeof(float) * N);
    const float kth = k && k < N ? kth_largest(tmp, N, k) : -INFINITY;
    int32_t best = -1; float best_v = 0;
    for (uint32_t c = 0; c < N; c++) {
        if (x[c] < kth) continue;   /* outside the top k (ties are kept) */
        const float v = x[c] / p->temp + p->noise(p->user, p->step, j, c);
        if (best < 0 || v > best_v) { best = (int32_t)c; best_v = v; }
    }
    p->pick[j] = best;
    if (p->prob) {
        float mx = x[0]; double sum = 0;
        for (uint32_t c = 1; c < N; c++) mx = x[c] > mx ? x[c] : mx;
        for (uint32_t c = 0; c < N; c++) sum += sm_exp(x[c] - mx);
        p->prob[j] = (float)(sm_exp(x[best] - mx) / sum);
    }
}

void sm_pick(const float *lg, uint32_t L, uint32_t N, const uint8_t *rows, float temp, uint32_t k,
             sm_noise_fn noise, void *user, uint32_t step, int32_t *pick, float *prob, float *tmp) {
    const pick_job p = {lg, L, N, rows, temp, k, noise, user, step, pick, prob};
    for (uint32_t j = 0; j < L; j++) if (!rows || rows[j]) pick_row(&p, j, tmp);
}

#define PICK_ROWS 4   /* rows per task */

static void pick_rows(void *arg, uint32_t task) {
    const pick_job *p = arg;
    float tmp[RES_N];
    for (uint32_t j = task * PICK_ROWS; j < p->L && j < (task + 1) * PICK_ROWS; j++)
        if (!p->rows || p->rows[j]) pick_row(p, j, tmp);
}

static void tie(const sm_run *r, int32_t *v, uint32_t layer) {   /* the wrap's tail copies the head */
    for (uint32_t k = 0, t = r->L - r->tie; k < r->tie; k++, t++) {
        v[t] = v[k];
        r->codes[t * SM_VQ_LAYERS + layer] = r->codes[k * SM_VQ_LAYERS + layer];
        if (r->conf) r->conf[t] = r->conf[k];
    }
}

/* code layer 0: each step re-masks the least confident free positions (cosine schedule) and samples them again */
static sm_status base_step(const sm_gen *g, const work *w, sm_run *r, float *scratch) {
    const uint32_t L = r->L, i = r->pass;
    noise_key k = {r->seed, 0};
    sm_lowest(w->score, L, sm_nmask(i, r->s.steps, L - r->fixed), w->mask);
    for (uint32_t j = 0; j < L; j++) if (w->mask[j]) w->ids[j] = MASK_ID;
    for (uint32_t j = 0; j < r->tie; j++) w->ids[L - r->tie + j] = w->ids[j];
    const sm_status st = guide(sm_gen_base(g, r->W, L, w->ids, scratch), L, N_CODES, N_CODES, r->s.cfg, w->lg);
    if (st != SM_OK) return st;
    const pick_job pj = {w->lg, L, N_CODES, w->mask, r->s.temp, r->s.top_k, code_noise, &k, i, w->pick, w->prob};
    sm_par_for(g->par, (L + PICK_ROWS - 1) / PICK_ROWS, pick_rows, (void *)&pj);
    for (uint32_t j = 0; j < L; j++) {
        w->score[j] = w->mask[j] ? w->prob[j] : 1e5f;
        if (w->mask[j]) w->ids[j] = w->pick[j];
        if (w->mask[j] && r->conf) r->conf[j] = w->prob[j];
        r->codes[j * SM_VQ_LAYERS] = (int16_t)w->ids[j];
    }
    tie(r, w->ids, 0);
    return SM_OK;
}

/* layer q from the summed embeddings of layers 0..q-1; the pad code (512) feeds the history and becomes -1 */
static sm_status res_step(const sm_gen *g, const work *w, sm_run *r, uint32_t q, float *scratch) {
    const size_t d = g->d, n = g->res_out[1].n, L = r->L;
    const int32_t *prev = q == 1 ? w->ids : w->pick;
    noise_key k = {r->seed, 1};
    if (q == 1) memset(w->hist, 0, sizeof(float) * L * d);
    for (size_t j = 0; j < L; j++) {
        sm_row(&g->in[q], (uint32_t)prev[j], w->row);
        for (size_t i = 0; i < d; i++) w->hist[j * d + i] += w->row[i];
    }
    const sm_status st = guide(sm_gen_res(g, r->W, r->L, q, w->hist, scratch), r->L, (uint32_t)n, RES_N, r->s.res_cfg,
                               w->lg);
    if (st != SM_OK) return st;
    const pick_job pj = {w->lg, r->L, RES_N, NULL, r->s.temp, r->s.top_k, code_noise, &k, q - 1, w->pick, NULL};
    sm_par_for(g->par, (r->L + PICK_ROWS - 1) / PICK_ROWS, pick_rows, (void *)&pj);
    for (size_t j = 0; j < L; j++) {
        const int16_t *p = r->pins ? r->pins + j * SM_VQ_LAYERS : NULL;
        if (p && p[0] >= 0 && p[q] != SM_FREE) w->pick[j] = p[q] < 0 ? N_CODES : p[q];
        r->codes[j * SM_VQ_LAYERS + q] = (int16_t)(w->pick[j] >= N_CODES ? -1 : w->pick[j]);
    }
    tie(r, w->pick, q);
    return SM_OK;
}

sm_status sm_gen_start(const sm_gen *g, sm_run *r, const float *vec, const float *words, float *scratch) {
    const uint32_t L = r->L;
    if (!L || L > g->max_tokens || !r->s.steps || r->s.res_layers >= SM_VQ_LAYERS || !(r->s.temp > 0) || r->tie > L)
        return SM_E_ARG;
    for (uint32_t j = 0; r->pins && j < L; j++) if (r->pins[j * SM_VQ_LAYERS] >= N_CODES) return SM_E_ARG;
    work w;
    carve(g, r->W, L, scratch, &w);
    sm_gen_prepare(g, vec, words, r->W, L, scratch);
    r->pass = r->fixed = 0;
    for (uint32_t j = 0; j < L; j++) {
        const int pin = r->pins && r->pins[j * SM_VQ_LAYERS] >= 0, tail = j >= L - r->tie;
        w.ids[j] = pin ? r->pins[j * SM_VQ_LAYERS] : MASK_ID;
        w.score[j] = pin || tail ? 1e5f : 0;
        if (r->conf) r->conf[j] = 1;
        r->fixed += (uint32_t)(pin || tail);
    }
    for (size_t j = 0; j < (size_t)L * SM_VQ_LAYERS; j++) r->codes[j] = -1;
    return SM_OK;
}

sm_status sm_gen_pass(const sm_gen *g, sm_run *r, float *scratch) {
    const uint32_t total = r->s.steps + r->s.res_layers;
    work w;
    carve(g, r->W, r->L, scratch, &w);
    if (r->pass >= total) return SM_E_STATE;
    const sm_status st = r->pass < r->s.steps ? base_step(g, &w, r, scratch)
                                              : res_step(g, &w, r, r->pass - r->s.steps + 1, scratch);
    return st != SM_OK ? st : ++r->pass == total ? SM_DONE : SM_OK;
}

#define PI 3.14159265358979323846

/* MoMask's cosine schedule (Guo et al. 2024, arXiv 2312.00063) */
uint32_t sm_nmask(uint32_t i, uint32_t n, uint32_t free) {
    const double t = n > 1 ? (double)i / (double)(n - 1) : 0.0;   /* np.linspace(0, 1, n)[i] */
    const double v = rint(cos(t * PI / 2) * (double)free);         /* rint: half to even, as Python round() */
    return v < 1 ? 1 : (uint32_t)v;
}

void sm_lowest(const float *score, uint32_t L, uint32_t nmask, uint8_t *mask) {
    memset(mask, 0, L);
    for (uint32_t c = 0; c < nmask && c < L; c++) {
        uint32_t best = L;
        for (uint32_t j = 0; j < L; j++) if (!mask[j] && (best == L || score[j] < score[best])) best = j;
        mask[best] = 1;
    }
}

static double pow2i(int64_t n) {   /* 2^n for -1022 <= n <= 1023 */
    const uint64_t b = (uint64_t)(n + 1023) << 52;
    double d;
    memcpy(&d, &b, 8);
    return d;
}

/* exp and log from + - * / only, so every target gives the same bits.
   exp(x) = 2^n exp(r), |r| <= ln2/2; Taylor to r^13 (error < 1e-17), Horner on 1/k! */
double sm_exp(double x) {
    if (x != x) return x;
    if (x < -745) return 0;
    if (x > 709) return INFINITY;
    const double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10;
    const double k = x * 1.4426950408889634 + (x < 0 ? -0.5 : 0.5);
    const int64_t n = (int64_t)k;
    const double r = x - (double)n * ln2_hi - (double)n * ln2_lo;
    static const double inv_fact[14] = {1, 1, 1 / 2.0, 1 / 6.0, 1 / 24.0, 1 / 120.0, 1 / 720.0, 1 / 5040.0, 1 / 40320.0,
                                        1 / 362880.0, 1 / 3628800.0, 1 / 39916800.0, 1 / 479001600.0, 1 / 6227020800.0};
    double p = inv_fact[13];
    for (int i = 12; i >= 0; i--) p = inv_fact[i] + r * p;
    return n < -1022 ? p * pow2i(n + 54) * 0x1p-54 : p * pow2i(n);
}

/* x > 0: x = m 2^e, m in [sqrt(1/2), sqrt(2)); log m = 2 atanh(s), s = (m - 1) / (m + 1), |s| < 0.172 */
double sm_log(double x) {
    if (!(x > 0)) return x == 0 ? -INFINITY : NAN;
    if (x == INFINITY) return x;
    uint64_t b;
    memcpy(&b, &x, 8);
    int64_t e = (int64_t)(b >> 52) - 1023;
    if (e == -1023) {   /* subnormal */
        x *= 0x1p54;
        memcpy(&b, &x, 8);
        e = (int64_t)(b >> 52) - 1023 - 54;
    }
    b = (b & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull;
    double m;
    memcpy(&m, &b, 8);
    if (m > 1.4142135623730951) m *= 0.5, e++;
    const double s = (m - 1) / (m + 1), s2 = s * s;
    double p = 0;
    for (int k = 23; k >= 1; k -= 2) p = 1.0 / k + s2 * p;
    return 2 * s * p + (double)e * 6.93147180559945309417e-01;
}

float sm_round_half(float x) {   /* to the nearest float16 and back, ties to even (torch .half().float()) */
    uint32_t b;
    memcpy(&b, &x, 4);
    const uint32_t s = b & 0x80000000u, a = b & 0x7FFFFFFFu;
    if (a >= 0x7F800000u) return x;
    if (a >= 0x477FF000u) b = s | 0x7F800000u;   /* rounds to >= 65536: inf */
    else if (a >= 0x38800000u) b = s | ((a + 0x0FFFu + ((a >> 13) & 1)) & ~0x1FFFu);   /* normal: 10 mantissa bits */
    else {   /* subnormal half: multiples of 2^-24; adding 2^23 rounds to an integer, half to even */
        const float v = ((s ? -x : x) * 0x1p24f + 0x1p23f - 0x1p23f) * 0x1p-24f;
        return s ? -v : v;
    }
    memcpy(&x, &b, 4);
    return x;
}
