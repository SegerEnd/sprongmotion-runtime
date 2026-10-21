/* A strict GGUF v3 reader (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and the model
   loader, which binds every tensor by name, type and shape. The input is untrusted: every read goes through take(),
   which stops at the end of the buffer. Beyond the spec: little endian, tensor types F32 / I32 / Q8_0, unique tensor
   names, <= 4096 tensors and entries. */
#define _POSIX_C_SOURCE 200809L   /* mmap, open */
#include "sm_internal.h"
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

enum { GGUF_U32 = 4, GGUF_I32 = 5, GGUF_F32 = 6, GGUF_BOOL = 7, GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_N_TYPES = 13 };
#define MAX_ENTRIES 4096
#define MAX_STRING  65535
#define MAX_ARRAY   (1u << 24)
#define MAX_DIM     (1u << 30)
#define MAX_DEPTH   4
#define BAD(msg) return sm_fail(err, en, SM_E_FORMAT, msg)

typedef struct { const uint8_t *p, *end; int ok; } rd;

static const uint8_t *take(rd *r, uint64_t n) {
    r->ok = r->ok && n <= (uint64_t)(r->end - r->p);
    if (!r->ok) return NULL;
    r->p += n;
    return r->p - n;
}

static uint64_t rd_u(rd *r, int bytes) {
    const uint8_t *q = take(r, (uint64_t)bytes);
    uint64_t v = 0;
    for (int i = bytes - 1; q && i >= 0; i--) v = v << 8 | q[i];
    return v;
}

static const char *rd_str(rd *r, uint32_t *len) {
    const uint64_t n = rd_u(r, 8);
    r->ok &= n <= MAX_STRING;
    *len = (uint32_t)n;
    return (const char *)take(r, n);
}

static const char *rd_key(rd *r, uint32_t *len, uint32_t *type) {
    const char *k = rd_str(r, len);
    *type = (uint32_t)rd_u(r, 4);
    return k;
}

static int scalar_size(uint32_t t) {  /* 0 for string / array, -1 for an unknown type */
    static const int8_t size[GGUF_N_TYPES] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    return t < GGUF_N_TYPES ? size[t] : -1;
}

static void skip_value(rd *r, uint32_t type, int depth) {
    uint32_t len;
    if (type == GGUF_STRING) {
        rd_str(r, &len);
    } else if (type == GGUF_ARRAY) {
        const uint32_t et = (uint32_t)rd_u(r, 4);
        const uint64_t n = rd_u(r, 8);
        r->ok &= depth < MAX_DEPTH && scalar_size(et) >= 0 && n <= MAX_ARRAY;
        if (et == GGUF_STRING || et == GGUF_ARRAY || et == GGUF_BOOL)
            for (uint64_t i = 0; i < n && r->ok; i++) skip_value(r, et, depth + 1);
        else take(r, n * (uint64_t)scalar_size(et));
    } else {
        r->ok &= scalar_size(type) > 0;
        const uint8_t *v = take(r, (uint64_t)scalar_size(type));
        r->ok &= !(v && type == GGUF_BOOL && *v > 1);
    }
}

static int lower_alnum(char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'); }

static int valid_key(const char *k, uint32_t n) {  /* lower_snake_case segments joined by '.' */
    if (!n || k[0] == '.' || k[n - 1] == '.') return 0;
    for (uint32_t i = 0; i < n; i++)
        if (!lower_alnum(k[i]) && k[i] != '_' && !(k[i] == '.' && k[i - 1] != '.')) return 0;
    return 1;
}

static int is_key(const char *k, uint32_t len, const char *want) {
    return len == strlen(want) && !memcmp(k, want, len);
}

int32_t sm_fail(char *err, size_t en, int32_t status, const char *msg) {
    if (err && en) snprintf(err, en, "%s", msg);
    return status;
}

int32_t sm_fail_range(char *err, size_t en, int32_t status, const char *name, double v, double lo, double hi) {
    sm_buf b = {err, en ? en - 1 : 0, 0};
    sm_puts(&b, name), sm_puts(&b, " = ");
    isfinite(v) ? sm_putn(&b, v) : sm_puts(&b, v != v ? "nan" : v > 0 ? "inf" : "-inf");
    sm_puts(&b, lo <= v && v <= hi ? " is not a whole number in " : " is out of range ");
    sm_putn(&b, lo), sm_puts(&b, " .. "), sm_putn(&b, hi);
    if (err && en) err[b.n < b.cap ? b.n : b.cap] = 0;
    return status;
}

static uint64_t sm_gguf_count_tensors(const void *buf, size_t size) {
    rd r = {buf, (const uint8_t *)buf + size, 1};
    const uint8_t *magic = take(&r, 4);
    const uint64_t version = rd_u(&r, 4), n = rd_u(&r, 8);
    return r.ok && !memcmp(magic, "GGUF", 4) && version == 3 && n <= MAX_ENTRIES ? n : 0;
}

/* the key-value section; general.alignment and general.architecture are checked here */
static int32_t read_metadata(rd *r, uint64_t n_kv, uint32_t *align, char *err, size_t en) {
    uint32_t len, type;
    for (uint64_t i = 0; i < n_kv && r->ok; i++) {
        const char *key = rd_key(r, &len, &type);
        if (r->ok && !valid_key(key, len)) BAD("bad metadata key");
        if (r->ok && type == GGUF_U32 && is_key(key, len, "general.alignment")) {
            *align = (uint32_t)rd_u(r, 4);
            if (!*align || *align % 8 || *align > 4096) BAD("bad alignment");
        } else if (r->ok && type == GGUF_STRING && is_key(key, len, "general.architecture")) {
            const char *arch = rd_str(r, &len);
            r->ok &= len > 0;
            for (uint32_t c = 0; arch && c < len; c++) r->ok &= lower_alnum(arch[c]);
        } else skip_value(r, type, 0);
    }
    return r->ok ? SM_OK : sm_fail(err, en, SM_E_FORMAT, "malformed metadata");
}

static int32_t read_tensor_info(rd *r, sm_tensor *x, uint32_t align, char *err, size_t en) {
    uint64_t elems = 1;
    x->name = rd_str(r, &x->name_len);
    x->nd = (uint32_t)rd_u(r, 4);
    if (!r->ok || !x->name_len || x->name_len > 64 || x->nd < 1 || x->nd > 4) BAD("bad tensor header");
    for (uint32_t d = 0; d < x->nd; d++) {
        x->dim[d] = rd_u(r, 8);
        if (!x->dim[d] || x->dim[d] > MAX_DIM || elems > (UINT64_MAX >> 31) / x->dim[d]) BAD("bad tensor shape");
        elems *= x->dim[d];
    }
    x->type = (uint32_t)rd_u(r, 4), x->offset = rd_u(r, 8);
    if (x->type == SM_T_Q8_0 && x->dim[0] % 32) BAD("bad Q8_0 row length");
    if (x->type != SM_T_F32 && x->type != SM_T_I32 && x->type != SM_T_Q8_0) BAD("bad tensor type");
    x->bytes = (size_t)(x->type == SM_T_Q8_0 ? elems / 32 * SM_Q8_BLOCK : elems * 4);
    if (!r->ok || x->offset % align) BAD("bad tensor offset");
    return SM_OK;
}

static int32_t sm_gguf_open(sm_gguf *g, const void *buf, size_t size, sm_tensor *t, uint64_t max_t,
                            char *err, size_t en) {
    rd r = {buf, (const uint8_t *)buf + size, 1};
    memset(g, 0, sizeof *g);
    g->base = buf, g->size = size;
    const uint8_t *magic = take(&r, 4);
    if (!magic || memcmp(magic, "GGUF", 4)) BAD("not a GGUF file");
    if (rd_u(&r, 4) != 3) return sm_fail(err, en, SM_E_VERSION, "GGUF version is not 3");
    g->n_t = rd_u(&r, 8), g->n_kv = rd_u(&r, 8);
    if (!r.ok || g->n_t > max_t || g->n_kv > MAX_ENTRIES) BAD("bad tensor count");
    g->kv = r.p;
    uint32_t align = 32;
    int32_t s = read_metadata(&r, g->n_kv, &align, err, en);
    g->t = t;
    for (uint64_t i = 0; i < g->n_t && s == SM_OK; i++) {
        s = read_tensor_info(&r, &t[i], align, err, en);
        for (uint64_t j = 0; j < i && s == SM_OK; j++)
            if (t[j].name_len == t[i].name_len && !memcmp(t[j].name, t[i].name, t[i].name_len)) BAD("duplicate name");
    }
    if (s != SM_OK) return s;
    const size_t start = ((size_t)(r.p - g->base) + align - 1) / align * align;
    if (start > size) BAD("truncated file");
    for (const uint8_t *p = r.p; p < g->base + start; p++)
        if (*p) BAD("header padding is not zero");
    g->data = g->base + start, g->data_size = size - start;
    for (uint64_t i = 0; i < g->n_t; i++) {
        if (t[i].offset > g->data_size || t[i].bytes > g->data_size - t[i].offset) BAD("tensor data outside the file");
        t[i].data = g->data + t[i].offset;
    }
    return SM_OK;
}

const void *sm_gguf_tensor(const sm_gguf *g, const char *name, uint32_t type, uint32_t nd, uint64_t *dims) {
    for (uint64_t i = 0; i < g->n_t; i++) {
        const sm_tensor *x = &g->t[i];
        if (!is_key(x->name, x->name_len, name)) continue;
        if (x->type != type || x->nd != nd) return NULL;
        for (uint32_t d = 0; d < nd; d++)
            if (dims[d] && x->dim[d] != dims[d]) return NULL;
        for (uint32_t d = 0; d < nd; d++) dims[d] = x->dim[d];
        return x->data;
    }
    return NULL;
}

int sm_gguf_num(const sm_gguf *g, const char *key, double *out) {
    rd r = {g->kv, g->base + g->size, 1};
    uint32_t len, type;
    for (uint64_t i = 0; i < g->n_kv && r.ok; i++) {
        const char *k = rd_key(&r, &len, &type);
        if (r.ok && is_key(k, len, key)) {
            const union { uint32_t u; int32_t i; float f; } x = {(uint32_t)rd_u(&r, 4)};
            const double v = type == GGUF_U32 ? (double)x.u : type == GGUF_I32 ? (double)x.i : (double)x.f;
            if (!r.ok || (type != GGUF_U32 && type != GGUF_I32 && type != GGUF_F32) || !isfinite(v)) return 0;
            return *out = v, 1;
        }
        skip_value(&r, type, 0);
    }
    return 0;
}

/* bytes for all string entries as NUL-terminated pairs; with dst, copies the first max into dst and kv[2 * max] */
static size_t sm_gguf_strings(const sm_gguf *g, char *dst, const char **kv, uint32_t max, uint32_t *n) {
    rd r = {g->kv, g->base + g->size, 1};
    size_t bytes = 0;
    uint32_t kl, vl, type;
    *n = 0;
    for (uint64_t i = 0; i < g->n_kv && r.ok; i++) {
        const char *k = rd_key(&r, &kl, &type), *v = type == GGUF_STRING ? rd_str(&r, &vl) : NULL;
        if (!v) skip_value(&r, type, 0);
        if (v && dst && *n < max) {
            kv[2 * *n] = memcpy(dst + bytes, k, kl);
            kv[2 * *n + 1] = memcpy(dst + bytes + kl + 1, v, vl);
            dst[bytes + kl] = dst[bytes + kl + 1 + vl] = 0;
            ++*n;
        }
        bytes += v ? (size_t)kl + vl + 2 : 0;
    }
    return bytes;
}

/* zlib's CRC-32; the table is built per call: no global state */
static uint32_t sm_crc32(const uint8_t *p, size_t n) {
    uint32_t tab[256], c = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t v = i;
        for (int k = 0; k < 8; k++) v = v & 1 ? 0xEDB88320u ^ (v >> 1) : v >> 1;
        tab[i] = v;
    }
    for (size_t i = 0; i < n; i++) c = tab[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

#define ALIGN 64   /* SIMD rows, cache lines */

uint32_t sm_version(void) { return SM_VERSION; }

const char *sm_build_info(void) {
#if defined(__wasm_simd128__)
    return "1.0.0 wasm-simd128 exact";
#elif defined(__wasm__)
    return "1.0.0 wasm-plain exact";
#elif defined(SM_FMA)
    return "1.0.0 native fma (not bit-identical to other builds)";
#else
    return "1.0.0 native exact";
#endif
}

#define ST(s, msg) {s, #s, msg}
static const struct { sm_status s; const char *name, *msg; } STATUS[] = {
    ST(SM_OK, "ok"), ST(SM_DONE, "done"), ST(SM_E_ARG, "invalid argument"),
    ST(SM_E_FORMAT, "not a valid SprongMotion model file"), ST(SM_E_VERSION, "the model file needs a newer runtime"),
    ST(SM_E_CHECKSUM, "the model file is corrupted (checksum)"), ST(SM_E_NOMEM, "out of memory"),
    ST(SM_E_BUFFER, "buffer too small"), ST(SM_E_CANCELLED, "cancelled"), ST(SM_E_NUMERIC, "NaN or Inf met"),
    ST(SM_E_UNSUPPORTED, "this model file has no such input"), ST(SM_E_IO, "file read error"),
    ST(SM_E_STATE, "call out of order"), ST(SM_E_INTERNAL, "internal error"),
};
static int status_at(sm_status s) {
    for (int i = 0; i < (int)(sizeof STATUS / sizeof STATUS[0]); i++)
        if (STATUS[i].s == s) return i;
    return -1;
}
const char *sm_status_name(sm_status s) { return status_at(s) < 0 ? "SM_E_UNKNOWN" : STATUS[status_at(s)].name; }
const char *sm_status_message(sm_status s) { return status_at(s) < 0 ? "unknown status" : STATUS[status_at(s)].msg; }

void *sm_alloc(const sm_hooks *h, size_t size) {
    if (h->alloc) return h->alloc(h->user, size, ALIGN);
#ifdef _WIN32
    return _aligned_malloc(size, ALIGN);
#else
    void *p = NULL;
    return posix_memalign(&p, ALIGN, size ? size : 1) ? NULL : p;
#endif
}

void sm_free(const sm_hooks *h, void *p) {
    if (h->free && p) h->free(h->user, p);
#ifdef _WIN32
    if (!h->free) _aligned_free(p);
#else
    if (!h->free) free(p);
#endif
}

const void *sm_need(const sm_model *m, const char *name, uint32_t type, uint32_t nd, uint64_t *dims, char *err,
                    size_t en) {
    const void *p = sm_gguf_tensor(&m->g, name, type, nd, dims);
    if (!p && err && en) snprintf(err, en, "tensor %s missing or with the wrong type or shape", name);
    return p;
}

static int u32_of(const sm_gguf *g, const char *key, uint32_t *out) {
    double v;
    if (!sm_gguf_num(g, key, &v) || v < 0 || v > UINT32_MAX || v != (double)(uint32_t)v) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int by_text(const void *a, const void *b) { return strcmp(*(const char *const *)a, *(const char *const *)b); }

static sm_status bind_tok(sm_model *m, char *err, size_t en) {
    sm_tok *t = &m->tok;
    const int32_t *keep = NULL;
    uint32_t n_keep = 0, n_bytes = 0;
    const struct { const int32_t **p; const char *name; uint32_t nd, d0, *count; } tabs[] = {   /* d0 0: any */
        {&t->merges, "tok.merges", 2, 2, &t->n_merges}, {&keep, "tok.keep", 1, 0, &n_keep},
        {&t->byte_id, "tok.byte_id", 1, SM_TOK_BYTES, &n_bytes}, {&t->uni_class, "tok.uni_class", 2, 3, &t->n_class},
        {&t->uni_lower, "tok.uni_lower", 2, 2, &t->n_lower}, {&t->ent_index, "tok.ent_index", 2, 4, &t->n_ent},
        {&t->ent_names, "tok.ent_names", 1, 0, &t->n_ent_bytes}, {&t->num_fix, "tok.num_fix", 2, 2, &t->n_num_fix},
    };
    for (size_t i = 0; i < sizeof tabs / sizeof tabs[0]; i++) {
        uint64_t d[2] = {tabs[i].d0, 0};
        if (!(*tabs[i].p = sm_need(m, tabs[i].name, SM_T_I32, tabs[i].nd, d, err, en))) return SM_E_FORMAT;
        *tabs[i].count = (uint32_t)d[tabs[i].nd - 1];
    }
    t->n_vocab = SM_TOK_BASE + t->n_merges + 2;
    if (t->sot != t->n_vocab - 2 || t->eot != t->n_vocab - 1) return sm_fail(err, en, SM_E_FORMAT, "bad tokenizer.sot");
    /* merge r may only use ids made before it */
    for (uint32_t r = 0; r < 2 * t->n_merges; r++)
        if (t->merges[r] < 0 || (uint32_t)t->merges[r] >= SM_TOK_BASE + r / 2)
            return sm_fail(err, en, SM_E_FORMAT, "bad tokenizer merge");
    for (int b = 0; b < SM_TOK_BYTES; b++)
        if (t->byte_id[b] < 0 || t->byte_id[b] >= SM_TOK_BYTES) return sm_fail(err, en, SM_E_FORMAT, "bad tok.byte_id");
    for (uint32_t i = 0; i < t->n_ent; i++) {
        const int32_t *e = t->ent_index + 4 * i;
        if (e[0] < 0 || e[1] < 1 || e[1] > 32 || (uint32_t)e[0] + (uint32_t)e[1] > t->n_ent_bytes ||
            e[2] < 0 || e[2] > 0x10FFFF || e[3] < -1 || e[3] > 0x10FFFF)
            return sm_fail(err, en, SM_E_FORMAT, "bad HTML entity table");
    }
    t->unk = n_keep, t->pad = n_keep + 1;
    t->remap = sm_alloc(&m->h, sizeof(int32_t) * t->n_vocab);
    m->tok_table = sm_alloc(&m->h, sm_tok_build(t, NULL));
    if (!t->remap || !m->tok_table) return sm_fail(err, en, SM_E_NOMEM, "out of memory");
    for (uint32_t i = 0; i < t->n_vocab; i++) t->remap[i] = (int32_t)t->unk;
    for (uint32_t i = 0; i < n_keep; i++) {
        if (keep[i] < 0 || (uint32_t)keep[i] >= t->n_vocab) return sm_fail(err, en, SM_E_FORMAT, "bad tok.keep");
        t->remap[keep[i]] = (int32_t)i;
    }
    sm_tok_build(t, m->tok_table);
    uint32_t n;
    const size_t bytes = sm_tok_words(t, NULL, &n);
    const char **w = sm_alloc(&m->h, sizeof(char *) * n + bytes);
    if (!(m->vocab = w)) return sm_fail(err, en, SM_E_NOMEM, "out of memory");
    char *p = (char *)(w + n);
    sm_tok_words(t, p, &n);
    for (uint32_t i = 0; i < n; p += strlen(p) + 1) w[i++] = p;
    qsort(w, n, sizeof *w, by_text);
    m->n_vocab = n;
    return SM_OK;
}

typedef struct { sm_conv *c; const char *name; uint64_t in, out, k; } conv_def;

static int bind_convs(sm_model *m, const conv_def *d, size_t n, char *err, size_t en) {
    for (size_t i = 0; i < n; i++) {
        char w[32], b[32];
        uint64_t dw[3] = {d[i].out, d[i].in, d[i].k}, db[1] = {d[i].out};
        snprintf(w, sizeof w, "vq.%s.w", d[i].name), snprintf(b, sizeof b, "vq.%s.b", d[i].name);
        if (!(d[i].c->w = sm_need(m, w, SM_T_F32, 3, dw, err, en)) ||
            !(d[i].c->b = sm_need(m, b, SM_T_F32, 1, db, err, en)))
            return 0;
    }
    return 1;
}

/* the decoder, and the encoder when sprongmotion.vq.encoder = 1 */
static sm_status bind_vq(sm_model *m, char *err, size_t en) {
    sm_vq *q = &m->vq;
    sm_vq_enc *e = &q->enc;
    q->par = &m->par;
    const uint32_t W = SM_VQ_W;
    const conv_def dec[] = {
        {&q->in, "in", W, W, 3}, {&q->up, "up", W, W, 3}, {&q->mid, "mid", W, W, 3}, {&q->out, "out", W, SM_F, 3},
        {&q->res1[0], "res0.a", W, W, 3}, {&q->res2[0], "res0.b", W, W, 1}, {&q->res1[1], "res1.a", W, W, 3},
        {&q->res2[1], "res1.b", W, W, 1}, {&q->res1[2], "res2.a", W, W, 3}, {&q->res2[2], "res2.b", W, W, 1},
    };
    const conv_def enc[] = {
        {&e->in, "enc.in", SM_F, W, 3}, {&e->down, "enc.down", W, W, 4}, {&e->out, "enc.out", W, W, 3},
        {&e->res1[0], "enc.res0.a", W, W, 3}, {&e->res2[0], "enc.res0.b", W, W, 1},
        {&e->res1[1], "enc.res1.a", W, W, 3}, {&e->res2[1], "enc.res1.b", W, W, 1},
        {&e->res1[2], "enc.res2.a", W, W, 3}, {&e->res2[2], "enc.res2.b", W, W, 1},
    };
    uint32_t flag = 0;
    q->has_enc = u32_of(&m->g, "sprongmotion.vq.encoder", &flag) && flag == 1;
    if (!bind_convs(m, dec, sizeof dec / sizeof dec[0], err, en) ||
        (q->has_enc && !bind_convs(m, enc, sizeof enc / sizeof enc[0], err, en)))
        return SM_E_FORMAT;
    uint64_t dc[3] = {SM_VQ_W, SM_VQ_N, SM_VQ_LAYERS}, dm[1] = {SM_F}, ds[1] = {SM_F};
    return (q->codebook = sm_need(m, "vq.codebook", SM_T_F32, 3, dc, err, en)) &&
           (q->mean = sm_need(m, "vq.mean", SM_T_F32, 1, dm, err, en)) &&
           (q->std = sm_need(m, "vq.std", SM_T_F32, 1, ds, err, en)) ? SM_OK : SM_E_FORMAT;
}

/* the rest pose: the 22-joint skeleton, every bone longer than 0.1 mm */
static sm_status bind_skeleton(sm_model *m, char *err, size_t en) {
    uint64_t dp[1] = {SM_J}, doff[2] = {3, SM_J};
    const int32_t *par = sm_need(m, "skeleton.parents", SM_T_I32, 1, dp, err, en);
    if (!par || !(m->rest = sm_need(m, "skeleton.offsets", SM_T_F32, 2, doff, err, en))) return SM_E_FORMAT;
    for (int j = 0; j < SM_J; j++) {
        const float *o = m->rest + 3 * j;
        const double len = sqrt((double)o[0] * o[0] + (double)o[1] * o[1] + (double)o[2] * o[2]);
        if (par[j] != sm_parent[j] || !(j ? len > 1e-4 && len < 10 : len == 0))
            return sm_fail(err, en, SM_E_FORMAT, "the skeleton is not sprong-22");
    }
    return SM_OK;
}

static sm_status bind_all(sm_model *m, uint32_t flags, char *err, size_t en) {
    const sm_gguf *g = &m->g;
    uint32_t version = 0;
    if (!u32_of(g, "sprongmotion.format_version", &version))
        return sm_fail(err, en, SM_E_FORMAT, "no sprongmotion.format_version");
    if (version != 2) return sm_fail(err, en, SM_E_VERSION, "model format version is newer than this runtime");
    if (!u32_of(g, "sprongmotion.data_crc32", &m->crc))
        return sm_fail(err, en, SM_E_FORMAT, "no sprongmotion.data_crc32");
    if (!(flags & SM_LOAD_NOCHECK) && sm_crc32(g->data, g->data_size) != m->crc)
        return sm_fail(err, en, SM_E_CHECKSUM, "model data checksum mismatch");
    const struct { const char *key; uint32_t *v; } settings[] = {
        {"sprongmotion.fps", &m->fps}, {"sprongmotion.frames_per_token", &m->frames_per_token},
        {"sprongmotion.max_frames", &m->max_frames}, {"sprongmotion.tokenizer.max_len", &m->tok.max_len},
        {"sprongmotion.tokenizer.sot", &m->tok.sot}, {"sprongmotion.tokenizer.eot", &m->tok.eot},
    };
    for (size_t i = 0; i < sizeof settings / sizeof settings[0]; i++)
        if (!u32_of(g, settings[i].key, settings[i].v)) return sm_fail(err, en, SM_E_FORMAT, "missing setting");
    if (m->tok.max_len < 3 || m->tok.max_len > 64) return sm_fail(err, en, SM_E_FORMAT, "bad tokenizer.max_len");
    if (!m->fps || !m->frames_per_token || m->max_frames < m->frames_per_token)
        return sm_fail(err, en, SM_E_FORMAT, "bad frame settings");
    sm_status s = bind_tok(m, err, en);
    if (s == SM_OK) s = bind_vq(m, err, en);
    if (s == SM_OK) s = sm_bind_gen(m, err, en);
    if (s == SM_OK) s = bind_skeleton(m, err, en);
    if (s == SM_OK) s = sm_bind_params(m, err, en);
    if (s != SM_OK) return s;
    if (!(m->meta_mem = sm_alloc(&m->h, sm_gguf_strings(g, NULL, NULL, 0, &m->n_meta) + 1)))
        return sm_fail(err, en, SM_E_NOMEM, "out of memory");
    sm_gguf_strings(g, m->meta_mem, m->meta, SM_MAX_META, &m->n_meta);
    const char *name = sm_model_meta(m, "general.name");
    snprintf(m->id, sizeof m->id, "%.64s@%08x", name ? name : "unnamed", m->crc);
    uint64_t params = 0;
    for (uint64_t i = 0; i < g->n_t; i++) {   /* the generator: everything but the tokenizer and the decoder */
        uint64_t n = 1;
        for (uint32_t d = 0; d < g->t[i].nd; d++) n *= g->t[i].dim[d];
        if (strncmp(g->t[i].name, "tok.", 4) && strncmp(g->t[i].name, "vq.", 3)) params += n;
    }
    m->params_k = m->gen.layers ? (int32_t)(params / 1000) : 0;
    if (!(m->schema = sm_alloc(&m->h, sm_schema(m, NULL, 0)))) return sm_fail(err, en, SM_E_NOMEM, "out of memory");
    sm_schema(m, m->schema, sm_schema(m, NULL, 0));
    return SM_OK;
}

sm_status sm_model_load_memory(const void *bytes, size_t size, uint32_t flags, const sm_hooks *hooks, sm_model **out,
                               char *err, size_t en) {
    if (err && en) err[0] = 0;
    if (!bytes || !out) return sm_fail(err, en, SM_E_ARG, "bytes and out must not be NULL");
    *out = NULL;
    sm_hooks h = {0};
    if (hooks && (!hooks->struct_size || hooks->struct_size > sizeof h))
        return sm_fail_range(err, en, SM_E_ARG, "sm_hooks.struct_size", hooks->struct_size, 1, sizeof h);
    if (hooks) memcpy(&h, hooks, hooks->struct_size);
    if (!h.alloc != !h.free) return sm_fail(err, en, SM_E_ARG, "set both alloc and free, or neither");
    const uint64_t nt = sm_gguf_count_tensors(bytes, size);
    if (!nt) return sm_fail(err, en, SM_E_FORMAT, "not a GGUF v3 file");
    sm_model *m = sm_alloc(&h, sizeof *m);
    if (!m) return sm_fail(err, en, SM_E_NOMEM, "out of memory");
    memset(m, 0, sizeof *m);
    m->h = h;
    const uint8_t *buf = bytes;
    if (!(flags & SM_LOAD_BORROW)) {   /* own a copy, so the caller may free theirs */
        if ((m->own = sm_alloc(&h, size))) memcpy(m->own, bytes, size);
        buf = m->own;
    }
    sm_status s = buf && (m->tensors = sm_alloc(&h, sizeof(sm_tensor) * (size_t)nt))
                ? sm_gguf_open(&m->g, buf, size, m->tensors, nt, err, en)
                : sm_fail(err, en, SM_E_NOMEM, "out of memory");
    if (s == SM_OK) s = bind_all(m, flags, err, en);
    if (s != SM_OK) return sm_model_free(m), s;
    if (h.parallel_for) m->par = (sm_par){h.parallel_for, h.user};
    else if ((m->pool = sm_pool_create(&h, h.n_threads))) m->par = (sm_par){sm_pool_run, m->pool};
    *out = m;
    return s;
}

#ifdef _WIN32
static wchar_t *widen(const char *utf8) {   /* UTF-8 -> UTF-16, malloc'd */
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, NULL, 0);
    wchar_t *w = n > 0 ? malloc(sizeof(wchar_t) * (size_t)n) : NULL;
    if (w && MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, w, n) != n) w = (free(w), NULL);
    return w;
}

static const void *map_file(const char *path, size_t *size) {
    wchar_t *w = widen(path);
    HANDLE f = w ? CreateFileW(w, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)
                 : INVALID_HANDLE_VALUE;
    free(w);
    if (f == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER n;
    HANDLE map = GetFileSizeEx(f, &n) && n.QuadPart > 0 ? CreateFileMappingW(f, NULL, PAGE_READONLY, 0, 0, NULL) : NULL;
    const void *p = map ? MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0) : NULL;
    if (map) CloseHandle(map);
    CloseHandle(f);
    *size = p ? (size_t)n.QuadPart : 0;
    return p;
}

static void unmap(const void *p, size_t n) { (void)n; UnmapViewOfFile(p); }

static FILE *open_file(const char *path) {
    wchar_t *w = widen(path);
    FILE *f = w ? _wfopen(w, L"rb") : NULL;
    free(w);
    return f;
}
#elif defined(__unix__) || defined(__APPLE__)
static const void *map_file(const char *path, size_t *size) {
    const int fd = open(path, O_RDONLY);
    struct stat st;
    void *p = fd >= 0 && !fstat(fd, &st) && st.st_size > 0
            ? mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
    if (fd >= 0) close(fd);
    *size = p == MAP_FAILED ? 0 : (size_t)st.st_size;
    return p == MAP_FAILED ? NULL : p;
}

static void unmap(const void *p, size_t n) { munmap((void *)(uintptr_t)p, n); }
static FILE *open_file(const char *path) { return fopen(path, "rb"); }
#else
static const void *map_file(const char *path, size_t *size) { (void)path; *size = 0; return NULL; }
static void unmap(const void *p, size_t n) { (void)p; (void)n; }
static FILE *open_file(const char *path) { return fopen(path, "rb"); }
#endif

sm_status sm_model_load_file(const char *path, const sm_hooks *hooks, sm_model **out, char *err, size_t en) {
    if (!path || !out) return sm_fail(err, en, SM_E_ARG, "path and out must not be NULL");
    size_t n = 0;
    const void *map = map_file(path, &n);
    if (map) {
        const sm_status s = sm_model_load_memory(map, n, SM_LOAD_BORROW, hooks, out, err, en);
        if (s != SM_OK) unmap(map, n);
        else (*out)->map = map, (*out)->map_size = n;
        return s;
    }
    FILE *f = open_file(path);
    if (!f) return sm_fail(err, en, SM_E_IO, "cannot open the model file");
    const long size = fseek(f, 0, SEEK_END) == 0 ? ftell(f) : -1;
    void *buf = size > 0 && fseek(f, 0, SEEK_SET) == 0 ? malloc((size_t)size) : NULL;
    const int ok = buf && fread(buf, 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    const sm_status s = ok ? sm_model_load_memory(buf, (size_t)size, 0, hooks, out, err, en)
                           : sm_fail(err, en, SM_E_IO, "cannot read the model file");
    free(buf);
    return s;
}

void sm_model_free(sm_model *m) {
    if (!m) return;
    const sm_hooks h = m->h;
    const void *map = m->map;
    const size_t map_size = m->map_size;
    sm_pool_free(&h, m->pool);
    void *owned[] = {m->schema, m->meta_mem, m->vocab, m->tok_table, m->tok.remap, m->tensors, m->own, m};
    for (size_t i = 0; i < sizeof owned / sizeof owned[0]; i++) sm_free(&h, owned[i]);
    if (map) unmap(map, map_size);
}

int32_t sm_model_info(const sm_model *m, sm_info key) {   /* by key: fps .. path input; 7 and 8 are unused */
    if (!m || key < SM_I_FPS || key > SM_I_PATH_INPUT) return -1;
    const int32_t v[] = {0, (int32_t)m->fps, (int32_t)m->frames_per_token, (int32_t)m->max_frames, SM_J, SM_F,
                         m->params_k, -1, -1, SM_VQ_LAYERS, (int32_t)m->text.out.n, m->vq.has_enc, 0};
    return v[key];
}

const char *sm_model_meta(const sm_model *m, const char *key) {
    for (uint32_t i = 0; m && key && i < m->n_meta; i++)
        if (!strcmp(m->meta[2 * i], key)) return m->meta[2 * i + 1];
    return NULL;
}

const char *sm_model_id(const sm_model *m) { return m ? m->id : NULL; }
const char *sm_model_vocab(const sm_model *m, uint32_t i) { return m && i < m->n_vocab ? m->vocab[i] : NULL; }
