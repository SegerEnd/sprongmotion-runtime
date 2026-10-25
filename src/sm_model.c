/* The model loader, which checks every tensor's name, type and shape, and the library basics. */
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

int32_t sm_fail(char *err, size_t err_len, int32_t status, const char *msg) {
    if (err && err_len) snprintf(err, err_len, "%s", msg);
    return status;
}

int32_t sm_fail_range(char *err, size_t err_len, int32_t status, const char *name, double v, double lo, double hi) {
    sm_buf b = {err, err_len ? err_len - 1 : 0, 0};
    sm_puts(&b, name);
    sm_puts(&b, " = ");
    if (isfinite(v)) sm_putn(&b, v);
    else sm_puts(&b, v != v ? "nan" : v > 0 ? "inf" : "-inf");
    sm_puts(&b, lo <= v && v <= hi ? " is not a whole number in " : " is out of range ");
    sm_putn(&b, lo);
    sm_puts(&b, " .. ");
    sm_putn(&b, hi);
    if (err && err_len) err[b.n < b.cap ? b.n : b.cap] = 0;
    return status;
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
                    size_t err_len) {
    const void *p = sm_gguf_tensor(&m->g, name, type, nd, dims);
    if (!p && err && err_len) snprintf(err, err_len, "tensor %s missing or with the wrong type or shape", name);
    return p;
}

static int u32_of(const sm_gguf *g, const char *key, uint32_t *out) {
    double v;
    if (!sm_gguf_num(g, key, &v) || v < 0 || v > UINT32_MAX || v != (double)(uint32_t)v) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int by_text(const void *a, const void *b) { return strcmp(*(const char *const *)a, *(const char *const *)b); }

static sm_status find_tok_tables(sm_model *m, const int32_t **keep, uint32_t *n_keep, char *err, size_t err_len) {
    sm_tok *t = &m->tok;
    uint32_t n_bytes = 0;
    const struct { const int32_t **p; const char *name; uint32_t nd, d0, *count; } tables[] = {   /* d0 0: any */
        {&t->merges, "tok.merges", 2, 2, &t->n_merges}, {keep, "tok.keep", 1, 0, n_keep},
        {&t->byte_id, "tok.byte_id", 1, SM_TOK_BYTES, &n_bytes}, {&t->uni_class, "tok.uni_class", 2, 3, &t->n_class},
        {&t->uni_lower, "tok.uni_lower", 2, 2, &t->n_lower}, {&t->ent_index, "tok.ent_index", 2, 4, &t->n_ent},
        {&t->ent_names, "tok.ent_names", 1, 0, &t->n_ent_bytes}, {&t->num_fix, "tok.num_fix", 2, 2, &t->n_num_fix},
    };
    for (size_t i = 0; i < sizeof tables / sizeof tables[0]; i++) {
        uint64_t dims[2] = {tables[i].d0, 0};
        *tables[i].p = sm_need(m, tables[i].name, SM_T_I32, tables[i].nd, dims, err, err_len);
        if (!*tables[i].p) return SM_E_FORMAT;
        *tables[i].count = (uint32_t)dims[tables[i].nd - 1];
    }
    return SM_OK;
}

static sm_status check_tok_tables(const sm_tok *t, char *err, size_t err_len) {
    if (t->sot != t->n_vocab - 2 || t->eot != t->n_vocab - 1)
        return sm_fail(err, err_len, SM_E_FORMAT, "bad tokenizer.sot");
    for (uint32_t r = 0; r < 2 * t->n_merges; r++)   /* merge r may only use ids made before it */
        if (t->merges[r] < 0 || (uint32_t)t->merges[r] >= SM_TOK_BASE + r / 2)
            return sm_fail(err, err_len, SM_E_FORMAT, "bad tokenizer merge");
    for (int b = 0; b < SM_TOK_BYTES; b++)
        if (t->byte_id[b] < 0 || t->byte_id[b] >= SM_TOK_BYTES)
            return sm_fail(err, err_len, SM_E_FORMAT, "bad tok.byte_id");
    for (uint32_t i = 0; i < t->n_ent; i++) {
        const int32_t *e = t->ent_index + 4 * i;
        const int bad_name = e[0] < 0 || e[1] < 1 || e[1] > 32 || (uint32_t)e[0] + (uint32_t)e[1] > t->n_ent_bytes;
        const int bad_cp = e[2] < 0 || e[2] > 0x10FFFF || e[3] < -1 || e[3] > 0x10FFFF;
        if (bad_name || bad_cp) return sm_fail(err, err_len, SM_E_FORMAT, "bad HTML entity table");
    }
    return SM_OK;
}

/* CLIP ids outside the kept vocabulary map to UNK */
static sm_status build_remap(sm_model *m, const int32_t *keep, uint32_t n_keep, char *err, size_t err_len) {
    sm_tok *t = &m->tok;
    t->unk = n_keep;
    t->pad = n_keep + 1;
    t->remap = sm_alloc(&m->h, sizeof(int32_t) * t->n_vocab);
    m->tok_table = sm_alloc(&m->h, sm_tok_build(t, NULL));
    if (!t->remap || !m->tok_table) return sm_fail(err, err_len, SM_E_NOMEM, "out of memory");
    for (uint32_t i = 0; i < t->n_vocab; i++) t->remap[i] = (int32_t)t->unk;
    for (uint32_t i = 0; i < n_keep; i++) {
        if (keep[i] < 0 || (uint32_t)keep[i] >= t->n_vocab) return sm_fail(err, err_len, SM_E_FORMAT, "bad tok.keep");
        t->remap[keep[i]] = (int32_t)i;
    }
    sm_tok_build(t, m->tok_table);
    return SM_OK;
}

/* the whole words, sorted, in one block with their pointers */
static sm_status build_vocab(sm_model *m, char *err, size_t err_len) {
    uint32_t n;
    const size_t bytes = sm_tok_words(&m->tok, NULL, &n);
    const char **words = sm_alloc(&m->h, sizeof(char *) * n + bytes);
    if (!(m->vocab = words)) return sm_fail(err, err_len, SM_E_NOMEM, "out of memory");
    char *p = (char *)(words + n);
    sm_tok_words(&m->tok, p, &n);
    for (uint32_t i = 0; i < n; i++) {
        words[i] = p;
        p += strlen(p) + 1;
    }
    qsort(words, n, sizeof *words, by_text);
    m->n_vocab = n;
    return SM_OK;
}

static sm_status bind_tok(sm_model *m, char *err, size_t err_len) {
    const int32_t *keep = NULL;
    uint32_t n_keep = 0;
    sm_status s = find_tok_tables(m, &keep, &n_keep, err, err_len);
    if (s != SM_OK) return s;
    m->tok.n_vocab = SM_TOK_BASE + m->tok.n_merges + 2;
    s = check_tok_tables(&m->tok, err, err_len);
    if (s == SM_OK) s = build_remap(m, keep, n_keep, err, err_len);
    if (s == SM_OK) s = build_vocab(m, err, err_len);
    return s;
}

#define FORMAT_VERSION 2
#define MIN_BONE 1e-4   /* m */
#define MAX_BONE 10

typedef struct { sm_conv *c; const char *name; uint64_t in, out, k; } conv_def;

static int bind_convs(sm_model *m, const conv_def *defs, size_t n, char *err, size_t err_len) {
    for (size_t i = 0; i < n; i++) {
        const conv_def *d = &defs[i];
        char weight[32], bias[32];
        uint64_t weight_dims[3] = {d->out, d->in, d->k}, bias_dims[1] = {d->out};
        snprintf(weight, sizeof weight, "vq.%s.w", d->name);
        snprintf(bias, sizeof bias, "vq.%s.b", d->name);
        if (!(d->c->w = sm_need(m, weight, SM_T_F32, 3, weight_dims, err, err_len))) return 0;
        if (!(d->c->b = sm_need(m, bias, SM_T_F32, 1, bias_dims, err, err_len))) return 0;
    }
    return 1;
}

/* the decoder, and the encoder when sprongmotion.vq.encoder = 1 */
static sm_status bind_vq(sm_model *m, char *err, size_t err_len) {
    sm_vq *q = &m->vq;
    sm_vq_enc *e = &q->enc;
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
    q->par = &m->par;
    q->has_enc = u32_of(&m->g, "sprongmotion.vq.encoder", &flag) && flag == 1;
    if (!bind_convs(m, dec, sizeof dec / sizeof dec[0], err, err_len)) return SM_E_FORMAT;
    if (q->has_enc && !bind_convs(m, enc, sizeof enc / sizeof enc[0], err, err_len)) return SM_E_FORMAT;

    uint64_t codebook_dims[3] = {SM_VQ_W, SM_VQ_N, SM_VQ_LAYERS}, mean_dims[1] = {SM_F}, std_dims[1] = {SM_F};
    if (!(q->codebook = sm_need(m, "vq.codebook", SM_T_F32, 3, codebook_dims, err, err_len))) return SM_E_FORMAT;
    if (!(q->mean = sm_need(m, "vq.mean", SM_T_F32, 1, mean_dims, err, err_len))) return SM_E_FORMAT;
    if (!(q->std = sm_need(m, "vq.std", SM_T_F32, 1, std_dims, err, err_len))) return SM_E_FORMAT;
    return SM_OK;
}

/* the rest pose: the 22-joint skeleton, every bone longer than 0.1 mm */
static sm_status bind_skeleton(sm_model *m, char *err, size_t err_len) {
    uint64_t parent_dims[1] = {SM_J}, offset_dims[2] = {3, SM_J};
    const int32_t *parents = sm_need(m, "skeleton.parents", SM_T_I32, 1, parent_dims, err, err_len);
    if (!parents) return SM_E_FORMAT;
    if (!(m->rest = sm_need(m, "skeleton.offsets", SM_T_F32, 2, offset_dims, err, err_len))) return SM_E_FORMAT;
    for (int j = 0; j < SM_J; j++) {
        const float *o = m->rest + 3 * j;
        const double len = sqrt((double)o[0] * o[0] + (double)o[1] * o[1] + (double)o[2] * o[2]);
        const int fits = j ? len > MIN_BONE && len < MAX_BONE : len == 0;
        if (parents[j] != sm_parent[j] || !fits)
            return sm_fail(err, err_len, SM_E_FORMAT, "the skeleton is not sprong-22");
    }
    return SM_OK;
}

static sm_status check_file(sm_model *m, uint32_t flags, char *err, size_t err_len) {
    const sm_gguf *g = &m->g;
    uint32_t version = 0;
    if (!u32_of(g, "sprongmotion.format_version", &version))
        return sm_fail(err, err_len, SM_E_FORMAT, "no sprongmotion.format_version");
    if (version != FORMAT_VERSION)
        return sm_fail(err, err_len, SM_E_VERSION, "model format version is newer than this runtime");
    if (!u32_of(g, "sprongmotion.data_crc32", &m->crc))
        return sm_fail(err, err_len, SM_E_FORMAT, "no sprongmotion.data_crc32");
    if (!(flags & SM_LOAD_NOCHECK) && sm_crc32(g->data, g->data_size) != m->crc)
        return sm_fail(err, err_len, SM_E_CHECKSUM, "model data checksum mismatch");
    return SM_OK;
}

static sm_status read_settings(sm_model *m, char *err, size_t err_len) {
    const struct { const char *key; uint32_t *v; } settings[] = {
        {"sprongmotion.fps", &m->fps}, {"sprongmotion.frames_per_token", &m->frames_per_token},
        {"sprongmotion.max_frames", &m->max_frames}, {"sprongmotion.tokenizer.max_len", &m->tok.max_len},
        {"sprongmotion.tokenizer.sot", &m->tok.sot}, {"sprongmotion.tokenizer.eot", &m->tok.eot},
    };
    for (size_t i = 0; i < sizeof settings / sizeof settings[0]; i++)
        if (!u32_of(&m->g, settings[i].key, settings[i].v))
            return sm_fail(err, err_len, SM_E_FORMAT, "missing setting");
    if (m->tok.max_len < 3 || m->tok.max_len > 64) return sm_fail(err, err_len, SM_E_FORMAT, "bad tokenizer.max_len");
    if (!m->fps || !m->frames_per_token || m->max_frames < m->frames_per_token)
        return sm_fail(err, err_len, SM_E_FORMAT, "bad frame settings");
    return SM_OK;
}

/* the string metadata, the id, and the generator's size: all but the tokenizer and the decoder */
static sm_status describe(sm_model *m, char *err, size_t err_len) {
    const sm_gguf *g = &m->g;
    if (!(m->meta_mem = sm_alloc(&m->h, sm_gguf_strings(g, NULL, NULL, 0, &m->n_meta) + 1)))
        return sm_fail(err, err_len, SM_E_NOMEM, "out of memory");
    sm_gguf_strings(g, m->meta_mem, m->meta, SM_MAX_META, &m->n_meta);
    const char *name = sm_model_meta(m, "general.name");
    snprintf(m->id, sizeof m->id, "%.64s@%08x", name ? name : "unnamed", m->crc);

    uint64_t params = 0;
    for (uint64_t i = 0; i < g->n_t; i++) {
        uint64_t n = 1;
        for (uint32_t d = 0; d < g->t[i].nd; d++) n *= g->t[i].dim[d];
        if (strncmp(g->t[i].name, "tok.", 4) && strncmp(g->t[i].name, "vq.", 3)) params += n;
    }
    m->params_k = m->gen.layers ? (int32_t)(params / 1000) : 0;

    const size_t schema = sm_schema(m, NULL, 0);
    if (!(m->schema = sm_alloc(&m->h, schema))) return sm_fail(err, err_len, SM_E_NOMEM, "out of memory");
    sm_schema(m, m->schema, schema);
    return SM_OK;
}

static sm_status bind_all(sm_model *m, uint32_t flags, char *err, size_t err_len) {
    sm_status s = check_file(m, flags, err, err_len);
    if (s == SM_OK) s = read_settings(m, err, err_len);
    if (s == SM_OK) s = bind_tok(m, err, err_len);
    if (s == SM_OK) s = bind_vq(m, err, err_len);
    if (s == SM_OK) s = sm_bind_gen(m, err, err_len);
    if (s == SM_OK) s = bind_skeleton(m, err, err_len);
    if (s == SM_OK) s = sm_bind_params(m, err, err_len);
    if (s == SM_OK) s = describe(m, err, err_len);
    return s;
}

static sm_status copy_hooks(const sm_hooks *hooks, sm_hooks *h, char *err, size_t err_len) {
    memset(h, 0, sizeof *h);
    if (hooks && (!hooks->struct_size || hooks->struct_size > sizeof *h))
        return sm_fail_range(err, err_len, SM_E_ARG, "sm_hooks.struct_size", hooks->struct_size, 1, sizeof *h);
    if (hooks) memcpy(h, hooks, hooks->struct_size);
    if (!h->alloc != !h->free) return sm_fail(err, err_len, SM_E_ARG, "set both alloc and free, or neither");
    return SM_OK;
}

sm_status sm_model_load_memory(const void *bytes, size_t size, uint32_t flags, const sm_hooks *hooks, sm_model **out,
                               char *err, size_t err_len) {
    if (err && err_len) err[0] = 0;
    if (!bytes || !out) return sm_fail(err, err_len, SM_E_ARG, "bytes and out must not be NULL");
    *out = NULL;
    sm_hooks h;
    sm_status s = copy_hooks(hooks, &h, err, err_len);
    if (s != SM_OK) return s;
    const uint64_t n_tensors = sm_gguf_count_tensors(bytes, size);
    if (!n_tensors) return sm_fail(err, err_len, SM_E_FORMAT, "not a GGUF v3 file");

    sm_model *m = sm_alloc(&h, sizeof *m);
    if (!m) return sm_fail(err, err_len, SM_E_NOMEM, "out of memory");
    memset(m, 0, sizeof *m);
    m->h = h;
    const uint8_t *buf = bytes;
    if (!(flags & SM_LOAD_BORROW)) {   /* own a copy, so the caller may free theirs */
        if ((m->own = sm_alloc(&h, size))) memcpy(m->own, bytes, size);
        buf = m->own;
    }
    m->tensors = buf ? sm_alloc(&h, sizeof(sm_tensor) * (size_t)n_tensors) : NULL;
    s = m->tensors ? sm_gguf_open(&m->g, buf, size, m->tensors, n_tensors, err, err_len)
                   : sm_fail(err, err_len, SM_E_NOMEM, "out of memory");
    if (s == SM_OK) s = bind_all(m, flags, err, err_len);
    if (s != SM_OK) {
        sm_model_free(m);
        return s;
    }

    if (h.parallel_for) m->par = (sm_par){h.parallel_for, h.user};
    else if ((m->pool = sm_pool_create(&h, h.n_threads))) m->par = (sm_par){sm_pool_run, m->pool};
    *out = m;
    return SM_OK;
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

sm_status sm_model_load_file(const char *path, const sm_hooks *hooks, sm_model **out, char *err, size_t err_len) {
    if (!path || !out) return sm_fail(err, err_len, SM_E_ARG, "path and out must not be NULL");
    size_t n = 0;
    const void *map = map_file(path, &n);
    if (map) {
        const sm_status s = sm_model_load_memory(map, n, SM_LOAD_BORROW, hooks, out, err, err_len);
        if (s != SM_OK) {
            unmap(map, n);
            return s;
        }
        (*out)->map = map;
        (*out)->map_size = n;
        return s;
    }
    FILE *f = open_file(path);
    if (!f) return sm_fail(err, err_len, SM_E_IO, "cannot open the model file");
    const long size = fseek(f, 0, SEEK_END) == 0 ? ftell(f) : -1;
    void *buf = size > 0 && fseek(f, 0, SEEK_SET) == 0 ? malloc((size_t)size) : NULL;
    const int ok = buf && fread(buf, 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    const sm_status s = ok ? sm_model_load_memory(buf, (size_t)size, 0, hooks, out, err, err_len)
                           : sm_fail(err, err_len, SM_E_IO, "cannot read the model file");
    free(buf);
    return s;
}

void sm_model_free(sm_model *m) {
    if (!m) return;
    const sm_hooks h = m->h;
    const void *map = m->map;
    const size_t map_size = m->map_size;
    sm_pool_free(&h, m->pool);
    void *owned[] = {m->packed, m->schema, m->meta_mem, m->vocab, m->tok_table, m->tok.remap, m->tensors, m->own, m};
    for (size_t i = 0; i < sizeof owned / sizeof owned[0]; i++) sm_free(&h, owned[i]);
    if (map) unmap(map, map_size);
}

/* by key: fps .. path input; 7 and 8 are unused */
int32_t sm_model_info(const sm_model *m, sm_info key) {
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
