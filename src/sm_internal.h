/* Types and functions shared by the files in src/. Sizes: T frames, L tokens, W words, M rows. */
#ifndef SM_INTERNAL_H
#define SM_INTERNAL_H
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "../include/sm.h"

enum { SM_J = 22, SM_F = 263, SM_VQ_W = 256, SM_VQ_N = 512, SM_VQ_LAYERS = 6, SM_N_PARAMS = 14 };
enum {
    SM_PELVIS, SM_L_HIP, SM_R_HIP, SM_SPINE1, SM_L_KNEE, SM_R_KNEE, SM_SPINE2, SM_L_ANKLE, SM_R_ANKLE, SM_SPINE3,
    SM_L_FOOT, SM_R_FOOT, SM_NECK, SM_L_COLLAR, SM_R_COLLAR, SM_HEAD, SM_L_SHOULDER, SM_R_SHOULDER
};
/* where the parts of a feature frame start, after root yaw velocity, xz velocity and height */
enum { SM_F_POS = 4, SM_F_ROT = 67, SM_F_VEL = 193, SM_F_CONTACT = 259 };

/* GGUF v3 (sm_gguf.c) */
enum { SM_T_F32 = 0, SM_T_Q8_0 = 8, SM_T_I32 = 26 };
typedef struct {
    const char *name; uint32_t name_len;
    uint32_t type, nd; uint64_t dim[4];  /* innermost first */
    uint64_t offset; const uint8_t *data; size_t bytes;
} sm_tensor;
typedef struct {
    const uint8_t *base; size_t size;
    const uint8_t *kv; uint64_t n_kv;
    sm_tensor *t; uint64_t n_t;
    const uint8_t *data; size_t data_size;
} sm_gguf;
uint64_t    sm_gguf_count_tensors(const void *buf, size_t size);   /* 0 if not GGUF v3 */
int32_t     sm_gguf_open(sm_gguf *g, const void *buf, size_t size, sm_tensor *t, uint64_t max_t, char *err,
                         size_t err_len);
/* bytes of all string entries as NUL-terminated pairs; with dst, the first max go into dst and kv */
size_t      sm_gguf_strings(const sm_gguf *g, char *dst, const char **kv, uint32_t max, uint32_t *n);
uint32_t    sm_crc32(const uint8_t *p, size_t n);
/* a dim given as 0 matches any size and is set to it */
const void *sm_gguf_tensor(const sm_gguf *g, const char *name, uint32_t type, uint32_t nd, uint64_t *dims);
int         sm_gguf_num(const sm_gguf *g, const char *key, double *out);   /* a finite U32, I32 or F32 value */
int32_t     sm_fail(char *err, size_t err_len, int32_t status, const char *msg);
/* "name = v is out of range lo .. hi", or "is not a whole number in" for a v in range */
int32_t     sm_fail_range(char *err, size_t err_len, int32_t status, const char *name, double v, double lo, double hi);

/* CLIP BPE, ids remapped to the model's smaller vocabulary */
#define SM_TOK_MAX_CP 4096
#define SM_TOK_BYTES  256   /* symbols < 256: bytes, 256..511: bytes ending a word */
#define SM_TOK_BASE   512   /* merge r makes symbol 512 + r */
typedef struct {
    const int32_t *merges; uint32_t n_merges;        /* [n][2] */
    const int32_t *byte_id;                          /* [256] */
    const int32_t *uni_class; uint32_t n_class;      /* [n][3] first, last, 1 letter / 2 number */
    const int32_t *uni_lower; uint32_t n_lower;      /* [n][2] cp, lower (-1 = keep) */
    const int32_t *ent_index; uint32_t n_ent;        /* [n][4] name offset, length, cp1, cp2 (-1) */
    const int32_t *ent_names; uint32_t n_ent_bytes;
    const int32_t *num_fix; uint32_t n_num_fix;      /* [n][2] &#N; -> cp (-1 = removed) */
    int32_t *remap; uint32_t n_vocab;                /* CLIP id -> model id */
    uint64_t *pair_key; int32_t *pair_rank; uint32_t pair_mask;   /* (a << 32 | b) -> merge rank */
    uint32_t sot, eot, unk, pad, max_len;
} sm_tok;
typedef struct { uint32_t cp[2][SM_TOK_MAX_CP]; int32_t sym[SM_TOK_MAX_CP + 1]; uint32_t span[64][2]; } sm_tok_work;
int     sm_utf8_valid(const char *s, size_t n);
size_t  sm_tok_build(sm_tok *t, void *table);   /* table NULL: only the bytes it needs */
/* max_len ids (SOT .. EOT, PAD); returns the count before padding, or -1 for invalid UTF-8 */
int32_t sm_tok_encode(const sm_tok *t, const char *utf8, size_t len, sm_tok_work *w, int32_t *out);
/* the whole lowercase words that are one token, NUL-terminated, in id order; returns their bytes */
size_t  sm_tok_words(const sm_tok *t, char *buf, uint32_t *count);
/* the last encode's unknown words, '\n'-separated, as many as fit; returns the bytes all of them need */
size_t  sm_tok_unknown(const sm_tok *t, const sm_tok_work *w, const int32_t *ids, int32_t n, char *buf, size_t cap,
                       uint32_t *count);

/* run(user, n, fn, arg) calls fn(arg, i) for every i < n and returns when all are done */
typedef struct { void (*run)(void *user, uint32_t n, sm_task_fn fn, void *arg); void *user; } sm_par;
void  sm_par_for(const sm_par *p, uint32_t n, sm_task_fn fn, void *arg);   /* p NULL: in order on this thread */
void *sm_pool_create(const sm_hooks *h, uint32_t threads);                  /* 0: the core count; NULL: one thread */
void  sm_pool_run(void *pool, uint32_t n, sm_task_fn fn, void *arg);
void  sm_pool_free(const sm_hooks *h, void *pool);

/* e^x = 2^n e^r, |r| <= ln2 / 2 (Cody-Waite: n ln2_hi is exact), Taylor to r^7: the same bits on every target */
static inline float sm_expf(float x) {
    const float c = x > 88 ? 88 : x < -87.3f ? -87.3f : x, k = c * 1.44269504f;
    const int32_t n = (int32_t)(k < 0 ? k - 0.5f : k + 0.5f);
    const float r = c - (float)n * 0.693145751953125f - (float)n * 1.42860677e-6f;
    const float p = 1 + r * (1 + r * (1 / 2.f + r * (1 / 6.f + r * (1 / 24.f + r * (1 / 120.f + r * (1 / 720.f +
                    r / 5040.f))))));
    union { uint32_t u; float f; } s = {(uint32_t)(n + 127) << 23};
    return x < -87.3f ? 0 : x > 88 ? INFINITY : p * s.f;
}

/* erf(x) = x P(x^2) / Q(x^2) on |x| <= 3.92, else +-1: a relative minimax fit, <= 7 ulp in float */
static inline float sm_erff(float x) {
    const float a = x < 0 ? -x : x, z = a > 3.92f ? 3.92f : a, t = z * z;
    const float p = 1.128379146f + t * (1.905675771e-1f + t * (5.276642650e-2f + t * (3.771581562e-3f +
                    t * (2.848490791e-4f + t * 2.046595652e-6f))));
    const float q = 1 + t * (5.022188500e-1f + t * (1.141724361e-1f + t * (1.498124558e-2f +
                    t * (1.163673146e-3f + t * 3.805392859e-5f))));
    const float e = a > 3.92f ? 1 : z * (p / q);
    return x < 0 ? -e : e;
}

static inline float sm_gelu(float x) { return 0.5f * x * (1 + sm_erff(x * 0.70710678f)); }  /* x/2 (1 + erf(x/sqrt2)) */

/* NEON, SSE, wasm SIMD128, or scalar code, all in the same lane order */
typedef float v4 __attribute__((vector_size(16), aligned(4)));
static inline v4 sm_load4(const float *p) { v4 v; memcpy(&v, p, 16); return v; }
static inline void sm_store4(float *p, v4 v) { memcpy(p, &v, 16); }
static inline float sm_sum4(v4 v) { return (v[0] + v[1]) + (v[2] + v[3]); }

/* Y[M][n] = X[M][k] W^T + b, then the epilogue, summed in k order. W rows: k floats (F32) or k/32 blocks of
   {fp16 scale, 32 int8} (Q8_0). p: the weights dequantized as [n / 16][k][16], or NULL */
#define SM_Q8_BLOCK 34
typedef struct { const void *w; const float *b; uint32_t type, k, n; const float *p; } sm_lin;
typedef enum { SM_EPI_NONE, SM_EPI_GELU, SM_EPI_ADD } sm_epi;   /* ADD: Y += add ([M][n], not Y itself) */
float sm_half(uint16_t h);
void  sm_row(const sm_lin *table, uint32_t i, float *out);   /* row i of a [n][k] table, dequantized */
void  sm_pack(sm_lin *l, float *p);   /* p: k * n floats */
void  sm_linear(const sm_par *par, const sm_lin *l, const float *X, uint32_t M, float *Y, sm_epi epi, const float *add);

/* rows [M][d]; a pass holds several independent sequences (segments) */
#define SM_MAX_LAYERS 8
#define SM_MAX_SEGS   2
typedef struct { const float *w, *b; } sm_norm;
typedef struct { sm_lin qkv, out, ff1, ff2; sm_norm ln1, ln2; } sm_layer;
typedef struct { uint32_t start, len; } sm_seg;
typedef struct { const sm_seg *seg; uint32_t n_seg, rows; } sm_batch;
void   sm_layernorm(const sm_par *par, const float *x, uint32_t M, uint32_t d, sm_norm n, double eps, float *y);
size_t sm_layer_scratch_floats(uint32_t M, uint32_t d, uint32_t ff, uint32_t heads);
void   sm_layer_post(const sm_par *par, const sm_layer *l, const sm_batch *b, uint32_t heads, float *x, float *scratch);
void   sm_layer_pre(const sm_par *par, const sm_layer *l, const sm_batch *b, uint32_t heads, float *x, float *scratch);

/* the MoMask RVQ-VAE: codes [L][6] <-> features [2L][263] */
typedef struct { const float *w, *b; } sm_conv;  /* w: [k][in][out] */
typedef struct { sm_conv in, down, res1[3], res2[3], out; } sm_vq_enc;
typedef struct {
    const float *codebook, *mean, *std; sm_conv in, res1[3], res2[3], up, mid, out;
    sm_vq_enc enc; int has_enc; const sm_par *par;
} sm_vq;
size_t sm_vq_scratch_floats(uint32_t L);
void   sm_vq_decode(const sm_vq *v, const int16_t *codes, uint32_t L, float *feat, float *scratch);
size_t sm_vq_enc_scratch_floats(uint32_t T);
void   sm_vq_encode(const sm_vq *v, const float *feat, uint32_t T, int16_t *codes, float *scratch);

typedef struct {
    sm_lin emb, emb_up, out; const float *pos; sm_layer L[SM_MAX_LAYERS]; sm_norm ln;
    uint32_t layers, d, heads, max_len; const sm_par *par;
} sm_textenc;
typedef struct { sm_layer L[SM_MAX_LAYERS]; sm_lin cond, dense, out, word; sm_norm norm; } sm_head;
typedef struct {
    sm_head mask, res;
    sm_lin tok, in[SM_VQ_LAYERS], res_out[SM_VQ_LAYERS];   /* in, res_out: [1..5] */
    const float *pe, *word_type, *quant_w, *quant_b, *in_b;
    uint32_t layers, d, heads, max_tokens, word_dim; const sm_par *par;
} sm_gen;
typedef struct { uint32_t steps, res_layers, top_k; float cfg, res_cfg, temp; } sm_sampling;
size_t sm_text_scratch_floats(const sm_textenc *t);
/* ids [W] -> the pooled vector and the words [W][d], rounded to float16 */
void   sm_text_encode(const sm_textenc *t, const int32_t *ids, uint32_t W, float *vec, float *words, float *scratch);
size_t sm_gen_scratch_floats(const sm_gen *g, uint32_t W, uint32_t L);
/* the conditional rows every pass reuses, into scratch. The passes return raw logits [2L][n], conditional rows
   first; hist [L][d]: the summed res.in rows */
void   sm_gen_prepare(const sm_gen *g, const float *vec, const float *words, uint32_t W, uint32_t L, float *scratch);
const float *sm_gen_base(const sm_gen *g, uint32_t W, uint32_t L, const int32_t *ids, float *scratch);
const float *sm_gen_res(const sm_gen *g, uint32_t W, uint32_t L, uint32_t q, const float *hist, float *scratch);
/* a run, one pass at a time. codes [L][6], conf [L] (may be NULL). pins [L][6] (may be NULL): base code < 0 is
   generated; a pinned token keeps each layer >= -1 and generates those set to SM_FREE. The last tie tokens copy the
   first ones (a loop's wrap) */
#define SM_FREE (-2)
typedef struct {
    uint32_t W, L, pass, fixed; sm_sampling s; uint64_t seed; int16_t *codes; float *conf;
    const int16_t *pins; uint32_t tie;
} sm_run;
sm_status sm_gen_start(const sm_gen *g, sm_run *r, const float *vec, const float *words, float *scratch);
sm_status sm_gen_pass(const sm_gen *g, sm_run *r, float *scratch);   /* SM_DONE after the last pass */
float     sm_hash_gumbel(uint64_t seed, uint32_t stream, uint32_t step, uint32_t pos, uint32_t code);
typedef float (*sm_noise_fn)(void *user, uint32_t step, uint32_t pos, uint32_t code);
uint32_t sm_nmask(uint32_t i, uint32_t n, uint32_t free);   /* max(1, rint(cos(t pi / 2) free)), t = i / (n - 1) */
/* the nmask lowest scores, ties to the lower position */
void     sm_lowest(const float *score, uint32_t L, uint32_t nmask, uint8_t *mask);
typedef struct {
    const float *logits; uint32_t L, N; const uint8_t *rows; float temp; uint32_t top_k; sm_noise_fn noise; void *user;
    uint32_t step; int32_t *pick; float *prob;
} sm_picking;
void     sm_pick_rows(const sm_par *par, const sm_picking *p);
/* argmax over the top-k of lg / temp + noise, first max wins; prob = softmax(lg)[pick]; tmp: N floats */
void     sm_pick(const float *lg, uint32_t L, uint32_t N, const uint8_t *rows, float temp, uint32_t k,
                 sm_noise_fn noise, void *user, uint32_t step, int32_t *pick, float *prob, float *tmp);
double   sm_exp(double x);
double   sm_log(double x);
float    sm_round_half(float x);   /* to the nearest float16 and back, ties to even */

#define SM_MAX_META 32
struct sm_model {
    sm_hooks h; uint8_t *own; const void *map; size_t map_size;
    sm_gguf g; sm_tensor *tensors; sm_tok tok; void *tok_table; sm_vq vq; sm_textenc text; sm_gen gen;
    sm_par par; void *pool; float *packed;
    const float *rest;              /* [22][3] the reference skeleton */
    double param[SM_N_PARAMS][3];   /* min, max, default */
    uint32_t fps, frames_per_token, max_frames, crc; int32_t params_k;
    char *meta_mem, id[96], *schema; const char *meta[2 * SM_MAX_META]; uint32_t n_meta;
    const char **vocab; uint32_t n_vocab;
};
void       *sm_alloc(const sm_hooks *h, size_t size);
void        sm_free(const sm_hooks *h, void *p);
const void *sm_need(const sm_model *m, const char *name, uint32_t type, uint32_t nd, uint64_t *dims, char *err,
                    size_t err_len);
sm_status   sm_bind_gen(sm_model *m, char *err, size_t err_len);   /* a file without text.* has no generator */
sm_status   sm_bind_params(sm_model *m, char *err, size_t err_len);

/* poses in double */
typedef struct { double w, x, y, z; } sm_quat;
sm_quat sm_qmul(sm_quat a, sm_quat b);
void    sm_qrot(sm_quat q, const double v[3], double out[3]);
sm_quat sm_qbetween(const double u[3], const double v[3]);  /* unit u, v */
extern const int8_t sm_parent[SM_J];
typedef struct { double yaw, x, z; float vel[3]; int32_t moving; } sm_track;   /* the root after a clip; zero: start */
void sm_features_to_joints(const float *feat, uint32_t T, sm_track *tr, float *joints);   /* recover_from_ric */
void sm_rotations(const float *joints, uint32_t T, const float *rest, float *xyzw);   /* [T][22][4], parent-relative */
void sm_clip_offsets(const float *joints, const float *xyzw, uint32_t T, double *tmp, float *off);
/* delta: T * 3. yaw_added (may be NULL): the yaw a path warp added per frame, which a planted toe undoes */
void sm_foot_lock(float *joints, const uint8_t *contacts, uint32_t T, double *delta, const double *yaw_added);
/* where a foot has no contact flag for a while (runs), it is on the ground while low and slow */
void sm_fill_contacts(const float *joints, uint8_t *contacts, uint32_t T, double fps);
/* joints [T][22][3] -> features [T-1][263]; tmp: sm_feat_tmp_doubles(T) */
size_t sm_feat_tmp_doubles(uint32_t T);
void   sm_joints_to_features(const float *joints, uint32_t T, double *tmp, float *feat);

/* points [n][4]: t, x, z, heading */
#define SM_MAX_PATH 1024
typedef struct { float p[SM_MAX_PATH * 4]; uint32_t n; } sm_path;
sm_status sm_path_check(const sm_model *m, const sm_path *p, uint32_t tokens, char *err, size_t err_len);
/* the root xz smoothed (g [T][2]) and its speed */
void   sm_root_smooth(const float *joints, uint32_t T, double fps, double *g, float *speed);
/* with warp, the root onto the path and every joint turned with it. Returns the largest distance of g from the
   path's point at its time, before the warp. tmp: 2T + 6 (SM_MAX_PATH + 2) doubles */
double sm_path_warp(const sm_path *p, float *joints, uint32_t T, double fps, const double *g, int warp,
                    double *yaw_added, double *tmp);
#define SM_POST_TMP(T) (8 * (size_t)(T) + 6 * (SM_MAX_PATH + 2))   /* doubles sm_motion_post needs */

#define SM_MAX_TEXT   4096
#define SM_OVERLAP    3    /* tokens of a prefix clip the new one starts from */
#define SM_CONTEXT    8    /* tokens of a prefix clip the decoder sees */
#define SM_MAX_CODES  6000 /* tokens of one decode: 10 minutes */
#define SM_MAX_PINS   512  /* tokens of pinned codes in a request */
typedef struct { int16_t codes[SM_CONTEXT * SM_VQ_LAYERS]; uint32_t n; sm_track track; float last[SM_F]; } sm_prefix;
struct sm_text { const sm_model *m; float *vec, *words; uint32_t W, ready, n_unknown; char unknown[1024]; };
struct sm_request {   /* has_text 1: text, 2: enc */
    const sm_model *m; double v[SM_N_PARAMS]; sm_prefix pre; sm_text *enc; sm_path path;
    uint32_t has_text, text_len, n_pins; char text[SM_MAX_TEXT + 1], err[128]; int16_t pins[SM_MAX_PINS * SM_VQ_LAYERS];
};
struct sm_motion {   /* mem holds feat .. contacts */
    const sm_model *m; uint32_t cap, tokens, frames; int32_t seed; sm_track end; void *mem; float path_err;
    float *feat, *joints, *rot, *root, *speed, *conf, off[SM_J * 3]; int16_t *codes; uint8_t *contacts;
};
#define SM_MAX_VARIANTS 16
#define SM_MAX_RUN 512   /* tokens per run */
#define SM_TIE 12        /* tokens a loop makes past its end as its start again, then drops */
#define SM_WRAP 16       /* tokens of a loop's decode window past each end: more than the decoder's reach (15) */
#define SM_ACTIVE "a run is active: finish or abort it first"
#define SM_CANCELLED_MSG "cancel is set: call sm_ctx_clear_cancel"
struct sm_ctx {
    const sm_model *m;
    _Atomic int cancel;
    sm_progress_fn progress;
    void *user;
    void *scratch, *dec;                   /* text encoder or generator; decoder */
    size_t n_scratch, n_dec;               /* bytes */
    sm_tok_work *tw;
    sm_text *text;
    int active;
    uint32_t n, at, L, fresh, skip, tie;   /* variants, the one running, tokens made, kept, the prefix's, the wrap's */
    double v[SM_N_PARAMS];
    sm_prefix pre;
    sm_path path;
    sm_run run;
    sm_motion *outs[SM_MAX_VARIANTS];
    int16_t codes[SM_MAX_RUN * SM_VQ_LAYERS];
    float conf[SM_MAX_RUN];
    int16_t pins[SM_MAX_RUN * SM_VQ_LAYERS];
    char err[128];
};
sm_status sm_ctx_fail(sm_ctx *c, sm_status s, const char *msg);
sm_status sm_ctx_fail_range(sm_ctx *c, const char *name, double v, double lo, double hi);
sm_status sm_ctx_fit(sm_ctx *c, uint32_t L, uint32_t Lw);   /* room for generating L tokens and decoding Lw */
uint32_t  sm_tokens_of(const sm_model *m, double seconds);
sm_status sm_ctx_encode(sm_ctx *c, const char *utf8, size_t len, sm_text *t);
sm_status sm_ctx_decode(sm_ctx *c, const int16_t *codes, uint32_t n, const sm_prefix *pre, const double *v,
                        const sm_path *path, sm_motion *mo);
sm_status sm_ctx_start_run(sm_ctx *c, const double *v, uint32_t L, uint32_t tie, int32_t seed);
sm_status sm_ctx_next_pass(sm_ctx *c);   /* SM_E_CANCELLED once cancel is set */
sm_status sm_ctx_report(sm_ctx *c, uint32_t done, uint32_t total);   /* SM_E_CANCELLED if the callback says stop */
void      sm_param_defaults(const sm_model *m, double *v);
sm_status sm_check_text(const char *utf8, int32_t len, size_t *n, char *err, size_t err_len);
void      sm_text_copy(sm_text *dst, const sm_text *src);
size_t    sm_schema(const sm_model *m, char *buf, size_t cap);
sm_status sm_motion_fit(sm_motion *mo, uint32_t tokens);
/* everything else from mo->feat; v: the parameters, path may be NULL, tmp: SM_POST_TMP(T) */
void      sm_motion_post(sm_motion *mo, const sm_track *tr, const double *v, const sm_path *path, double *tmp);
typedef struct { char *p; size_t cap, n; } sm_buf;   /* writes while it fits, counts always */
void      sm_puts(sm_buf *b, const char *s);
void      sm_putn(sm_buf *b, double v);   /* C locale, at most 6 decimals */
#endif
