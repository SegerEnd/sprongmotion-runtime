/* Matrix multiply on F32 or Q8_0 weights (bias, GELU or residual add fused into the store), transformer
   encoder layers as torch.nn.TransformerEncoderLayer (eval, GELU), and the motion VQ decoder. */
#include "sm_internal.h"

#define KC 512   /* K slice: a 32 KB panel that stays in L1 */
#define NP 16    /* panel width */
#define TR 4     /* tile rows: 16 accumulators, enough for 4 FMA pipes of latency 4 */
#define INLINE static inline __attribute__((always_inline))

float sm_half(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    if (!e) {
        const float f = (float)m * 0x1p-24f;   /* subnormal, exact in f32 */
        return s ? -f : f;
    }
    const uint32_t b = s | (e == 31 ? 0x7F800000u : (e + 112) << 23) | m << 13;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

typedef int8_t i8x16 __attribute__((vector_size(16), aligned(1)));

/* k-chunk [k, k + 4) of 4 rows -> 4 vectors over the rows: a 4 x 4 transpose */
static void transpose_store(const v4 r[4], float *buf) {
    const v4 t0 = __builtin_shufflevector(r[0], r[1], 0, 4, 1, 5), t1 = __builtin_shufflevector(r[2], r[3], 0, 4, 1, 5);
    const v4 t2 = __builtin_shufflevector(r[0], r[1], 2, 6, 3, 7), t3 = __builtin_shufflevector(r[2], r[3], 2, 6, 3, 7);
    sm_store4(buf, __builtin_shufflevector(t0, t1, 0, 1, 4, 5));
    sm_store4(buf + NP, __builtin_shufflevector(t0, t1, 2, 3, 6, 7));
    sm_store4(buf + 2 * NP, __builtin_shufflevector(t2, t3, 0, 1, 4, 5));
    sm_store4(buf + 3 * NP, __builtin_shufflevector(t2, t3, 2, 3, 6, 7));
}

/* 16 consecutive weights of a row, dequantized: 4 vectors */
static void row16(const sm_lin *l, uint32_t n, uint32_t k, v4 out[4]) {
    if (l->type == SM_T_F32) {
        const float *w = (const float *)l->w + (size_t)n * l->k + k;
        for (int i = 0; i < 4; i++) out[i] = sm_load4(w + 4 * i);
        return;
    }
    const uint8_t *blk = (const uint8_t *)l->w + ((size_t)n * (l->k / 32) + k / 32) * SM_Q8_BLOCK;
    const float d = sm_half((uint16_t)(blk[0] | blk[1] << 8));
    i8x16 q;
    memcpy(&q, blk + 2 + k % 32, 16);
    out[0] = d * __builtin_convertvector(__builtin_shufflevector(q, q, 0, 1, 2, 3), v4);
    out[1] = d * __builtin_convertvector(__builtin_shufflevector(q, q, 4, 5, 6, 7), v4);
    out[2] = d * __builtin_convertvector(__builtin_shufflevector(q, q, 8, 9, 10, 11), v4);
    out[3] = d * __builtin_convertvector(__builtin_shufflevector(q, q, 12, 13, 14, 15), v4);
}

void sm_row(const sm_lin *t, uint32_t i, float *out) {
    for (uint32_t k = 0; k < t->k; k += 16) {
        v4 r[4];
        row16(t, i, k, r);
        for (int c = 0; c < 4; c++) sm_store4(out + k + 4 * c, r[c]);
    }
}

/* buf[k][j] = W[n0 + j][k0 + k] */
static void expand(const sm_lin *l, uint32_t n0, uint32_t k0, uint32_t kc, float *buf) {
    for (uint32_t j = 0; j < NP; j += 4)
        for (uint32_t k = 0; k < kc; k += 16) {
            v4 r[4][4], t[4];
            for (uint32_t i = 0; i < 4; i++) row16(l, n0 + j + i, k0 + k, r[i]);
            for (int c = 0; c < 4; c++) {
                for (int i = 0; i < 4; i++) t[i] = r[i][c];
                transpose_store(t, buf + (k + 4 * (uint32_t)c) * NP + j);
            }
        }
}

/* y[R][16] (+)= x[R][kc] buf; R is a constant at each call, so the accumulators stay in registers. Each output sums
   in k order, as the naive loop. */
INLINE void tile(int R, const float *x, uint32_t K, const float *buf, uint32_t kc, float *y, uint32_t N, int more,
                 const float *b) {
    v4 a[TR][4];
    for (int r = 0; r < R; r++)
        for (int c = 0; c < 4; c++) a[r][c] = more ? sm_load4(y + (size_t)r * N + 4 * c) : (v4){0};
    for (uint32_t k = 0; k < kc; k++) {
        const float *wk = buf + k * NP;
        const v4 w[4] = {sm_load4(wk), sm_load4(wk + 4), sm_load4(wk + 8), sm_load4(wk + 12)};
        for (int r = 0; r < R; r++)
            for (int c = 0; c < 4; c++) a[r][c] += x[(size_t)r * K + k] * w[c];
    }
    for (int r = 0; r < R; r++)
        for (int c = 0; c < 4; c++) sm_store4(y + (size_t)r * N + 4 * c, b ? a[r][c] + sm_load4(b + 4 * c) : a[r][c]);
}

static void epilogue(const sm_lin *l, uint32_t M, uint32_t n0, float *Y, sm_epi epi, const float *add) {
    for (uint32_t m = 0; m < M; m++) {
        float *y = Y + (size_t)m * l->n + n0;
        if (epi == SM_EPI_GELU) {
            for (uint32_t j = 0; j < NP; j++) y[j] = sm_gelu(y[j]);
            continue;
        }
        const float *r = add + (size_t)m * l->n + n0;
        for (uint32_t j = 0; j < NP; j++) y[j] += r[j];
    }
}

typedef struct { const sm_lin *l; const float *X; uint32_t M; float *Y; sm_epi epi; const float *add; } job;

/* one task: output columns [16 i, 16 i + 16) of every row, all of K */
static void panel(void *arg, uint32_t i) {
    const job *j = arg;
    const sm_lin *l = j->l;
    const uint32_t K = l->k, N = l->n, M = j->M, n0 = i * NP;
    float buf[KC * NP];
    for (uint32_t k0 = 0; k0 < K; k0 += KC) {
        const uint32_t kc = K - k0 < KC ? K - k0 : KC;
        const float *b = l->b && k0 + kc == K ? l->b + n0 : NULL, *x = j->X + k0;
        float *y = j->Y + n0;
        expand(l, n0, k0, kc, buf);
        uint32_t m = 0;
        for (; m + TR <= M; m += TR) tile(TR, x + (size_t)m * K, K, buf, kc, y + (size_t)m * N, N, k0 > 0, b);
        for (; m < M; m++) tile(1, x + (size_t)m * K, K, buf, kc, y + (size_t)m * N, N, k0 > 0, b);
    }
    if (j->epi != SM_EPI_NONE) epilogue(l, M, n0, j->Y, j->epi, j->add);
}

void sm_linear(const sm_par *par, const sm_lin *l, const float *X, uint32_t M, float *Y, sm_epi epi, const float *add) {
    job j = {l, X, M, Y, epi, add};
    sm_par_for(par, l->n / NP, panel, &j);
}

#define LN_ROWS 16   /* rows per task */

typedef struct { const float *x; uint32_t M, d; sm_norm n; double eps; float *y; } ln_job;

/* biased variance; statistics in double, summed in 4 interleaved partial sums */
static void ln_rows(void *arg, uint32_t task) {
    const ln_job *j = arg;
    const uint32_t d = j->d, end = (task + 1) * LN_ROWS < j->M ? (task + 1) * LN_ROWS : j->M;
    for (uint32_t m = task * LN_ROWS; m < end; m++) {
        const float *x = j->x + (size_t)m * d;
        double s[4] = {0}, v[4] = {0};
        for (uint32_t i = 0; i < d; i++) s[i % 4] += x[i];
        const double mean = ((s[0] + s[1]) + (s[2] + s[3])) / d;
        for (uint32_t i = 0; i < d; i++) v[i % 4] += (x[i] - mean) * (x[i] - mean);
        const double inv = 1.0 / sqrt(((v[0] + v[1]) + (v[2] + v[3])) / d + j->eps);
        float *y = j->y + (size_t)m * d;
        for (uint32_t i = 0; i < d; i++) y[i] = (float)((x[i] - mean) * inv) * j->n.w[i] + j->n.b[i];
    }
}

void sm_layernorm(const sm_par *par, const float *x, uint32_t M, uint32_t d, sm_norm n, double eps, float *y) {
    ln_job j = {x, M, d, n, eps, y};
    sm_par_for(par, (M + LN_ROWS - 1) / LN_ROWS, ln_rows, &j);
}

#define QB 4   /* query rows per block */

typedef struct { const float *q, *k, *v; uint32_t S, w, hd; } head;

/* p[r][j] = q_r . k_j * scale for R query rows and J keys (constants at each call). Each dot sums 4 lanes over the
   head dimension, then the lanes in a fixed order. */
INLINE void qk(int R, int J, const head *h, uint32_t i, uint32_t j, float scale, float *p, uint32_t ps) {
    v4 a[QB][4] = {{{0}}};
    for (uint32_t c = 0; c < h->hd; c += 4) {
        v4 kv[4];
        for (int t = 0; t < J; t++) kv[t] = sm_load4(h->k + (size_t)(j + (uint32_t)t) * h->w + c);
        for (int r = 0; r < R; r++) {
            const v4 qv = sm_load4(h->q + (size_t)(i + (uint32_t)r) * h->w + c);
            for (int t = 0; t < J; t++) a[r][t] += qv * kv[t];
        }
    }
    for (int r = 0; r < R; r++)
        for (int t = 0; t < J; t++) p[(size_t)r * ps + j + (uint32_t)t] = sm_sum4(a[r][t]) * scale;
}

/* in place: p = exp(p - max) zero-padded to a multiple of 4; returns 1 / sum */
static float softmax_row(float *p, uint32_t S) {
    float mx = -INFINITY;
    for (uint32_t j = 0; j < S; j++) mx = p[j] > mx ? p[j] : mx;
    for (uint32_t j = 0; j < S; j++) p[j] = sm_expf(p[j] - mx);
    for (uint32_t j = S; j % 4; j++) p[j] = 0;
    v4 sum = {0};
    for (uint32_t j = 0; j < S; j += 4) sum += sm_load4(p + j);
    return 1 / sm_sum4(sum);
}

/* o_r = inv_r sum_j p[r][j] v_j for R query rows, 8 columns at a time, j in order */
INLINE void pv(int R, const head *h, const float *p, uint32_t ps, const float *inv, float *o, uint32_t od) {
    for (uint32_t c = 0; c < h->hd; c += 8) {
        v4 a[QB][2] = {{{0}}};
        for (uint32_t j = 0; j < h->S; j++) {
            const v4 v0 = sm_load4(h->v + (size_t)j * h->w + c), v1 = sm_load4(h->v + (size_t)j * h->w + c + 4);
            for (int r = 0; r < R; r++) {
                a[r][0] += p[(size_t)r * ps + j] * v0;
                a[r][1] += p[(size_t)r * ps + j] * v1;
            }
        }
        for (int r = 0; r < R; r++) {
            sm_store4(o + (size_t)r * od + c, a[r][0] * inv[r]);
            sm_store4(o + (size_t)r * od + c + 4, a[r][1] * inv[r]);
        }
    }
}

/* softmax(q k^T / sqrt(hd)) v for query rows [i, i + R) */
INLINE void rows(int R, const head *h, uint32_t i, float *p, uint32_t ps, float *o, uint32_t od) {
    const float scale = 1.0f / sqrtf((float)h->hd);
    float inv[QB];
    uint32_t j = 0;
    for (; j + 4 <= h->S; j += 4) qk(R, 4, h, i, j, scale, p, ps);
    for (; j < h->S; j++) qk(R, 1, h, i, j, scale, p, ps);
    for (int r = 0; r < R; r++) inv[r] = softmax_row(p + (size_t)r * ps, h->S);
    pv(R, h, p, ps, inv, o, od);
}

typedef struct { const float *qkv; const sm_batch *b; uint32_t d, heads, stride; float *out, *p; } attn;

/* one task: one head of one sequence; qkv rows hold q, k, v of all heads */
static void attend(void *arg, uint32_t task) {
    const attn *a = arg;
    const sm_seg seg = a->b->seg[task / a->heads];
    const uint32_t d = a->d, hd = d / a->heads, w = 3 * d, ps = a->stride;
    const float *q = a->qkv + (size_t)seg.start * w + task % a->heads * hd;
    const head h = {q, q + d, q + 2 * d, seg.len, w, hd};
    float *p = a->p + (size_t)task * QB * ps, *o = a->out + (size_t)seg.start * d + task % a->heads * hd;
    uint32_t i = 0;
    for (; i + QB <= seg.len; i += QB) rows(QB, &h, i, p, ps, o + (size_t)i * d, d);
    for (; i < seg.len; i++) rows(1, &h, i, p, ps, o + (size_t)i * d, d);
}

static void attention(const sm_par *par, const float *qkv, const sm_batch *b, uint32_t d, uint32_t heads, float *out,
                      float *p) {
    attn a = {qkv, b, d, heads, b->rows + 4, out, p};
    sm_par_for(par, b->n_seg * heads, attend, &a);
}

static uint32_t widest(uint32_t d, uint32_t ff) { return 3 * d > ff ? 3 * d : ff; }

size_t sm_layer_scratch_floats(uint32_t M, uint32_t d, uint32_t ff, uint32_t heads) {
    return (size_t)M * widest(d, ff) + (size_t)2 * M * d + (size_t)SM_MAX_SEGS * heads * QB * (M + 4);
}

/* scratch: big [M][max(3d, ff)] (qkv, then ff1), a and h [M][d], then the attention rows */
typedef struct { float *big, *a, *h, *p; } bufs;
static bufs carve(const sm_layer *l, uint32_t M, float *scratch) {
    const uint32_t d = l->out.n;
    float *a = scratch + (size_t)M * widest(d, l->ff1.n);
    return (bufs){scratch, a, a + (size_t)M * d, a + (size_t)2 * M * d};
}

void sm_layer_post(const sm_par *par, const sm_layer *l, const sm_batch *b, uint32_t heads, float *x, float *scratch) {
    const uint32_t M = b->rows, d = l->out.n;
    const bufs s = carve(l, M, scratch);
    sm_linear(par, &l->qkv, x, M, s.big, SM_EPI_NONE, NULL);
    attention(par, s.big, b, d, heads, s.a, s.p);
    sm_linear(par, &l->out, s.a, M, s.h, SM_EPI_ADD, x);
    sm_layernorm(par, s.h, M, d, l->ln1, 1e-5, x);
    sm_linear(par, &l->ff1, x, M, s.big, SM_EPI_GELU, NULL);
    sm_linear(par, &l->ff2, s.big, M, s.h, SM_EPI_ADD, x);
    sm_layernorm(par, s.h, M, d, l->ln2, 1e-5, x);
}

void sm_layer_pre(const sm_par *par, const sm_layer *l, const sm_batch *b, uint32_t heads, float *x, float *scratch) {
    const uint32_t M = b->rows, d = l->out.n;
    const bufs s = carve(l, M, scratch);
    sm_layernorm(par, x, M, d, l->ln1, 1e-5, s.a);
    sm_linear(par, &l->qkv, s.a, M, s.big, SM_EPI_NONE, NULL);
    attention(par, s.big, b, d, heads, s.a, s.p);
    sm_linear(par, &l->out, s.a, M, s.h, SM_EPI_ADD, x);
    sm_layernorm(par, s.h, M, d, l->ln2, 1e-5, s.a);
    sm_linear(par, &l->ff1, s.a, M, s.big, SM_EPI_GELU, NULL);
    sm_linear(par, &l->ff2, s.big, M, x, SM_EPI_ADD, s.h);
}

#define CONV_ROWS 8   /* output frames per task */

typedef struct {
    const float *x; uint32_t T, C; sm_conv w; uint32_t O, K, dil, stride, To; int relu_in; float *y;
} conv_job;

static void conv_rows(void *arg, uint32_t task) {
    const conv_job *j = arg;
    const int64_t pad = (int64_t)((j->K - 1) * j->dil / 2);
    const uint32_t C = j->C, O = j->O, end = (task + 1) * CONV_ROWS < j->To ? (task + 1) * CONV_ROWS : j->To;
    for (uint32_t t = task * CONV_ROWS; t < end; t++) {
        float *yt = j->y + (size_t)t * O;
        memcpy(yt, j->w.b, sizeof(float) * O);
        for (uint32_t k = 0; k < j->K; k++) {
            const int64_t s = (int64_t)t * j->stride + (int64_t)(k * j->dil) - pad;
            if (s < 0 || s >= (int64_t)j->T) continue;
            const float *xs = j->x + (size_t)s * C, *wk = j->w.w + (size_t)k * C * O;
            for (uint32_t c = 0; c < C; c++) {
                const float v = j->relu_in && xs[c] < 0 ? 0 : xs[c];
                if (v == 0) continue;   /* exact, and ReLU outputs are often zero */
                const float *wc = wk + (size_t)c * O;
                for (uint32_t o = 0; o < O; o++) yt[o] += v * wc[o];
            }
        }
    }
}

/* y[t][o] = b[o] + sum_k sum_c act(x[t stride + k dil - pad][c]) w[k][c][o], zero padding pad = (K - 1) dil / 2
   (PyTorch's padding 1 for the stride-2 K = 4 conv); act = ReLU if relu_in. Returns the output length. */
static uint32_t conv(const sm_par *par, const float *x, uint32_t T, uint32_t C, sm_conv w, uint32_t O, uint32_t K,
                     uint32_t dil, uint32_t stride, int relu_in, float *y) {
    const int64_t pad = (int64_t)((K - 1) * dil / 2);
    const uint32_t To = (uint32_t)(((int64_t)T + 2 * pad - (int64_t)(dil * (K - 1)) - 1) / stride + 1);
    conv_job j = {x, T, C, w, O, K, dil, stride, To, relu_in, y};
    sm_par_for(par, (To + CONV_ROWS - 1) / CONV_ROWS, conv_rows, &j);
    return To;
}

static const uint32_t dil[3] = {9, 3, 1};   /* Resnet1D with reverse_dilation: 3^2, 3^1, 3^0 */

/* b += conv2(relu(conv1(relu(b)))) per block; a, c: scratch of b's size */
static void res_stack(const sm_par *par, const sm_conv *c1, const sm_conv *c2, float *b, uint32_t L, float *a,
                      float *c) {
    const size_t n = (size_t)L * SM_VQ_W;
    for (int r = 0; r < 3; r++) {
        conv(par, b, L, SM_VQ_W, c1[r], SM_VQ_W, 3, dil[r], 1, 1, a);
        conv(par, a, L, SM_VQ_W, c2[r], SM_VQ_W, 1, 1, 1, 1, c);
        for (size_t i = 0; i < n; i++) b[i] += c[i];
    }
}

/* the VQ decoder: summed code vectors, then a conv1d stack; activations are [T][C] */
size_t sm_vq_scratch_floats(uint32_t L) { return (size_t)3 * 2 * L * SM_VQ_W; }

void sm_vq_decode(const sm_vq *v, const int16_t *codes, uint32_t L, float *feat, float *scratch) {
    const uint32_t W = SM_VQ_W, T = 2 * L;
    const size_t n = (size_t)L * W;
    float *a = scratch, *b = a + (size_t)T * W, *c = b + (size_t)T * W;

    memset(a, 0, sizeof(float) * n);   /* code -1: layer left out */
    for (uint32_t t = 0; t < L; t++)
        for (uint32_t q = 0; q < SM_VQ_LAYERS; q++) {
            const int16_t id = codes[t * SM_VQ_LAYERS + q];
            if (id < 0) continue;
            const float *e = v->codebook + ((size_t)q * SM_VQ_N + (size_t)(uint16_t)id) * W;
            for (uint32_t i = 0; i < W; i++) a[(size_t)t * W + i] += e[i];
        }
    conv(v->par, a, L, W, v->in, W, 3, 1, 1, 0, b);   /* the residual stream starts after this ReLU */
    for (size_t i = 0; i < n; i++) b[i] = b[i] < 0 ? 0 : b[i];
    res_stack(v->par, v->res1, v->res2, b, L, a, c);
    for (uint32_t t = 0; t < T; t++)   /* nearest 2x upsampling */
        memcpy(a + (size_t)t * W, b + (size_t)(t / 2) * W, sizeof(float) * W);
    conv(v->par, a, T, W, v->up, W, 3, 1, 1, 0, b);
    conv(v->par, b, T, W, v->mid, W, 3, 1, 1, 0, a);
    conv(v->par, a, T, W, v->out, SM_F, 3, 1, 1, 1, feat);   /* ReLU on its input (after model.3) */
    for (uint32_t t = 0; t < T; t++) {
        float *ft = feat + (size_t)t * SM_F;
        for (uint32_t i = 0; i < SM_F; i++) ft[i] = ft[i] * v->std[i] + v->mean[i];
    }
}

size_t sm_vq_enc_scratch_floats(uint32_t T) { return (size_t)T * (SM_F + SM_VQ_W) + (size_t)2 * (T / 2) * SM_VQ_W; }

/* out(res(down(relu(in((feat - mean) / std))))), then residual quantization: the squared distance in double, the
   first of equal ones (torch argmin), the residual in float as torch subtracts it */
void sm_vq_encode(const sm_vq *v, const float *feat, uint32_t T, int16_t *codes, float *scratch) {
    const sm_vq_enc *e = &v->enc;
    const uint32_t W = SM_VQ_W, L = T / 2;
    float *x = scratch, *a = x + (size_t)T * SM_F, *b = a + (size_t)T * W, *c = b + (size_t)L * W;
    for (size_t t = 0; t < T; t++)
        for (size_t i = 0; i < SM_F; i++) x[t * SM_F + i] = (feat[t * SM_F + i] - v->mean[i]) / v->std[i];
    conv(v->par, x, T, SM_F, e->in, W, 3, 1, 1, 0, a);
    conv(v->par, a, T, W, e->down, W, 4, 1, 2, 1, b);
    res_stack(v->par, e->res1, e->res2, b, L, a, c);
    conv(v->par, b, L, W, e->out, W, 3, 1, 1, 0, c);
    for (uint32_t t = 0; t < L; t++) {
        float *r = c + (size_t)t * W;
        for (uint32_t q = 0; q < SM_VQ_LAYERS; q++) {
            const float *cb = v->codebook + (size_t)q * SM_VQ_N * W;
            double best = INFINITY;
            uint32_t pick = 0;
            for (uint32_t k = 0; k < SM_VQ_N; k++) {
                const float *ck = cb + (size_t)k * W;
                double d2 = 0;
                for (uint32_t i = 0; i < W; i++) d2 += ((double)r[i] - ck[i]) * ((double)r[i] - ck[i]);
                if (d2 < best) best = d2, pick = k;
            }
            for (uint32_t i = 0; i < W; i++) r[i] -= cb[(size_t)pick * W + i];
            codes[t * SM_VQ_LAYERS + q] = (int16_t)pick;
        }
    }
}
