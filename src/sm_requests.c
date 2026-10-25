/* Parameters (one table), requests, text objects and the JSON schemas. */
#include "sm_internal.h"

/* id, name, int, unit, about, min, max, default, JSON separator. min and max are the runtime's limits (seconds: up to
   the model's max_frames / fps); a model file may narrow them and set its own defaults (sprongmotion.defaults.*). */
#define PARAMS(X)                                                                                                     \
    X(1, seconds, 0, "s", "clip length", 0.5, 10, 4, ",")                                                             \
    X(2, seed, 1, "", "same seed, same motion; variant i uses seed + i", 0, 2147483647, 0, ",")                       \
    X(3, steps, 1, "", "base-layer unmasking steps", 1, 64, 10, ",")                                                  \
    X(4, res_layers, 1, "", "residual code layers; 0 = a fast draft", 0, 5, 5, ",")                                   \
    X(5, cfg, 0, "", "text guidance of the base layer", 0, 20, 3, ",")                                                \
    X(6, res_cfg, 0, "", "text guidance of the residual layers", 0, 20, 3, ",")                                       \
    X(7, temp, 0, "", "sampling temperature", 0.01, 10, 1, ",")                                                       \
    X(8, top_k, 1, "", "sample from the k likeliest codes; 0 = all", 0, 512, 52, ",")                                 \
    X(9, foot_lock, 1,  "", "planted feet stay put and above the floor", 0, 1, 1, ",")                                \
    X(10, in_place, 1,  "", "joints and BVH without the horizontal root travel", 0, 1, 0, ",")                       \
    X(11, loop, 1,  "", "a seamless cycle: the clip's end leads into its start", 0, 1, 0, ",")                        \
    X(13, path_warp, 1, "", "with a path: the root lands on it", 0, 1, 1, "")

typedef struct { const char *name, *unit, *about; uint8_t is_int; double min, max, dflt; } pdef;
#define ROW(id, name, is_int, unit, about, lo, hi, d, sep) [id] = {#name, unit, about, is_int, lo, hi, d},
static const pdef P[SM_N_PARAMS] = {PARAMS(ROW)};

#define IO_SCHEMA                                                                                                     \
    "\"inputs\":[{\"name\":\"text\",\"type\":\"string\",\"max_bytes\":4096},"                                         \
    "{\"name\":\"text_vector\",\"type\":\"float[]\"},{\"name\":\"prefix\",\"type\":\"motion\"},"                      \
    "{\"name\":\"codes\",\"type\":\"int16[tokens][6]\"},"                                                             \
    "{\"name\":\"path\",\"type\":\"float[n][4]\",\"unit\":\"s, m, m, rad\"}],"                                        \
    "\"outputs\":[{\"name\":\"joints\",\"type\":\"float[frames][22][3]\",\"unit\":\"m\"},"                            \
    "{\"name\":\"rotations\",\"type\":\"float[frames][22][4]\",\"unit\":\"quaternion xyzw\"},"                        \
    "{\"name\":\"offsets\",\"type\":\"float[22][3]\",\"unit\":\"m\"},"                                                \
    "{\"name\":\"root\",\"type\":\"float[frames][3]\",\"unit\":\"m\"},"                                               \
    "{\"name\":\"speed\",\"type\":\"float[frames]\",\"unit\":\"m/s\"},"                                               \
    "{\"name\":\"path_error\",\"type\":\"float\",\"unit\":\"m\"},"                                                    \
    "{\"name\":\"contacts\",\"type\":\"uint8[frames][4]\"},{\"name\":\"features\",\"type\":\"float[frames][263]\"},"  \
    "{\"name\":\"codes\",\"type\":\"int16[tokens][6]\"},{\"name\":\"confidence\",\"type\":\"float[tokens]\"},"        \
    "{\"name\":\"bvh\",\"type\":\"string\"}],\"frame\":\"Y up, metres, starts at the origin facing +Z\"}"
#define TYPE_0 "float"
#define TYPE_1 "int"
#define JSON(id, name, is_int, unit, about, lo, hi, d, sep)                                                           \
    "{\"name\":\"" #name "\",\"type\":\"" TYPE_##is_int "\",\"unit\":\"" unit "\",\"about\":\"" about                 \
    "\",\"id\":" #id ",\"min\":" #lo ",\"max\":" #hi ",\"default\":" #d "}" sep
static const char API_SCHEMA[] = "{\"version\":\"1.0.0\",\"fps\":20,\"params\":[" PARAMS(JSON) "]," IO_SCHEMA;

static int known(sm_param p) { return p > 0 && p < SM_N_PARAMS && P[p].name; }

sm_status sm_bind_params(sm_model *m, char *err, size_t err_len) {
    static const char *const part[3] = {".min", ".max", ""};
    for (sm_param p = 1; p < SM_N_PARAMS; p++) {
        double *v = m->param[p], x;   /* min, max, default */
        char key[64] = "";
        v[0] = P[p].min;
        v[1] = p == SM_P_SECONDS ? (double)m->max_frames / m->fps : P[p].max;
        v[2] = P[p].dflt;
        for (int i = 0; i < 3 && known(p); i++) {
            snprintf(key, sizeof key, "sprongmotion.defaults.%s%s", P[p].name, part[i]);
            if (sm_gguf_num(&m->g, key, &x)) v[i] = i == 0 ? fmax(v[0], x) : i == 1 ? fmin(v[1], x) : x;
        }
        if (!(v[0] <= v[2] && v[2] <= v[1]) || (P[p].is_int && v[2] != floor(v[2])))
            return sm_fail_range(err, err_len, SM_E_FORMAT, key, v[2], v[0], v[1]);
    }
    return SM_OK;
}

sm_param sm_param_next(sm_param p) {
    for (p = p < 0 ? SM_N_PARAMS : p + 1; p < SM_N_PARAMS; p++)
        if (known(p)) return p;
    return -1;
}

sm_param sm_param_from_name(const char *name) {
    for (sm_param p = sm_param_next(0); name && p > 0; p = sm_param_next(p))
        if (!strcmp(P[p].name, name)) return p;
    return -1;
}

const char *sm_param_name(sm_param p) { return known(p) ? P[p].name : NULL; }

sm_status sm_param_bounds(const sm_model *m, sm_param p, double *min, double *max, double *dflt, int32_t *is_int) {
    if (!known(p)) return SM_E_ARG;
    const double *v = m ? m->param[p] : (const double[3]){P[p].min, P[p].max, P[p].dflt};
    if (min) *min = v[0];
    if (max) *max = v[1];
    if (dflt) *dflt = v[2];
    if (is_int) *is_int = P[p].is_int;
    return SM_OK;
}

static sm_status rfail(sm_request *r, sm_status s, const char *msg) { return sm_fail(r->err, sizeof r->err, s, msg); }

static sm_status set_value(sm_request *r, sm_param p, double v) {
    if (!r) return SM_E_ARG;
    r->err[0] = 0;
    if (!known(p)) return rfail(r, SM_E_ARG, "unknown parameter");
    const double *b = r->m->param[p];
    if (!(v >= b[0] && v <= b[1]) || (P[p].is_int && v != (double)(int64_t)v))
        return sm_fail_range(r->err, sizeof r->err, SM_E_ARG, P[p].name, v, b[0], b[1]);
    r->v[p] = v;
    return SM_OK;
}

sm_status sm_request_set_int(sm_request *r, sm_param p, int32_t v) { return set_value(r, p, v); }
sm_status sm_request_set_float(sm_request *r, sm_param p, double v) { return set_value(r, p, v); }

/* no message: a const request may be read by several threads */
static sm_status get_value(const sm_request *r, sm_param p, const void *v, int want_int) {
    return !r || !v || !known(p) || (want_int && !P[p].is_int) ? SM_E_ARG : SM_OK;
}

sm_status sm_request_get_int(const sm_request *r, sm_param p, int32_t *v) {
    const sm_status s = get_value(r, p, v, 1);
    if (s != SM_OK) return s;
    *v = (int32_t)r->v[p];
    return SM_OK;
}

sm_status sm_request_get_float(const sm_request *r, sm_param p, double *v) {
    const sm_status s = get_value(r, p, v, 0);
    if (s != SM_OK) return s;
    *v = r->v[p];
    return SM_OK;
}

/* [-+]digits[.digits], whatever the locale */
static int parse_number(const char *s, double *out) {
    double v = 0, scale = 1;
    int digits = 0, frac = 0;
    const int neg = *s == '-';
    for (s += *s == '-' || *s == '+'; (*s >= '0' && *s <= '9') || (*s == '.' && !frac); s++) {
        if (*s == '.') {
            frac = 1;
            continue;
        }
        v = v * 10 + (*s - '0');
        scale *= frac ? 10 : 1;
        digits++;
    }
    *out = (neg ? -v : v) / scale;
    return digits && !*s;
}

sm_status sm_request_set_by_name(sm_request *r, const char *name, const char *value) {
    double v;
    if (!r || !name || !value) return SM_E_ARG;
    if (sm_param_from_name(name) < 0) return rfail(r, SM_E_ARG, "unknown parameter name");
    if (!parse_number(value, &v)) return rfail(r, SM_E_ARG, "not a number");
    return set_value(r, sm_param_from_name(name), v);
}

void sm_param_defaults(const sm_model *m, double *v) {
    for (int p = 0; p < SM_N_PARAMS; p++) v[p] = m->param[p][2];
}

static void init(sm_request *r, const sm_model *m) {
    memset(r, 0, sizeof *r);
    r->m = m;
    sm_param_defaults(m, r->v);
}

sm_status sm_request_create(const sm_model *m, sm_request **out) {
    sm_request *r = m && out ? sm_alloc(&m->h, sizeof *r) : NULL;
    if (!r) return m && out ? SM_E_NOMEM : SM_E_ARG;
    init(r, m);
    *out = r;
    return SM_OK;
}

void sm_request_free(sm_request *r) {
    if (!r) return;
    sm_text_free(r->enc);
    sm_free(&r->m->h, r);
}

sm_status sm_request_reset(sm_request *r) {
    if (!r) return SM_E_ARG;
    sm_text_free(r->enc);
    init(r, r->m);
    return SM_OK;
}

const char *sm_request_last_error(const sm_request *r) { return r ? r->err : "request is NULL"; }

void sm_text_copy(sm_text *d, const sm_text *s) {
    memcpy(d->vec, s->vec, sizeof(float) * s->m->text.out.n);
    memcpy(d->words, s->words, sizeof(float) * s->W * s->m->text.d);
    d->W = s->W;
    d->ready = s->ready;
    d->n_unknown = s->n_unknown;
    memcpy(d->unknown, s->unknown, sizeof d->unknown);
}

sm_status sm_request_clone(const sm_request *r, sm_request **out) {
    if (!r || !out) return SM_E_ARG;
    sm_request *c = sm_alloc(&r->m->h, sizeof *c);
    if (!c) return SM_E_NOMEM;
    memcpy(c, r, sizeof *c);
    c->enc = NULL;
    if (r->enc && sm_text_create(r->m, &c->enc) != SM_OK) {
        sm_free(&r->m->h, c);
        return SM_E_NOMEM;
    }
    if (r->enc) sm_text_copy(c->enc, r->enc);
    *out = c;
    return SM_OK;
}

/* at most 4096 bytes of UTF-8; a NUL only as the terminator of a len -1 text */
sm_status sm_check_text(const char *utf8, int32_t len, size_t *n, char *err, size_t err_len) {
    if (!utf8) return sm_fail(err, err_len, SM_E_ARG, "text is NULL");
    *n = len < 0 ? strlen(utf8) : (size_t)len;
    if (*n > SM_MAX_TEXT) return sm_fail_range(err, err_len, SM_E_ARG, "text bytes", (double)*n, 0, SM_MAX_TEXT);
    return !memchr(utf8, 0, *n) && sm_utf8_valid(utf8, *n) ? SM_OK : sm_fail(err, err_len, SM_E_ARG, "text is not UTF-8");
}

sm_status sm_request_set_text(sm_request *r, const char *utf8, int32_t len) {
    if (!r) return SM_E_ARG;
    size_t n;
    if (sm_check_text(utf8, len, &n, r->err, sizeof r->err) != SM_OK) return SM_E_ARG;
    memcpy(r->text, utf8, n);
    r->text[n] = 0;
    r->text_len = (uint32_t)n;
    r->has_text = 1;
    r->err[0] = 0;
    return SM_OK;
}

sm_status sm_request_set_text_encoded(sm_request *r, const sm_text *t) {
    if (!r) return SM_E_ARG;
    if (!t || t->m != r->m || !t->ready) return rfail(r, SM_E_ARG, "text NULL, not encoded or from another model");
    if (!r->enc && sm_text_create(r->m, &r->enc) != SM_OK) return rfail(r, SM_E_NOMEM, "out of memory");
    sm_text_copy(r->enc, t);
    r->has_text = 2;
    r->err[0] = 0;
    return SM_OK;
}

sm_status sm_request_set_prefix(sm_request *r, const sm_motion *prev) {
    if (!r) return SM_E_ARG;
    r->err[0] = 0;
    if (prev && prev->m != r->m) return rfail(r, SM_E_ARG, "the prefix clip is from another model");
    if (prev && prev->tokens < SM_OVERLAP)
        return sm_fail_range(r->err, sizeof r->err, SM_E_ARG, "prefix tokens", prev->tokens, SM_OVERLAP, SM_MAX_CODES);
    memset(&r->pre, 0, sizeof r->pre);
    if (!prev) return SM_OK;
    r->pre.n = prev->tokens < SM_CONTEXT ? prev->tokens : SM_CONTEXT;
    memcpy(r->pre.codes, prev->codes + (prev->tokens - r->pre.n) * SM_VQ_LAYERS,
           sizeof(int16_t) * r->pre.n * SM_VQ_LAYERS);
    r->pre.track = prev->end;
    memcpy(r->pre.last, prev->feat + (size_t)(prev->frames - 1) * SM_F, sizeof r->pre.last);
    return SM_OK;
}

sm_status sm_request_set_codes(sm_request *r, const int16_t *codes, uint32_t n_tokens) {
    if (!r) return SM_E_ARG;
    r->err[0] = 0;
    if (!codes && n_tokens) return rfail(r, SM_E_ARG, "codes is NULL");
    if (n_tokens > SM_MAX_PINS)
        return sm_fail_range(r->err, sizeof r->err, SM_E_ARG, "n_tokens", n_tokens, 0, SM_MAX_PINS);
    for (size_t i = 0; i < (size_t)n_tokens * SM_VQ_LAYERS; i++)
        if (codes[i] < -1 || codes[i] >= SM_VQ_N)
            return sm_fail_range(r->err, sizeof r->err, SM_E_ARG, "a code", codes[i], -1, SM_VQ_N - 1);
    if (n_tokens) memcpy(r->pins, codes, sizeof(int16_t) * n_tokens * SM_VQ_LAYERS);
    r->n_pins = n_tokens;
    return SM_OK;
}

sm_status sm_text_create(const sm_model *m, sm_text **out) {
    const size_t floats = m ? m->text.out.n + (size_t)m->tok.max_len * m->text.d : 0;
    sm_text *t = m && out ? sm_alloc(&m->h, sizeof *t + sizeof(float) * floats) : NULL;
    if (!t) return m && out ? SM_E_NOMEM : SM_E_ARG;
    memset(t, 0, sizeof *t);
    t->m = m;
    t->vec = (float *)(t + 1);
    t->words = t->vec + m->text.out.n;
    *out = t;
    return SM_OK;
}

void sm_text_free(sm_text *t) {
    if (t) sm_free(&t->m->h, t);
}

sm_status sm_text_vector(const sm_text *t, const float **v, uint32_t *dim) {
    if (!t || !v) return SM_E_ARG;
    if (!t->ready) return SM_E_STATE;
    *v = t->vec;
    if (dim) *dim = t->m->text.out.n;
    return SM_OK;
}

sm_status sm_text_set_vector(sm_text *t, const float *v, uint32_t dim) {
    if (!t || !v || !dim || dim != t->m->text.out.n) return SM_E_ARG;
    for (uint32_t i = 0; i < dim; i++)
        if (!isfinite(v[i])) return SM_E_ARG;
    memcpy(t->vec, v, sizeof(float) * dim);
    t->W = t->n_unknown = 0;
    t->unknown[0] = 0;
    t->ready = 1;
    return SM_OK;
}

sm_status sm_text_unknown_words(const sm_text *t, char *buf, size_t cap, size_t *needed, uint32_t *n) {
    if (!t) return SM_E_ARG;
    if (!t->ready) return SM_E_STATE;
    const size_t need = strlen(t->unknown) + 1;
    if (needed) *needed = need;
    if (buf && cap < need) return SM_E_BUFFER;
    if (buf) memcpy(buf, t->unknown, need);
    if (n) *n = t->n_unknown;
    return SM_OK;
}

/* the API schema's layout, with the model's ranges and defaults */

size_t sm_schema(const sm_model *m, char *buf, size_t cap) {
    sm_buf b = {buf, cap, 0};
    sm_puts(&b, "{\"version\":\"1.0.0\",\"fps\":");
    sm_putn(&b, m->fps);
    sm_puts(&b, ",\"params\":[");
    for (sm_param p = sm_param_next(0); p > 0; p = sm_param_next(p)) {
        const char *keys[] = {"{\"name\":\"", P[p].name, "\",\"type\":\"", P[p].is_int ? "int" : "float",
                              "\",\"unit\":\"", P[p].unit, "\",\"about\":\"", P[p].about, "\",\"id\":"};
        for (size_t i = 0; i < sizeof keys / sizeof keys[0]; i++) sm_puts(&b, keys[i]);
        const double nums[] = {p, m->param[p][0], m->param[p][1], m->param[p][2]};
        const char *sep[] = {",\"min\":", ",\"max\":", ",\"default\":", sm_param_next(p) > 0 ? "}," : "}"};
        for (int i = 0; i < 4; i++) {
            sm_putn(&b, nums[i]);
            sm_puts(&b, sep[i]);
        }
    }
    sm_puts(&b, "]," IO_SCHEMA);
    if (buf && b.n < cap) buf[b.n] = 0;
    return b.n + 1;
}

const char *sm_api_schema_json(void) { return API_SCHEMA; }
const char *sm_model_schema_json(const sm_model *m) { return m ? m->schema : NULL; }
