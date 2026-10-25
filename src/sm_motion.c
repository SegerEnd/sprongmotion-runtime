/* The skeleton's names, the motion object and BVH text. */
#include "sm_internal.h"

static const char *const joint_names[SM_J] = {
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3", "l_foot",
    "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist",
    "r_wrist",
};

const char *sm_skeleton_joint_name(const sm_model *m, uint32_t j) { return m && j < SM_J ? joint_names[j] : NULL; }
int32_t sm_skeleton_parent(const sm_model *m, uint32_t j) { return m && j < SM_J ? sm_parent[j] : -1; }

sm_status sm_skeleton_offsets(const sm_model *m, float *out, uint32_t n_floats) {
    if (!m || !out) return SM_E_ARG;
    if (n_floats < SM_J * 3) return SM_E_BUFFER;
    memcpy(out, m->rest, sizeof(float) * SM_J * 3);
    return SM_OK;
}

#define MAX_TOKENS 6000   /* 10 minutes */

sm_status sm_motion_create(const sm_model *m, sm_motion **out) {
    sm_motion *mo = m && out ? sm_alloc(&m->h, sizeof *mo) : NULL;
    if (!mo) return m && out ? SM_E_NOMEM : SM_E_ARG;
    memset(mo, 0, sizeof *mo);
    mo->m = m;
    mo->seed = -1;
    *out = mo;
    return SM_OK;
}

void sm_motion_free(sm_motion *mo) {
    if (!mo) return;
    sm_free(&mo->m->h, mo->mem);
    sm_free(&mo->m->h, mo);
}

/* one block per capacity; the floats first, then codes, then contacts */
sm_status sm_motion_fit(sm_motion *mo, uint32_t tokens) {
    if (tokens <= mo->cap) return SM_OK;
    if (tokens > MAX_TOKENS) return SM_E_ARG;
    const size_t T = 2 * (size_t)tokens, floats = T * (SM_F + SM_J * 3 + SM_J * 4 + 4) + tokens;
    float *f = sm_alloc(&mo->m->h, sizeof(float) * floats + sizeof(int16_t) * tokens * SM_VQ_LAYERS + T * 4);
    if (!f) return SM_E_NOMEM;
    sm_free(&mo->m->h, mo->mem);
    mo->mem = mo->feat = f;
    mo->joints = mo->feat + T * SM_F;
    mo->rot = mo->joints + T * SM_J * 3;
    mo->root = mo->rot + T * SM_J * 4;
    mo->speed = mo->root + T * 3;
    mo->conf = mo->speed + T;
    mo->codes = (int16_t *)(mo->conf + tokens);
    mo->contacts = (uint8_t *)(mo->codes + (size_t)tokens * SM_VQ_LAYERS);
    mo->cap = tokens;
    mo->tokens = mo->frames = 0;
    return SM_OK;
}

sm_status sm_motion_reserve(sm_motion *mo, double max_seconds) {
    if (!mo || !(max_seconds > 0 && max_seconds <= MAX_TOKENS * 0.1)) return SM_E_ARG;
    const double per_token = (double)mo->m->frames_per_token / mo->m->fps;
    return sm_motion_fit(mo, (uint32_t)ceil(max_seconds / per_token - 1e-9));
}

sm_status sm_motion_copy(sm_motion *dst, const sm_motion *src) {
    if (!dst || !src || dst->m != src->m) return SM_E_ARG;
    if (dst == src) return SM_OK;
    const sm_status s = sm_motion_fit(dst, src->tokens);
    if (s != SM_OK) return s;
    const size_t T = src->frames, L = src->tokens;
    memcpy(dst->feat, src->feat, sizeof(float) * T * SM_F);
    memcpy(dst->joints, src->joints, sizeof(float) * T * SM_J * 3);
    memcpy(dst->rot, src->rot, sizeof(float) * T * SM_J * 4);
    memcpy(dst->root, src->root, sizeof(float) * T * 3);
    memcpy(dst->speed, src->speed, sizeof(float) * T);
    memcpy(dst->conf, src->conf, sizeof(float) * L);
    memcpy(dst->codes, src->codes, sizeof(int16_t) * L * SM_VQ_LAYERS);
    memcpy(dst->contacts, src->contacts, T * 4);
    memcpy(dst->off, src->off, sizeof dst->off);
    dst->tokens = src->tokens;
    dst->frames = src->frames;
    dst->seed = src->seed;
    dst->end = src->end;
    dst->path_err = src->path_err;
    return SM_OK;
}

double   sm_motion_seconds(const sm_motion *mo) { return mo ? (double)mo->frames / mo->m->fps : 0; }
uint32_t sm_motion_frames(const sm_motion *mo) { return mo ? mo->frames : 0; }
int32_t  sm_motion_seed(const sm_motion *mo) { return mo ? mo->seed : -1; }

static const void *view(const sm_motion *mo, const void *p, size_t per_frame, size_t per_token, uint32_t *n) {
    const size_t count = mo ? mo->frames * per_frame + mo->tokens * per_token : 0;
    if (n) *n = (uint32_t)count;
    return count ? p : NULL;
}

const float *sm_motion_joints(const sm_motion *mo, uint32_t *n) { return view(mo, mo ? mo->joints : 0, 66, 0, n); }
const float *sm_motion_rotations(const sm_motion *mo, uint32_t *n) { return view(mo, mo ? mo->rot : 0, 88, 0, n); }
const float *sm_motion_root(const sm_motion *mo, uint32_t *n) { return view(mo, mo ? mo->root : 0, 3, 0, n); }
const float *sm_motion_speed(const sm_motion *mo, uint32_t *n) { return view(mo, mo ? mo->speed : 0, 1, 0, n); }
float sm_motion_path_error(const sm_motion *mo) { return mo && mo->frames ? mo->path_err : -1; }
const uint8_t *sm_motion_contacts(const sm_motion *mo, uint32_t *n) { return view(mo, mo ? mo->contacts : 0, 4, 0, n); }
const float *sm_motion_features(const sm_motion *mo, uint32_t *n) { return view(mo, mo ? mo->feat : 0, SM_F, 0, n); }
const int16_t *sm_motion_codes(const sm_motion *mo, uint32_t *n) { return view(mo, mo ? mo->codes : 0, 0, 6, n); }
const float *sm_motion_confidence(const sm_motion *mo, uint32_t *n) { return view(mo, mo ? mo->conf : 0, 0, 1, n); }

const float *sm_motion_offsets(const sm_motion *mo, uint32_t *n) {
    if (n) *n = mo && mo->frames ? SM_J * 3 : 0;
    return mo && mo->frames ? mo->off : NULL;
}

static void remove_travel(sm_motion *mo) {
    for (size_t i = 0; i < (size_t)mo->frames * SM_J; i++) {
        mo->joints[3 * i] -= mo->root[3 * (i / SM_J)];
        mo->joints[3 * i + 2] -= mo->root[3 * (i / SM_J) + 2];
    }
}

void sm_motion_post(sm_motion *mo, const sm_track *tr, const double *v, const sm_path *path, double *tmp) {
    const uint32_t T = mo->frames;
    const double fps = mo->m->fps;
    double *yaw_added = tmp + 3 * (size_t)T, *smooth_root = yaw_added + T;
    sm_track track = *tr;
    sm_features_to_joints(mo->feat, T, &track, mo->joints);
    mo->end = track;

    for (size_t t = 0; t < T; t++)
        for (size_t k = 0; k < 4; k++) mo->contacts[4 * t + k] = mo->feat[t * SM_F + SM_F_CONTACT + k] > 0.5f;
    sm_fill_contacts(mo->joints, mo->contacts, T, fps);
    sm_root_smooth(mo->joints, T, fps, smooth_root, mo->speed);

    const int has_path = path && path->n, warp = has_path && v[SM_P_PATH_WARP] != 0;
    mo->path_err = -1;
    if (has_path)
        mo->path_err = (float)sm_path_warp(path, mo->joints, T, fps, smooth_root, warp, yaw_added,
                                           smooth_root + 2 * (size_t)T);
    for (size_t t = 0; t < T; t++) memcpy(mo->root + 3 * t, mo->joints + t * SM_J * 3, sizeof(float) * 3);

    if (v[SM_P_FOOT_LOCK] != 0) sm_foot_lock(mo->joints, mo->contacts, T, tmp, warp ? yaw_added : NULL);
    sm_rotations(mo->joints, T, mo->m->rest, mo->rot);
    sm_clip_offsets(mo->joints, mo->rot, T, tmp, mo->off);
    if (v[SM_P_IN_PLACE] != 0) remove_travel(mo);
}

void sm_puts(sm_buf *b, const char *s) {
    for (; *s; s++, b->n++)
        if (b->p && b->n < b->cap) b->p[b->n] = *s;
}

void sm_putn(sm_buf *b, double v) {   /* at most 6 decimals, no trailing zeros */
    const double a = fabs(v) * 1e6 + 0.5;
    const uint64_t u = a < 9e18 ? (uint64_t)a : 0;
    uint64_t f = u % 1000000, i = u / 1000000;
    char s[40], *p = s + sizeof s;
    *--p = 0;
    int digits = 6;
    while (digits && f % 10 == 0) {
        f /= 10;
        digits--;
    }
    for (; digits; digits--, f /= 10) *--p = (char)('0' + f % 10);
    if (*p) *--p = '.';
    do *--p = (char)('0' + i % 10);
    while (i /= 10);
    if (v < 0 && u) *--p = '-';
    sm_puts(b, p);
}

#define DEGREES 57.29577951308232   /* per radian */

static const uint8_t dfs[SM_J] = {0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21};

static void line(sm_buf *b, int depth, const char *s) {
    for (int i = 0; i < depth; i++) sm_puts(b, "  ");
    sm_puts(b, s);
}

static void hierarchy(const sm_motion *mo, double scale, sm_buf *b) {
    int depth[SM_J], open = 0;
    sm_puts(b, "HIERARCHY\n");
    for (int i = 0; i < SM_J; i++) {
        const int j = dfs[i], leaf = i + 1 == SM_J || sm_parent[dfs[i + 1]] != j;
        depth[j] = j ? depth[sm_parent[j]] + 1 : 0;
        for (; open > depth[j]; open--) line(b, open - 1, "}\n");
        line(b, depth[j], j ? "JOINT " : "ROOT ");
        sm_puts(b, joint_names[j]);
        sm_puts(b, "\n");
        line(b, depth[j], "{\n");
        line(b, depth[j] + 1, "OFFSET");
        for (int k = 0; k < 3; k++) {
            sm_puts(b, " ");
            sm_putn(b, mo->off[3 * j + k] * scale);
        }
        sm_puts(b, j ? "\n" : "\n  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n");
        if (j) line(b, depth[j] + 1, "CHANNELS 3 Zrotation Yrotation Xrotation\n");
        const char *end[4] = {"End Site\n", "{\n", "  OFFSET 0 0 0\n", "}\n"};
        for (int k = 0; leaf && k < 4; k++) line(b, depth[j] + 1, end[k]);
        open = depth[j] + 1;
    }
    for (; open > 0; open--) line(b, open - 1, "}\n");
}

/* R = Rz(a) Ry(b) Rx(c) from x y z w, degrees; each angle the nearest turn to the previous frame's */
static void euler_zyx(const float *q, double *e) {
    const double x = q[0], y = q[1], z = q[2], w = q[3], r20 = 2 * (x * z - y * w);
    const double a = atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z));
    const double b = asin(r20 > 1 ? -1 : r20 < -1 ? 1 : -r20), c = atan2(2 * (y * z + x * w), 1 - 2 * (x * x + y * y));
    const double next[3] = {a * DEGREES, b * DEGREES, c * DEGREES};
    for (int k = 0; k < 3; k++) e[k] = next[k] + 360 * round((e[k] - next[k]) / 360);
}

static void frames(const sm_motion *mo, double scale, sm_buf *b) {
    double angles[SM_J][3] = {{0}};
    sm_puts(b, "MOTION\nFrames: ");
    sm_putn(b, mo->frames);
    sm_puts(b, "\nFrame Time: ");
    sm_putn(b, 1.0 / mo->m->fps);
    for (size_t t = 0; t < mo->frames; t++) {
        sm_puts(b, "\n");
        for (int k = 0; k < 3; k++) {
            sm_putn(b, mo->joints[t * SM_J * 3 + (size_t)k] * scale);
            sm_puts(b, " ");
        }
        for (int i = 0; i < SM_J; i++) {
            const int j = dfs[i];
            euler_zyx(mo->rot + (t * SM_J + (size_t)j) * 4, angles[j]);
            for (int k = 0; k < 3; k++) {
                sm_putn(b, angles[j][k]);
                sm_puts(b, i + 1 < SM_J || k < 2 ? " " : "");
            }
        }
    }
    sm_puts(b, "\n");
}

sm_status sm_motion_write_bvh(const sm_motion *mo, double scale, char *buf, size_t cap, size_t *needed) {
    if (!mo || !(scale > 0 && scale <= 1e4) || (!buf && !needed)) return SM_E_ARG;
    if (!mo->frames) return SM_E_STATE;
    sm_buf b = {NULL, 0, 0};
    for (int pass = 0; pass < 2; pass++) {
        hierarchy(mo, scale, &b);
        frames(mo, scale, &b);
        if (!pass && needed) *needed = b.n + 1;
        if (!pass && (!buf || cap < b.n + 1)) return buf ? SM_E_BUFFER : SM_OK;
        if (!pass) b = (sm_buf){buf, cap, 0};
    }
    buf[b.n] = 0;
    return SM_OK;
}
