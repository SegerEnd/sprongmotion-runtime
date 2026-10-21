/* The skeleton, features to joints, joint rotations, foot locking, the motion object and BVH text. */
#include "sm_internal.h"
#include <stdlib.h>

const int8_t sm_parent[SM_J] = {-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19};
static const int8_t child[SM_J] = {-1, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, 15, 16, 17, -1, 18, 19, 20, 21, -1, -1};
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

sm_quat sm_qmul(sm_quat a, sm_quat b) {
    return (sm_quat){a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                     a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

static void cross(const double a[3], const double b[3], double o[3]) {
    o[0] = a[1] * b[2] - a[2] * b[1], o[1] = a[2] * b[0] - a[0] * b[2], o[2] = a[0] * b[1] - a[1] * b[0];
}

static double norm2(const double v[3]) { return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]; }

static void unit(double v[3]) {
    const double n = sqrt(norm2(v));
    for (int i = 0; i < 3; i++) v[i] /= n;
}

void sm_qrot(sm_quat q, const double v[3], double out[3]) {  /* v + 2 (w (u x v) + u x (u x v)), u = q.xyz */
    const double u[3] = {q.x, q.y, q.z};
    double uv[3], uuv[3];
    cross(u, v, uv), cross(u, uv, uuv);
    for (int i = 0; i < 3; i++) out[i] = v[i] + 2 * (q.w * uv[i] + uuv[i]);
}

sm_quat sm_qbetween(const double u[3], const double v[3]) {
    double c[3], w = 1 + u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    cross(u, v, c);
    if (w < 1e-9) {   /* opposite: 180 degrees about any axis perpendicular to u */
        const double x[3] = {1, 0, 0}, z[3] = {0, 0, 1};
        cross(u, x, c);
        if (norm2(c) < 1e-12) cross(u, z, c);
        w = 0;
    }
    const double n = sqrt(w * w + c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    return (sm_quat){w / n, c[0] / n, c[1] / n, c[2] / n};
}

/* recover_from_ric: yaw = cumsum of rot_vel shifted by one frame; root xz = cumsum of the velocities turned by the
   inverse yaw; local joint positions turned by the inverse yaw and moved by the root xz */
void sm_features_to_joints(const float *f, uint32_t T, sm_track *tr, float *joints) {
    double yaw = tr->yaw, px = tr->x, pz = tr->z;
    for (uint32_t t = 0; t < T; t++) {
        const float *ft = f + (size_t)t * SM_F, *prev = t ? ft - SM_F : tr->moving ? tr->vel : NULL;
        float *jt = joints + (size_t)t * SM_J * 3;
        if (prev) yaw += prev[0];
        const sm_quat qi = {cos(yaw), 0, -sin(yaw), 0};
        double r[3];
        if (prev) {
            const double vel[3] = {prev[1], 0, prev[2]};
            sm_qrot(qi, vel, r);
            px += r[0], pz += r[2];
        }
        jt[0] = (float)px, jt[1] = ft[3], jt[2] = (float)pz;
        for (int j = 1; j < SM_J; j++) {
            const double p[3] = {ft[3 * j + 1], ft[3 * j + 2], ft[3 * j + 3]};
            sm_qrot(qi, p, r);
            jt[3 * j] = (float)(r[0] + px), jt[3 * j + 1] = (float)r[1], jt[3 * j + 2] = (float)(r[2] + pz);
        }
    }
    if (!T) return;
    *tr = (sm_track){yaw, px, pz, {0}, 1};
    memcpy(tr->vel, f + (size_t)(T - 1) * SM_F, sizeof tr->vel);
}

static void dif(const float *j, int a, int b, double o[3]) {
    for (int k = 0; k < 3; k++) o[k] = (double)j[3 * a + k] - j[3 * b + k];
}

/* turns a0 onto a1 exactly, then about a1 to bring b0 closest to b1 (TRIAD) */
static sm_quat turn(double a0[3], const double b0[3], double a1[3], const double b1[3]) {
    double b[3], p0[3], p1[3];
    unit(a0), unit(a1);
    const sm_quat q = sm_qbetween(a0, a1);
    sm_qrot(q, b0, b);
    const double d0 = b[0] * a1[0] + b[1] * a1[1] + b[2] * a1[2], d1 = b1[0] * a1[0] + b1[1] * a1[1] + b1[2] * a1[2];
    for (int k = 0; k < 3; k++) p0[k] = b[k] - d0 * a1[k], p1[k] = b1[k] - d1 * a1[k];
    if (norm2(p0) < 1e-18 || norm2(p1) < 1e-18) return q;
    unit(p0), unit(p1);
    return sm_qmul(sm_qbetween(p0, p1), q);
}

/* global rotation S of each joint, which turns its rest offsets towards its children: the pelvis from the hip line and
   spine1, the chest from the neck and the shoulder line, every other joint the shortest arc from its parent's
   (the usual per-bone IK), a leaf its parent's. Out: S relative to the parent's, x y z w. */
void sm_rotations(const float *joints, uint32_t T, const float *rest, float *xyzw) {
    for (size_t t = 0; t < T; t++) {
        const float *j = joints + t * SM_J * 3;
        sm_quat S[SM_J];
        for (int i = 0; i < SM_J; i++) {
            const sm_quat P = i ? S[sm_parent[i]] : (sm_quat){1, 0, 0, 0};
            double a0[3], b0[3], a1[3], b1[3];
            if (i == 0) {
                dif(rest, 1, 2, a0), dif(j, 1, 2, a1), dif(rest, 3, 0, b0), dif(j, 3, 0, b1);
                S[i] = turn(a0, b0, a1, b1);
            } else if (i == 9) {   /* the shoulder line: the collar line (15 cm) jitters the chest's twist */
                dif(rest, 12, 0, a0), dif(j, 12, 9, a1), dif(j, 16, 17, b1);
                for (int k = 0; k < 3; k++) b0[k] = (double)rest[39 + k] + rest[48 + k] - rest[42 + k] - rest[51 + k];
                S[i] = turn(a0, b0, a1, b1);
            } else if (child[i] >= 0) {
                const double r[3] = {rest[3 * child[i]], rest[3 * child[i] + 1], rest[3 * child[i] + 2]};
                sm_qrot(P, r, a0);
                dif(j, child[i], i, a1);
                unit(a0), unit(a1);
                S[i] = sm_qmul(sm_qbetween(a0, a1), P);
            } else S[i] = P;
            const sm_quat l = sm_qmul((sm_quat){P.w, -P.x, -P.y, -P.z}, S[i]);
            const float q[4] = {(float)l.x, (float)l.y, (float)l.z, (float)l.w};
            memcpy(xyzw + (t * SM_J + (size_t)i) * 4, q, sizeof q);
        }
    }
}

static int cmp(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* each joint's offset in its parent's frame, the median over the frames per axis. Bones the rotations aim keep the
   rest direction and get the median length; the pelvis and chest side bones get their median place. tmp: 3T doubles */
void sm_clip_offsets(const float *joints, const float *xyzw, uint32_t T, double *tmp, float *off) {
    memset(off, 0, sizeof(float) * 3);
    for (int j = 1; j < SM_J; j++) {
        for (uint32_t t = 0; t < T; t++) {
            const float *q = xyzw + (size_t)t * SM_J * 4;
            sm_quat g = {1, 0, 0, 0};
            for (int a = sm_parent[j]; a >= 0; a = sm_parent[a])
                g = sm_qmul((sm_quat){q[4 * a + 3], q[4 * a], q[4 * a + 1], q[4 * a + 2]}, g);
            double d[3], v[3];
            dif(joints + (size_t)t * SM_J * 3, j, sm_parent[j], d);
            sm_qrot((sm_quat){g.w, -g.x, -g.y, -g.z}, d, v);
            for (int k = 0; k < 3; k++) tmp[(size_t)k * T + t] = v[k];
        }
        for (int k = 0; k < 3; k++) {
            qsort(tmp + (size_t)k * T, T, sizeof(double), cmp);
            off[3 * j + k] = (float)tmp[(size_t)k * T + T / 2];
        }
    }
}

/* two-bone IK: the ankle to target, the knee in its current bend plane; the toe moves with the ankle */
static void leg_ik(float *j, int hip, int knee, int ankle, int toe, const double target[3]) {
    double hk[3], ka[3], d[3], pole[3];
    dif(j, knee, hip, hk), dif(j, ankle, knee, ka);
    for (int k = 0; k < 3; k++) d[k] = target[k] - j[3 * hip + k];
    const double a = sqrt(norm2(hk)), b = sqrt(norm2(ka)), lo = fabs(a - b) + 1e-6, hi = a + b - 1e-6;
    double dist = sqrt(norm2(d));
    if (dist < 1e-9) return;
    unit(d);
    const double along = hk[0] * d[0] + hk[1] * d[1] + hk[2] * d[2];
    for (int k = 0; k < 3; k++) pole[k] = hk[k] - along * d[k];
    if (norm2(pole) < 1e-18) return;
    unit(pole);
    dist = dist < lo ? lo : dist > hi ? hi : dist;
    const double x = (a * a - b * b + dist * dist) / (2 * dist), h = sqrt(fmax(0, a * a - x * x));   /* cosine rule */
    for (int k = 0; k < 3; k++) {
        const double hip_k = j[3 * hip + k], new_ankle = hip_k + d[k] * dist;
        j[3 * toe + k] += (float)(new_ankle - j[3 * ankle + k]);
        j[3 * knee + k] = (float)(hip_k + d[k] * x + pole[k] * h);
        j[3 * ankle + k] = (float)new_ankle;
    }
}

#define BLEND 4   /* frames over which a lock fades in and out */

/* while a foot is planted (ankle or toe contact) its toe keeps the yaw it had at touch-down, and returns over BLEND
   frames after lift-off: a path warp turns the body, which would swivel a planted foot */
static void hold_yaw(float *joints, const uint8_t *contacts, uint32_t T, int side, const double *dpsi) {
    const int ankle = 7 + side, toe = 10 + side, col = 2 * side;
    double at = 0, off = 0;
    uint32_t since = BLEND + 1;
    for (uint32_t t = 0; t < T; t++) {
        const uint8_t *ct = contacts + 4 * t + col;
        const int on = ct[0] || ct[1];
        if (on && (!t || !(ct[-4] || ct[-3]))) at = dpsi[t];
        if (on) since = 0, off = at - dpsi[t];
        const double r = on ? off : ++since <= BLEND ? off * (1 - since / (BLEND + 1.0)) : 0;
        if (r == 0) continue;
        float *a = joints + ((size_t)t * SM_J + (size_t)ankle) * 3, *f = joints + ((size_t)t * SM_J + (size_t)toe) * 3;
        const double c = cos(r), sn = sin(r), dx = (double)f[0] - a[0], dz = (double)f[2] - a[2];
        f[0] = (float)(a[0] + dx * c + dz * sn), f[2] = (float)(a[2] - dx * sn + dz * c);
    }
}

/* each ankle contact run holds the ankle where the run starts; the offset fades out over BLEND frames around the run;
   nothing goes below the floor (y = 0) */
void sm_foot_lock(float *joints, const uint8_t *contacts, uint32_t T, double *delta, const double *dpsi) {
    static const int leg[2][5] = {{1, 4, 7, 10, 0}, {2, 5, 8, 11, 2}};   /* hip, knee, ankle, toe, contact column */
    for (int s = 0; s < 2; s++) {
        const int *g = leg[s];
        double anchor[3] = {0};
        for (uint32_t t = 0; t < T; t++) {
            const float *a = joints + ((size_t)t * SM_J + (size_t)g[2]) * 3;
            const uint8_t *ct = contacts + 4 * t + g[4];
            const int on = ct[0], start = on && (!t || !ct[-4]);
            for (int k = 0; on && start && k < 3; k++) anchor[k] = a[k];
            for (int k = 0; k < 3; k++) delta[3 * t + (uint32_t)k] = on ? anchor[k] - a[k] : 0;
        }
        for (uint32_t t = 0; t < T; t++) {
            if (contacts[4 * t + (uint32_t)g[4]]) continue;
            for (uint32_t d = 1; d <= BLEND; d++) {
                const int64_t src = t >= d && contacts[4 * (t - d) + (uint32_t)g[4]] ? (int64_t)(t - d)
                                  : t + d < T && contacts[4 * (t + d) + (uint32_t)g[4]] ? (int64_t)(t + d) : -1;
                if (src < 0) continue;
                for (int k = 0; k < 3; k++) delta[3 * t + (uint32_t)k] = delta[3 * src + k] * (1 - d / (BLEND + 1.0));
                break;
            }
        }
        for (uint32_t t = 0; t < T; t++) {
            float *jt = joints + (size_t)t * SM_J * 3, *a = jt + 3 * g[2];
            const double *d = delta + 3 * t, low = fmin(a[1], jt[3 * g[3] + 1]) + d[1];
            if (d[0] == 0 && d[1] == 0 && d[2] == 0 && low >= 0) continue;
            const double target[3] = {a[0] + d[0], a[1] + d[1] - fmin(low, 0), a[2] + d[2]};
            leg_ik(jt, g[0], g[1], g[2], g[3], target);
        }
        if (dpsi) hold_yaw(joints, contacts, T, s, dpsi);
    }
}

#define GAP 20   /* frames: a foot column with no contact this long has its flags missing (runs) */

/* where a column has no flag for GAP frames, its joint is on the ground while it is low and slow, as the flags are
   set in walks: ankle below 10 cm, toe below 4 cm, slower than 0.8 m/s */
static void fill_contacts(const float *joints, uint8_t *contacts, uint32_t T, double fps) {
    static const int joint[4] = {7, 10, 8, 11};
    static const float low[4] = {0.10f, 0.04f, 0.10f, 0.04f};
    for (uint32_t k = 0; k < 4; k++)
        for (uint32_t a = 0, b; a < T; a = b + 1) {
            for (b = a; b < T && !contacts[4 * b + k]; b++) {}
            for (uint32_t t = a; b - a >= GAP && t < b; t++) {
                const uint32_t p = t ? t - 1 : 0, n = t + 1 < T ? t + 1 : T - 1;
                const float *x = joints + ((size_t)t * SM_J + (size_t)joint[k]) * 3;
                const float *xp = joints + ((size_t)p * SM_J + (size_t)joint[k]) * 3, *xn = xp + (n - p) * SM_J * 3;
                const double dx = (double)xn[0] - xp[0], dz = (double)xn[2] - xp[2];
                contacts[4 * t + k] = x[1] < low[k] && sqrt(dx * dx + dz * dz) * fps < 0.8 * (n - p);
            }
        }
}

#define MAX_TOKENS 6000   /* 10 minutes */

sm_status sm_motion_create(const sm_model *m, sm_motion **out) {
    sm_motion *mo = m && out ? sm_alloc(&m->h, sizeof *mo) : NULL;
    if (!mo) return m && out ? SM_E_NOMEM : SM_E_ARG;
    memset(mo, 0, sizeof *mo);
    mo->m = m, mo->seed = -1;
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
    mo->mem = mo->feat = f, mo->joints = f + T * SM_F, mo->rot = mo->joints + T * SM_J * 3;
    mo->root = mo->rot + T * SM_J * 4, mo->speed = mo->root + T * 3, mo->conf = mo->speed + T;
    mo->codes = (int16_t *)(mo->conf + tokens);
    mo->contacts = (uint8_t *)(mo->codes + (size_t)tokens * SM_VQ_LAYERS);
    mo->cap = tokens, mo->tokens = mo->frames = 0;
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
    dst->tokens = src->tokens, dst->frames = src->frames, dst->seed = src->seed, dst->end = src->end;
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

void sm_motion_post(sm_motion *mo, const sm_track *tr, const double *v, const sm_path *path, double *tmp) {
    const uint32_t T = mo->frames;
    const double fps = mo->m->fps;
    double *dpsi = tmp + 3 * (size_t)T, *g = dpsi + T;
    sm_track track = *tr;
    sm_features_to_joints(mo->feat, T, &track, mo->joints);
    mo->end = track;
    for (size_t t = 0; t < T; t++)
        for (size_t k = 0; k < 4; k++) mo->contacts[4 * t + k] = mo->feat[t * SM_F + SM_F - 4 + k] > 0.5f;
    fill_contacts(mo->joints, mo->contacts, T, fps);
    sm_root_smooth(mo->joints, T, fps, g, mo->speed);
    const int warp = path && path->n && v[SM_P_PATH_WARP] != 0;
    mo->path_err = -1;
    if (path && path->n) mo->path_err = (float)sm_path_warp(path, mo->joints, T, fps, g, warp, dpsi, g + 2 * (size_t)T);
    for (size_t t = 0; t < T; t++) memcpy(mo->root + 3 * t, mo->joints + t * SM_J * 3, sizeof(float) * 3);
    if (v[SM_P_FOOT_LOCK] != 0) sm_foot_lock(mo->joints, mo->contacts, T, tmp, warp ? dpsi : NULL);
    sm_rotations(mo->joints, T, mo->m->rest, mo->rot);
    sm_clip_offsets(mo->joints, mo->rot, T, tmp, mo->off);
    for (size_t i = 0; v[SM_P_IN_PLACE] != 0 && i < (size_t)T * SM_J; i++) {
        mo->joints[3 * i] -= mo->root[3 * (i / SM_J)];
        mo->joints[3 * i + 2] -= mo->root[3 * (i / SM_J) + 2];
    }
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
    while (digits && f % 10 == 0) f /= 10, digits--;
    for (; digits; digits--, f /= 10) *--p = (char)('0' + f % 10);
    if (*p) *--p = '.';
    do *--p = (char)('0' + i % 10);
    while (i /= 10);
    if (v < 0 && u) *--p = '-';
    sm_puts(b, p);
}

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
        line(b, depth[j], j ? "JOINT " : "ROOT "), sm_puts(b, joint_names[j]), sm_puts(b, "\n");
        line(b, depth[j], "{\n");
        line(b, depth[j] + 1, "OFFSET");
        for (int k = 0; k < 3; k++) sm_puts(b, " "), sm_putn(b, mo->off[3 * j + k] * scale);
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
    const double next[3] = {a * 57.29577951308232, b * 57.29577951308232, c * 57.29577951308232};
    for (int k = 0; k < 3; k++) e[k] = next[k] + 360 * round((e[k] - next[k]) / 360);
}

static void frames(const sm_motion *mo, double scale, sm_buf *b) {
    double e[SM_J][3] = {{0}};
    sm_puts(b, "MOTION\nFrames: "), sm_putn(b, mo->frames);
    sm_puts(b, "\nFrame Time: "), sm_putn(b, 1.0 / mo->m->fps);
    for (size_t t = 0; t < mo->frames; t++) {
        sm_puts(b, "\n");
        for (int k = 0; k < 3; k++) sm_putn(b, mo->joints[t * SM_J * 3 + (size_t)k] * scale), sm_puts(b, " ");
        for (int i = 0; i < SM_J; i++) {
            euler_zyx(mo->rot + (t * SM_J + dfs[i]) * 4, e[dfs[i]]);
            for (int k = 0; k < 3; k++) sm_putn(b, e[dfs[i]][k]), sm_puts(b, i + 1 < SM_J || k < 2 ? " " : "");
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
