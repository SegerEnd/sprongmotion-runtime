/* Joint positions to the 263-d motion features the encoder reads: the training data's recipe (Guo et al.
   2022, process_file): the clip on the target skeleton, on the floor, at the origin facing +Z, then root yaw velocity,
   xz velocity and height, local joint positions, 6D joint rotations, joint velocities and foot contacts. */
#include "sm_internal.h"

static const double raw[SM_J][3] = {
    {0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 1, 0},
    {0, 0, 1}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0},
    {0, -1, 0}, {0, -1, 0}, {0, -1, 0},
};
static const double target_len[SM_J] = {   /* bone lengths of the target skeleton, metres */
    0, 0.10307398438453674, 0.1098833754658699, 0.13156822323799133, 0.3936232030391693, 0.39018821716308594,
    0.1431901901960373, 0.43243294954299927, 0.4256434142589569, 0.0573648139834404, 0.14338180422782898,
    0.14941906929016113, 0.21936000883579254, 0.1374867856502533, 0.14338283240795135, 0.1030391976237297,
    0.13161399960517883, 0.12298436462879181, 0.2568398416042328, 0.2630917429924011, 0.2660118341445923,
    0.2698763310909271,
};
static const int8_t chains[5][7] = {
    {0, 2, 5, 8, 11, -1}, {0, 1, 4, 7, 10, -1}, {0, 3, 6, 9, 12, 15, -1}, {9, 14, 17, 19, 21, -1},
    {9, 13, 16, 18, 20, -1},
};
enum { SIGMA = 20, RADIUS = 80 };   /* the facing smoothing: scipy gaussian_filter1d, radius int(4 sigma + 0.5) */

static void sub(const double *a, const double *b, double *o) { for (int k = 0; k < 3; k++) o[k] = a[k] - b[k]; }
static double len3(const double *v) { return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
static void unit3(double *v) { const double n = len3(v); for (int k = 0; k < 3; k++) v[k] /= n; }
static sm_quat conj(sm_quat q) { return (sm_quat){q.w, -q.x, -q.y, -q.z}; }

/* asin x = 2 atan(x / (1 + sqrt(1 - x^2))); atan halved twice to |z| <= tan(pi/16), then its series to z^25 (error
   < 1e-19): only + - * / and sqrt, the same bits on every target */
static double asin_(double x) {
    x = x > 1 ? 1 : x < -1 ? -1 : x;
    double z = x / (1 + sqrt(1 - x * x)), s = 0;
    z = z / (1 + sqrt(1 + z * z)), z = z / (1 + sqrt(1 + z * z));
    for (int k = 12; k >= 0; k--) s = 1.0 / (2 * k + 1) - z * z * s;
    return 8 * z * s;
}

/* (az, 0, -ax): Y x unit(a), a = the hip line plus the shoulder line */
static void facing(const double *p, int sign, double *f) {
    double a[3];
    for (int k = 0; k < 3; k++) a[k] = sign * (p[3 + k] - p[6 + k]) + (p[51 + k] - p[48 + k]);
    unit3(a);
    f[0] = a[2], f[1] = 0, f[2] = -a[0];
}

/* the bones' directions with the target lengths, the root scaled by leg length (IK then FK onto the target skeleton) */
static void retarget(const float *joints, uint32_t T, double *P) {
    double d[3], src[2];
    for (int i = 0; i < 2; i++) {   /* right knee and ankle bones of the first frame, rounded as the reference does */
        const int j = i ? 8 : 5;
        for (int k = 0; k < 3; k++) d[k] = (double)joints[3 * j + k] - joints[3 * sm_parent[j] + k];
        src[i] = (float)len3(d);
    }
    const double scale = (target_len[5] + target_len[8]) / (src[0] + src[1]);
    for (size_t t = 0; t < T; t++) {
        const float *j = joints + t * SM_J * 3;
        double *p = P + t * SM_J * 3;
        for (int k = 0; k < 3; k++) p[k] = j[k] * scale;
        for (int b = 1; b < SM_J; b++) {
            const int a = sm_parent[b];
            for (int k = 0; k < 3; k++) d[k] = (double)j[3 * b + k] - j[3 * a + k];
            unit3(d);
            for (int k = 0; k < 3; k++) p[3 * b + k] = p[3 * a + k] + d[k] * target_len[b];
        }
    }
}

/* on the floor, root at x = z = 0 and facing +Z in the first frame */
static void canonical(double *P, uint32_t T) {
    const size_t n = (size_t)T * SM_J;
    double lo = INFINITY, f[3], o[3];
    for (size_t i = 0; i < n; i++) lo = P[3 * i + 1] < lo ? P[3 * i + 1] : lo;
    const double x0 = P[0], z0 = P[2], z[3] = {0, 0, 1};
    for (size_t i = 0; i < n; i++) P[3 * i] -= x0, P[3 * i + 1] -= lo, P[3 * i + 2] -= z0;
    facing(P, -1, f);
    unit3(f);
    const sm_quat q = sm_qbetween(f, z);
    for (size_t i = 0; i < n; i++) sm_qrot(q, P + 3 * i, o), memcpy(P + 3 * i, o, sizeof o);
}

/* the root's yaw per frame from the smoothed facing (edges repeated); frame 0 keeps no turn, as the reference */
static void root_yaw(const double *P, uint32_t T, double *F, sm_quat *R) {
    double w[RADIUS + 1], sum = 0;
    for (int k = 0; k <= RADIUS; k++) w[k] = sm_exp(-0.5 / (SIGMA * SIGMA) * k * k), sum += k ? 2 * w[k] : w[k];
    for (int k = 0; k <= RADIUS; k++) w[k] /= sum;
    for (size_t t = 0; t < T; t++) facing(P + t * SM_J * 3, 1, F + 3 * t);
    for (int64_t t = 0; t < T; t++) {
        double s[3], z[3] = {0, 0, 1};
        for (int k = 0; k < 3; k++) s[k] = F[3 * t + k] * w[0];
        for (int64_t d = RADIUS; d >= 1; d--) {
            const int64_t a = t - d < 0 ? 0 : t - d, b = t + d >= T ? T - 1 : t + d;
            for (int k = 0; k < 3; k++) s[k] += (F[3 * a + k] + F[3 * b + k]) * w[d];
        }
        unit3(s);
        R[t] = t ? sm_qbetween(s, z) : (sm_quat){1, 0, 0, 0};
    }
}

/* the first two columns of the rotation matrix (Zhou et al. 2019) */
static void cont6d(sm_quat q, float *o) {
    const double s = 2 / (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    const double m[6] = {1 - s * (q.y * q.y + q.z * q.z), s * (q.x * q.y + q.z * q.w), s * (q.x * q.z - q.y * q.w),
                         s * (q.x * q.y - q.z * q.w), 1 - s * (q.x * q.x + q.z * q.z), s * (q.y * q.z + q.x * q.w)};
    for (int i = 0; i < 6; i++) o[i] = (float)m[i];
}

/* the rotations of each chain's bones relative to the rotation before (the chain's start: the root's yaw) */
static void bone_rotations(const double *p, sm_quat root, float *rot6d) {
    for (int c = 0; c < 5; c++) {
        sm_quat rot = root;
        for (int i = 1; chains[c][i] >= 0; i++) {
            const int a = chains[c][i - 1], b = chains[c][i];
            double d[3];
            sub(p + 3 * b, p + 3 * a, d);
            unit3(d);
            const sm_quat local = sm_qmul(conj(rot), sm_qbetween(raw[b], d));
            cont6d(local, rot6d + 6 * (b - 1));
            rot = sm_qmul(rot, local);
        }
    }
}

size_t sm_feat_tmp_doubles(uint32_t T) { return (size_t)T * (SM_J * 3 + 4 + 3); }

void sm_joints_to_features(const float *joints, uint32_t T, double *tmp, float *feat) {
    double *P = tmp, *F = P + (size_t)T * SM_J * 3;
    sm_quat *R = (sm_quat *)(void *)(F + 3 * (size_t)T);
    static const int feet[4] = {7, 10, 8, 11};
    retarget(joints, T, P);
    canonical(P, T);
    root_yaw(P, T, F, R);
    for (size_t t = 0; t + 1 < T; t++) {
        const double *p = P + t * SM_J * 3, *n = p + SM_J * 3;
        float *f = feat + t * SM_F;
        double d[3], o[3];
        sub(n, p, d);
        sm_qrot(R[t + 1], d, o);
        const sm_quat yv = sm_qmul(R[t + 1], conj(R[t]));
        f[0] = (float)asin_(yv.y), f[1] = (float)o[0], f[2] = (float)o[2], f[3] = (float)p[1];
        for (int j = 1; j < SM_J; j++) {
            const double c[3] = {p[3 * j] - p[0], p[3 * j + 1], p[3 * j + 2] - p[2]};
            sm_qrot(R[t], c, o);
            for (int k = 0; k < 3; k++) f[4 + 3 * (j - 1) + k] = (float)o[k];
        }
        bone_rotations(p, R[t], f + 67);
        for (int j = 0; j < SM_J; j++) {
            sub(n + 3 * j, p + 3 * j, d);
            sm_qrot(R[t], d, o);
            for (int k = 0; k < 3; k++) f[193 + 3 * j + k] = (float)o[k];
        }
        for (int i = 0; i < 4; i++) {
            sub(n + 3 * feet[i], p + 3 * feet[i], d);
            f[259 + i] = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < 0.002 ? 1.0f : 0.0f;
        }
    }
}
