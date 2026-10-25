/* Quaternions, features to joints vs the reference, rotations back onto the bones. */
#include "../src/sm_internal.h"
#include "check.h"
#include <stdlib.h>
#include <string.h>

static void rand_unit(unsigned *s, double v[3]) {
    double n; do { v[0] = rnd(s) * 2 - 1; v[1] = rnd(s) * 2 - 1; v[2] = rnd(s) * 2 - 1; n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); } while (n < 0.1);
    v[0] /= n; v[1] /= n; v[2] /= n;
}

static void test_quat(void) {
    unsigned s = 1;
    for (int i = 0; i < 1000; i++) {
        double u[3], v[3], r[3]; rand_unit(&s, u); rand_unit(&s, v);
        const sm_quat q = sm_qbetween(u, v); sm_qrot(q, u, r);
        for (int k = 0; k < 3; k++) NEAR(r[k], v[k], 1e-12);                       /* takes u onto v */
        NEAR(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z, 1, 1e-12);              /* unit */
        const sm_quat a = sm_qbetween(u, u); NEAR(a.w, 1, 1e-12);                   /* equal vectors: identity */
        const double o[3] = {-u[0], -u[1], -u[2]}; sm_qrot(sm_qbetween(u, o), u, r);
        for (int k = 0; k < 3; k++) NEAR(r[k], o[k], 1e-12);                        /* opposite vectors */
        const sm_quat b = sm_qbetween(v, u), c = sm_qbetween(u, o);                 /* (ab)c = a(bc) */
        const sm_quat l = sm_qmul(sm_qmul(q, b), c), m = sm_qmul(q, sm_qmul(b, c));
        NEAR(l.w, m.w, 1e-12); NEAR(l.x, m.x, 1e-12); NEAR(l.y, m.y, 1e-12); NEAR(l.z, m.z, 1e-12);
    }
    const double x[3] = {1, 0, 0}, mx[3] = {-1, 0, 0}, r0[3] = {0, 0, 0}; double r[3];  /* the axis-aligned opposite case */
    sm_qrot(sm_qbetween(x, mx), x, r); NEAR(r[0], -1, 1e-12);
    sm_qrot((sm_quat){1, 0, 0, 0}, r0, r); NEAR(r[0], 0, 0);
}

static void test_golden(const char *path, const float *rest) {
    FILE *f = fopen(path, "rb"); CHECK(f != NULL); if (!f) return;
    uint32_t T; int clips = 0;
    while (fread(&T, 4, 1, f) == 1) {
        float *feat = malloc(sizeof(float) * T * SM_F), *ref = malloc(sizeof(float) * T * SM_J * 3), *got = malloc(sizeof(float) * T * SM_J * 3);
        float *rot = malloc(sizeof(float) * T * SM_J * 4);
        CHECK(fread(feat, sizeof(float), (size_t)T * SM_F, f) == (size_t)T * SM_F);
        CHECK(fread(ref, sizeof(float), (size_t)T * SM_J * 3, f) == (size_t)T * SM_J * 3);
        sm_track tr = {0};
        sm_features_to_joints(feat, T, &tr, got);
        double worst = 0;
        for (size_t i = 0; i < (size_t)T * SM_J * 3; i++) worst = fmax(worst, fabs(got[i] - ref[i]));
        NEAR(worst, 0, 1e-5);                                                        /* metres: 0.01 mm */
        sm_rotations(got, T, rest, rot);
        for (uint32_t t = 0; t < T; t++) {   /* every bone but the side bones of the pelvis and chest points exactly at its joint */
            sm_quat G[SM_J]; const float *j = got + (size_t)t * SM_J * 3, *q = rot + (size_t)t * SM_J * 4;
            for (int b = 0; b < SM_J; b++) {
                const sm_quat l = {q[4 * b + 3], q[4 * b], q[4 * b + 1], q[4 * b + 2]}; const int a = sm_parent[b];
                NEAR(l.w * l.w + l.x * l.x + l.y * l.y + l.z * l.z, 1, 1e-6);
                G[b] = a < 0 ? l : sm_qmul(G[a], l);
                if (a < 0 || b == 1 || b == 2 || b == 3 || b == 13 || b == 14) continue;   /* the side bones */
                double r[3] = {rest[3 * b], rest[3 * b + 1], rest[3 * b + 2]}, d[3], o[3], n = 0, m = 0, dot = 0;
                for (int k = 0; k < 3; k++) d[k] = (double)j[3 * b + k] - j[3 * a + k];
                sm_qrot(G[a], r, o);
                for (int k = 0; k < 3; k++) n += o[k] * o[k], m += d[k] * d[k], dot += o[k] * d[k];
                NEAR(dot / sqrt(n * m), 1, 1e-6);
            }
        }
        free(feat); free(ref); free(got); free(rot); clips++;
    }
    fclose(f); CHECK(clips == 8);
}

int main(int argc, char **argv) {
    test_quat();
    sm_model *m = NULL; CHECK(sm_model_load_file("tests/data/test.gguf", NULL, &m, NULL, 0) == SM_OK); if (!m) return CHECK_DONE();
    test_golden(argc > 1 ? argv[1] : "tests/data/motion_golden.bin", m->rest);
    sm_model_free(m);
    return CHECK_DONE();
}
