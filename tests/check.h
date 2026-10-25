/* The whole test framework: CHECK(cond) and NEAR(a, b, tol) count failures, plus the helpers several tests share. */
#ifndef CHECK_H
#define CHECK_H
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int check_fails, check_runs;
#define CHECK(c) do { check_runs++; if (!(c)) { check_fails++; fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c); } } while (0)
#define NEAR(a, b, tol) do { double a_ = (a), b_ = (b); check_runs++; if (!(fabs(a_ - b_) <= (tol))) { check_fails++; \
    fprintf(stderr, "%s:%d: %s = %.9g, %s = %.9g, |diff| %.3g > %.3g\n", __FILE__, __LINE__, #a, a_, #b, b_, fabs(a_ - b_), (double)(tol)); } } while (0)
#define CHECK_DONE() (printf("%s: %d checks, %d failed\n", __FILE__, check_runs, check_fails), check_fails != 0)

static inline int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* 0 .. 1 */
static inline double rnd(unsigned *s) {
    *s = *s * 1103515245u + 12345u;
    return ((*s >> 8) & 0xFFFF) / 65535.0;
}

/* the reference files: a count, then per array its name, dims and floats */
typedef struct { char name[64]; uint32_t nd, dim[4]; size_t n; float *data; } golden;
golden goldens[256];
uint32_t n_goldens;

static inline int load_goldens(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int ok = fread(&n_goldens, 4, 1, f) == 1 && n_goldens <= 256;
    for (uint32_t i = 0; ok && i < n_goldens; i++) {
        uint32_t len;
        golden *a = &goldens[i];
        a->n = 1;
        ok = fread(&len, 4, 1, f) == 1 && len < 64 && fread(a->name, 1, len, f) == len;
        ok = ok && fread(&a->nd, 4, 1, f) == 1 && a->nd <= 4 && fread(a->dim, 4, a->nd, f) == a->nd;
        for (uint32_t d = 0; ok && d < a->nd; d++) a->n *= a->dim[d];
        ok = ok && (a->data = malloc(sizeof(float) * a->n)) && fread(a->data, sizeof(float), a->n, f) == a->n;
    }
    fclose(f);
    return ok;
}

static inline const golden *find_golden(const char *name) {
    for (uint32_t k = 0; k < n_goldens; k++)
        if (!strcmp(goldens[k].name, name)) return &goldens[k];
    fprintf(stderr, "missing %s\n", name);
    exit(2);
}

static inline void free_goldens(void) {
    for (uint32_t i = 0; i < n_goldens; i++) free(goldens[i].data);
    n_goldens = 0;
}
#endif
