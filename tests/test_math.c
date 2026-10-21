/* Own exp and log against libm in double, float exp and erf in ulps against libm, float16 rounding against
   the compiler's _Float16. */
#include "../src/sm_internal.h"
#include "check.h"
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 99;
static double urand(void) { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return (double)(rs >> 11) * 0x1p-53; }

static double rel(double a, double b) { return b == 0 ? fabs(a) : fabs(a - b) / fabs(b); }

static int64_t ordered(float f) { int32_t i; memcpy(&i, &f, 4); return i < 0 ? (int64_t)INT32_MIN - i : i; }

static float from_ordered(int64_t o) { const int32_t i = (int32_t)(o < 0 ? INT32_MIN - o : o); float f; memcpy(&f, &i, 4); return f; }

/* every 31st float in [lo, hi): the most ulps off and the share more than 2 ulps off (against double libm, rounded) */
static void ulps(float (*f)(float), double (*ref)(double), float lo, float hi, int64_t *worst, double *over2) {
    size_t n = 0, over = 0; *worst = 0;
    for (int64_t o = ordered(lo); o < ordered(hi); o += 31, n++) {
        const float x = from_ordered(o);
        const int64_t u = llabs(ordered(f(x)) - ordered((float)ref(x)));
        *worst = u > *worst ? u : *worst; over += u > 2;
    }
    *over2 = (double)over / (double)n;
}
static float expf_(float x) { return sm_expf(x); }
static float erff_(float x) { return sm_erff(x); }

int main(void) {
    double e_exp = 0, e_log = 0;
    for (int i = 0; i < 1000000; i++) {
        const double x = (urand() - 0.5) * 1400, u = urand() * pow(2, -60 * urand());
        e_exp = fmax(e_exp, rel(sm_exp(x), exp(x)));
        e_log = fmax(e_log, fabs(sm_log(u) - log(u)) / fmax(1, fabs(log(u))));
    }
    printf("max error: exp %.2g rel, log %.2g\n", e_exp, e_log);
    CHECK(e_exp < 4e-16); CHECK(e_log < 4e-16);
    CHECK(sm_exp(-800) == 0); CHECK(isinf(sm_exp(800))); CHECK(sm_exp(0) == 1);
    CHECK(rel(sm_exp(-740), exp(-740)) < 1e-6);   /* subnormal result */
    CHECK(isinf(sm_log(0)) && sm_log(0) < 0); CHECK(isnan(sm_log(-1))); CHECK(sm_log(1) == 0);
    CHECK(rel(sm_log(5e-320), log(5e-320)) < 1e-15);
    int64_t u_exp, u_erf; double o_exp, o_erf;
    ulps(expf_, exp, -87.3f, 88, &u_exp, &o_exp); ulps(erff_, erf, -6, 6, &u_erf, &o_erf);
    printf("float exp: max %lld ulp (%.1e of floats > 2 ulp), erf: max %lld ulp (%.1e > 2 ulp)\n", (long long)u_exp, o_exp,
           (long long)u_erf, o_erf);
    CHECK(u_exp <= 2); CHECK(u_erf <= 8 && o_erf < 3e-3);
    CHECK(sm_expf(-100) == 0); CHECK(isinf(sm_expf(100))); CHECK(sm_expf(0) == 1);
    CHECK(sm_erff(0) == 0); CHECK(sm_erff(10) == 1); CHECK(sm_erff(-10) == -1);
    NEAR(sm_gelu(1.0f), 0.8413447460685429, 2e-7);

#ifdef __FLT16_MAX__
    int bad = 0;
    for (uint32_t b = 0; b < 0xFFFFFFFFu - 4093; b += 4093) {
        float x, want;
        memcpy(&x, &b, 4);
        want = (float)__extension__(_Float16)x;
        const float got = sm_round_half(x);
        bad += memcmp(&got, &want, 4) != 0 && !(isnan(x) && isnan(got));
    }
    printf("float16 rounding: %d of 1049k floats differ from _Float16\n", bad);
    CHECK(bad == 0);
#endif
    return CHECK_DONE();
}
