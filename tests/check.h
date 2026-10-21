/* The whole test framework: CHECK(cond) and NEAR(a, b, tol) count failures; main returns the count. */
#ifndef CHECK_H
#define CHECK_H
#include <math.h>
#include <stdio.h>
static int check_fails, check_runs;
#define CHECK(c) do { check_runs++; if (!(c)) { check_fails++; fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c); } } while (0)
#define NEAR(a, b, tol) do { double a_ = (a), b_ = (b); check_runs++; if (!(fabs(a_ - b_) <= (tol))) { check_fails++; \
    fprintf(stderr, "%s:%d: %s = %.9g, %s = %.9g, |diff| %.3g > %.3g\n", __FILE__, __LINE__, #a, a_, #b, b_, fabs(a_ - b_), (double)(tol)); } } while (0)
#define CHECK_DONE() (printf("%s: %d checks, %d failed\n", __FILE__, check_runs, check_fails), check_fails != 0)
#endif
