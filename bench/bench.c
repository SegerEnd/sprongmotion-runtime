/* Time per phase, best of N runs. Usage: bench model.gguf [threads (0 = the core count)] [N] */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sm.h"

static double now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

static double best(double a, double b) { return b < a ? b : a; }

typedef struct { double text, base, res, decode, total; } phases;

static phases clip(sm_ctx *c, sm_request *r, sm_motion *mo, double seconds, int runs) {
    phases b = {1e9, 1e9, 1e9, 1e9, 1e9};
    int32_t steps, res;
    sm_request_set_float(r, SM_P_SECONDS, seconds);
    sm_request_get_int(r, SM_P_STEPS, &steps);
    sm_request_get_int(r, SM_P_RES_LAYERS, &res);
    for (int i = 0; i < runs; i++) {
        phases p = {0};
        double t0 = now(), t = t0;
        sm_status s = sm_ctx_begin(c, r, &mo, 1);
        p.text = now() - t;
        for (uint32_t pass = 0; s == SM_OK; pass++) {
            t = now();
            s = sm_ctx_step(c, NULL, NULL);
            *(pass < (uint32_t)steps ? &p.base : &p.res) += now() - t;
        }
        t = now();
        if (s != SM_DONE || sm_ctx_finish(c) != SM_OK) exit((puts(sm_ctx_last_error(c)), 1));
        p.decode = now() - t, p.total = now() - t0;
        b = (phases){best(b.text, p.text), best(b.base, p.base / steps), best(b.res, res ? p.res / res : 0),
                     best(b.decode, p.decode), best(b.total, p.total)};
    }
    return b;
}

int main(int argc, char **argv) {
    if (argc < 2) return fprintf(stderr, "usage: %s model.gguf [threads] [runs]\n", argv[0]), 2;
    sm_hooks h = SM_HOOKS_INIT;
    h.n_threads = argc > 2 ? (uint32_t)atoi(argv[2]) : 0;
    const int runs = argc > 3 ? atoi(argv[3]) : 10;
    char err[256];
    sm_model *m;
    if (sm_model_load_file(argv[1], &h, &m, err, sizeof err) != SM_OK) return fprintf(stderr, "%s\n", err), 1;
    sm_ctx *c;
    sm_request *r;
    sm_motion *mo;
    sm_ctx_create(m, &c);
    sm_request_create(m, &r);
    sm_motion_create(m, &mo);
    sm_ctx_reserve(c, 10, 1);
    sm_motion_reserve(mo, 12);
    sm_request_set_text(r, "a person walks forward and waves with the left hand", -1);
    printf("%s, %u threads, best of %d (ms)\n", sm_build_info(), h.n_threads, runs);
    const char *texts[6] = {"a person walks forward", "a person waves", "a person jumps", "a person turns left",
                            "a person kneels down", "a person stands up and stretches"};
    const double secs[6] = {2, 2, 2, 2, 2, 2};
    const int32_t seeds[6] = {1, 2, 3, 4, 5, 6};
    double tl = 1e9;
    for (int i = 0; i < runs; i++) {
        const double t = now();
        if (sm_ctx_generate_blocks(c, r, 6, texts, NULL, secs, seeds, NULL, NULL, mo) != SM_OK)
            return puts(sm_ctx_last_error(c)), 1;
        tl = best(tl, now() - t);
    }
    printf("timeline 6 x 2 s   %7.1f total  %6.1f per block\n", tl, tl / 6);
    printf("clip    text  base/step  res/layer  decode   total\n");
    const double len[3] = {2, 5, 10};
    for (int i = 0; i < 3; i++) {
        const phases p = clip(c, r, mo, len[i], runs);
        printf("%4.0f s %6.2f %10.2f %10.2f %7.2f %7.1f\n", len[i], p.text, p.base, p.res, p.decode, p.total);
    }
    sm_motion_free(mo);
    sm_request_free(r);
    sm_ctx_free(c);
    sm_model_free(m);
    return 0;
}
