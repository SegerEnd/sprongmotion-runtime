/* Time per phase, best of N runs. Usage: bench model.gguf [threads (0 = the core count)] [N] */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sprongmotion.h"

static double now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

static double best(double a, double b) { return b < a ? b : a; }

typedef struct { double text, base, res, decode, total; } phases;
typedef struct { double last, base, res; int steps; } pass_times;

static bool on_pass(void *user, int done, int total) {
    (void)total;
    pass_times *p = user;
    const double t = now();
    *(done <= p->steps ? &p->base : &p->res) += t - p->last;
    p->last = t;
    return true;
}

static phases clip(sm_model model, const char *prompt, float seconds, int runs) {
    phases b = {1e9, 1e9, 1e9, 1e9, 1e9};
    sm_options options = sm_default_options(model);
    options.seconds = seconds;
    options.progress = on_pass;
    for (int i = 0; i < runs; i++) {
        pass_times p = {0, 0, 0, options.steps};
        const double t0 = now();
        sm_text text = sm_encode_text(model, prompt);
        options.text = &text;
        options.user = &p;
        p.last = now();
        const double text_ms = p.last - t0;
        sm_motion motion = sm_generate_motion_ex(model, NULL, options);
        const double end = now();
        if (!sm_is_motion_valid(motion)) {
            puts(sm_get_error());
            exit(1);
        }
        sm_unload_motion(motion);
        sm_unload_text(text);
        const int res = options.residual_layers;
        const phases run = {text_ms, p.base / options.steps, res ? p.res / res : 0, end - p.last, end - t0};
        b = (phases){best(b.text, run.text), best(b.base, run.base), best(b.res, run.res),
                     best(b.decode, run.decode), best(b.total, run.total)};
    }
    return b;
}

static double timeline(sm_model model, int runs) {
    const sm_block blocks[6] = {
        {"a person walks forward", NULL, 2, 1, NULL}, {"a person waves", NULL, 2, 2, NULL},
        {"a person jumps", NULL, 2, 3, NULL}, {"a person turns left", NULL, 2, 4, NULL},
        {"a person kneels down", NULL, 2, 5, NULL}, {"a person stands up and stretches", NULL, 2, 6, NULL},
    };
    double fastest = 1e9;
    for (int i = 0; i < runs; i++) {
        const double t = now();
        sm_motion motion = sm_generate_timeline(model, blocks, 6, sm_default_options(model));
        fastest = best(fastest, now() - t);
        if (!sm_is_motion_valid(motion)) {
            puts(sm_get_error());
            exit(1);
        }
        sm_unload_motion(motion);
    }
    return fastest;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [threads] [runs]\n", argv[0]);
        return 2;
    }
    const int threads = argc > 2 ? atoi(argv[2]) : 0, runs = argc > 3 ? atoi(argv[3]) : 10;
    sm_set_thread_count(threads);
    sm_model model = sm_load_model(argv[1]);
    if (!sm_is_model_valid(model)) {
        fprintf(stderr, "%s\n", sm_get_error());
        return 1;
    }
    printf("%s, %d threads, best of %d (ms)\n", sm_get_build_info(), threads, runs);
    const double tl = timeline(model, runs);
    printf("timeline 6 x 2 s   %7.1f total  %6.1f per block\n", tl, tl / 6);
    printf("clip    text  base/step  res/layer  decode   total\n");
    const float len[3] = {2, 5, 10};
    for (int i = 0; i < 3; i++) {
        const phases p = clip(model, "a person walks forward and waves with the left hand", len[i], runs);
        printf("%4.0f s %6.2f %10.2f %10.2f %7.2f %7.1f\n", len[i], p.text, p.base, p.res, p.decode, p.total);
    }
    sm_unload_model(model);
    return 0;
}
