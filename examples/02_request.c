/* Parameters, three variants of one prompt, and error messages. Usage: 02_request model.gguf */
#include <stdio.h>
#include "sm.h"

int main(int argc, char **argv) {
    char err[256];
    sm_model *m;
    if (argc != 2) return fprintf(stderr, "usage: %s model.gguf\n", argv[0]), 2;
    if (sm_model_load_file(argv[1], NULL, &m, err, sizeof err) != SM_OK) return fprintf(stderr, "%s\n", err), 1;
    for (sm_param p = sm_param_next(0); p > 0; p = sm_param_next(p)) {   /* what this model file accepts */
        double lo, hi, dflt;
        sm_param_bounds(m, p, &lo, &hi, &dflt, NULL);
        printf("%-10s %.10g .. %.10g, default %.10g\n", sm_param_name(p), lo, hi, dflt);
    }
    sm_ctx *c;
    sm_request *r;
    sm_motion *outs[3];
    sm_ctx_create(m, &c);
    sm_request_create(m, &r);
    for (int i = 0; i < 3; i++) sm_motion_create(m, &outs[i]);
    if (sm_request_set_float(r, SM_P_SECONDS, 60) != SM_OK) printf("rejected: %s\n", sm_request_last_error(r));
    sm_status s = sm_request_set_text(r, "a person jumps forward", -1);
    if (s == SM_OK) s = sm_request_set_float(r, SM_P_SECONDS, 3.0);
    if (s == SM_OK) s = sm_request_set_int(r, SM_P_SEED, 42);
    if (s == SM_OK) s = sm_request_set_by_name(r, "steps", "12");   /* by name, as from a config file */
    if (s != SM_OK) fprintf(stderr, "request: %s\n", sm_request_last_error(r));
    else if ((s = sm_ctx_generate(c, r, outs, 3)) != SM_OK) fprintf(stderr, "generate: %s\n", sm_ctx_last_error(c));
    for (int i = 0; s == SM_OK && i < 3; i++)
        printf("variant %d: seed %d, %u frames\n", i, sm_motion_seed(outs[i]), sm_motion_frames(outs[i]));
    for (int i = 0; i < 3; i++) sm_motion_free(outs[i]);
    sm_request_free(r);
    sm_ctx_free(c);
    sm_model_free(m);
    return s != SM_OK;
}
