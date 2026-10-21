/* One pass at a time, with progress, a preview and a cancel. Usage: 03_steps model.gguf */
#include <stdio.h>
#include "sm.h"

static int32_t stop_after(void *user, uint32_t done, uint32_t total) {   /* nonzero cancels the run */
    printf("progress %u / %u\n", done, total);
    return done >= *(const uint32_t *)user;
}

int main(int argc, char **argv) {
    char err[256];
    sm_model *m;
    if (argc != 2) return fprintf(stderr, "usage: %s model.gguf\n", argv[0]), 2;
    if (sm_model_load_file(argv[1], NULL, &m, err, sizeof err) != SM_OK) return fprintf(stderr, "%s\n", err), 1;
    sm_ctx *c;
    sm_request *r;
    sm_motion *mo;
    sm_ctx_create(m, &c);
    sm_request_create(m, &r);
    sm_motion_create(m, &mo);
    sm_request_set_text(r, "a person waves with both hands", -1);
    uint32_t done = 0, total = 0;
    sm_status s = sm_ctx_begin(c, r, &mo, 1);
    while (s == SM_OK) {
        s = sm_ctx_step(c, &done, &total);
        if (s == SM_OK && done == total / 2 && sm_ctx_preview(c) == SM_OK)
            printf("preview after %u of %u passes: %u frames\n", done, total, sm_motion_frames(mo));
    }
    if (s == SM_DONE) s = sm_ctx_finish(c);
    printf("%s: %u frames\n", sm_status_name(s), sm_motion_frames(mo));
    uint32_t passes = 3;   /* a UI's cancel button would call sm_ctx_cancel(c), from any thread */
    sm_ctx_set_progress(c, stop_after, &passes);
    printf("%s: %s\n", sm_status_name(sm_ctx_generate(c, r, &mo, 1)), sm_ctx_last_error(c));
    sm_motion_free(mo);
    sm_request_free(r);
    sm_ctx_free(c);
    sm_model_free(m);
    return s != SM_OK;
}
