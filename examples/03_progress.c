/* Progress and cancel. Usage: 03_progress model.gguf */
#include <stdio.h>
#include "sprongmotion.h"

/* returning false cancels; a UI's cancel button sets a flag this reads, from any thread */
static bool report(void *user, int done, int total) {
    printf("pass %d of %d\n", done, total);
    return done < *(const int *)user;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    sm_model model = sm_load_model(argv[1]);
    int stop_after = 100;
    sm_options options = sm_default_options(model);
    options.progress = report;
    options.user = &stop_after;

    sm_motion motion = sm_generate_motion_ex(model, "a person waves with both hands", options);
    const bool ok = sm_is_motion_valid(motion);
    printf("%s: %d frames\n", ok ? "done" : sm_get_error(), motion.frame_count);
    sm_unload_motion(motion);

    stop_after = 3;
    motion = sm_generate_motion_ex(model, "a person waves with both hands", options);
    printf("cancelled: %s\n", sm_get_error());
    sm_unload_motion(motion);
    sm_unload_model(model);
    return !ok;
}
