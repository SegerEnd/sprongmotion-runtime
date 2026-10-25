/* Options, three seeds of one encoded prompt, and error messages. Usage: 02_options model.gguf */
#include <stdio.h>
#include "sprongmotion.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    sm_model model = sm_load_model(argv[1]);
    if (!sm_is_model_valid(model)) {
        fprintf(stderr, "%s\n", sm_get_error());
        return 1;
    }

    sm_options options = sm_default_options(model);
    options.seconds = 60;
    sm_motion rejected = sm_generate_motion_ex(model, "a person jumps forward", options);
    printf("rejected: %s\n", sm_get_error());
    sm_unload_motion(rejected);

    options.seconds = 3;
    options.steps = 12;
    sm_text text = sm_encode_text(model, "a person jumps forward");   /* encoded once for every seed */
    options.text = &text;
    int failed = !sm_is_text_valid(text);
    for (int seed = 42; seed < 45 && !failed; seed++) {
        options.seed = seed;
        sm_motion motion = sm_generate_motion_ex(model, NULL, options);
        failed = !sm_is_motion_valid(motion);
        if (!failed) printf("seed %d: %d frames\n", motion.seed, motion.frame_count);
        sm_unload_motion(motion);
    }
    if (failed) fprintf(stderr, "%s\n", sm_get_error());
    sm_unload_text(text);
    sm_unload_model(model);
    return failed;
}
