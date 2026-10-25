/* Change a motion without starting over: make a range again, make a loop, make one block of a timeline again, and
   (with a model file that has the motion encoder) continue motion that came from elsewhere. Usage: 04_edit model.gguf */
#include <stdio.h>
#include <string.h>
#include "sprongmotion.h"

#define TOKENS 50   /* 5 s: 2 frames per token, 6 codes per token */

static bool made(sm_motion motion) {
    if (!sm_is_motion_valid(motion)) fprintf(stderr, "%s\n", sm_get_error());
    return sm_is_motion_valid(motion);
}

static bool made_model(sm_model model) {
    if (!sm_is_model_valid(model)) fprintf(stderr, "%s\n", sm_get_error());
    return sm_is_model_valid(model);
}

static bool remake_range(sm_model model, sm_options options) {
    int16_t codes[TOKENS * 6];
    options.seconds = 5;
    sm_motion walk = sm_generate_motion_ex(model, "a person walks forward", options);
    if (!made(walk)) return false;
    memcpy(codes, walk.codes, sizeof codes);
    sm_unload_motion(walk);

    for (int t = 20; t < 30; t++) codes[6 * t] = -1;   /* 2 s to make again, the rest stays */
    options.pinned_codes = codes;
    options.pinned_token_count = TOKENS;
    sm_motion jump = sm_generate_motion_ex(model, "a person jumps up and down", options);
    if (!made(jump)) return false;
    const bool kept = !memcmp(jump.codes, codes, 20 * 6 * sizeof(int16_t));
    printf("walk, a jump in the middle: %d frames, the first 2 s unchanged: %s\n", jump.frame_count, kept ? "yes" : "no");
    sm_unload_motion(jump);
    return true;
}

static bool make_loop(sm_model model, sm_options options) {
    options.seconds = 2;
    options.loop = true;   /* the end leads into the start: play it on repeat */
    sm_motion run = sm_generate_motion_ex(model, "a person runs forward", options);
    if (!made(run)) return false;
    printf("run cycle: %d frames\n", run.frame_count);
    sm_unload_motion(run);
    return true;
}

/* the wave made again between the kept walk and sit, then the timeline's joints encoded and continued */
static bool edit_timeline(sm_model model, sm_options options) {
    sm_block blocks[3] = {
        {"a person walks forward", NULL, 3, 1, NULL}, {"a person waves", NULL, 2, 2, NULL},
        {"a person sits down", NULL, 3, 3, NULL},
    };
    sm_motion first = sm_generate_timeline(model, blocks, 3, options);
    if (!made(first)) return false;
    blocks[0].kept_codes = first.codes + first.block_start_frames[0] / 2 * 6;
    blocks[2].kept_codes = first.codes + first.block_start_frames[2] / 2 * 6;
    blocks[1].seed = 9;
    sm_motion second = sm_generate_timeline(model, blocks, 3, options);
    sm_unload_motion(first);
    if (!made(second)) return false;
    printf("timeline: blocks at frames %d, %d, %d of %d\n", second.block_start_frames[0], second.block_start_frames[1],
           second.block_start_frames[2], second.block_start_frames[3]);

    bool ok = true;
    if (model.has_encoder) {
        sm_motion encoded = sm_encode_motion(model, second.joints, second.frame_count, options);
        options.previous = &encoded;
        options.seconds = 3;
        sm_motion next = sm_generate_motion_ex(model, "a person stands up and walks away", options);
        ok = made(encoded) && made(next);
        if (ok) printf("encoded %d frames into %d tokens, then went on for %d frames\n", second.frame_count,
                       encoded.token_count, next.frame_count);
        sm_unload_motion(next);
        sm_unload_motion(encoded);
    }
    sm_unload_motion(second);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    sm_model model = sm_load_model(argv[1]);
    const sm_options options = sm_default_options(model);
    const bool ok = made_model(model) && remake_range(model, options) && make_loop(model, options) &&
                    edit_timeline(model, options);
    sm_unload_model(model);
    return !ok;
}
