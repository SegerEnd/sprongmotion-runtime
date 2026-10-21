/* Change a motion without starting over: regenerate a range, make a loop, edit one block of a timeline, and
   (with a model file that has the motion encoder) continue motion that came from elsewhere. Usage: 04_edit model.gguf */
#include <stdio.h>
#include <string.h>
#include "sm.h"

static int failed(sm_status s, sm_ctx *c) { return s != SM_OK && fprintf(stderr, "%s\n", sm_ctx_last_error(c)); }

static int run(sm_model *m, sm_ctx *c, sm_request *r, sm_motion *a, sm_motion *b) {
    int16_t codes[50 * 6];
    sm_request_set_text(r, "a person walks forward", -1);
    sm_request_set_float(r, SM_P_SECONDS, 5);
    if (failed(sm_ctx_generate(c, r, &a, 1), c)) return 1;
    memcpy(codes, sm_motion_codes(a, NULL), sizeof codes);   /* 50 tokens (2 frames each), 6 codes per token */
    for (int t = 20; t < 30; t++) codes[6 * t] = -1;          /* 2 s to make again, the rest stays */
    sm_request_set_text(r, "a person jumps up and down", -1);
    sm_request_set_codes(r, codes, 50);
    if (failed(sm_ctx_generate(c, r, &b, 1), c)) return 1;
    printf("walk, a jump in the middle: %u frames, the first 2 s unchanged: %s\n", sm_motion_frames(b),
           memcmp(sm_motion_codes(b, NULL), codes, 20 * 6 * sizeof(int16_t)) ? "no" : "yes");

    sm_request_set_codes(r, NULL, 0);
    sm_request_set_text(r, "a person runs forward", -1);
    sm_request_set_float(r, SM_P_SECONDS, 2);
    sm_request_set_int(r, SM_P_LOOP, 1);   /* the end leads into the start: play it on repeat */
    if (failed(sm_ctx_generate(c, r, &a, 1), c)) return 1;
    printf("run cycle: %u frames\n", sm_motion_frames(a));
    sm_request_set_int(r, SM_P_LOOP, 0);

    const char *texts[3] = {"a person walks forward", "a person waves", "a person sits down"};
    const double seconds[3] = {3, 2, 3};
    int32_t seeds[3] = {1, 2, 3};
    uint32_t first[4];
    if (failed(sm_ctx_generate_blocks(c, r, 3, texts, NULL, seconds, seeds, NULL, first, a), c)) return 1;
    const int16_t *all = sm_motion_codes(a, NULL);
    const int16_t *keep[3] = {all + first[0] / 2 * 6, NULL, all + first[2] / 2 * 6};   /* keep blocks 0 and 2 */
    seeds[1] = 9;                                                                        /* another wave */
    if (failed(sm_ctx_generate_blocks(c, r, 3, texts, NULL, seconds, seeds, keep, first, b), c)) return 1;
    printf("timeline: blocks at frames %u, %u, %u of %u; the wave made again between the kept blocks\n", first[0],
           first[1], first[2], first[3]);

    if (!sm_model_info(m, SM_I_ENCODER)) return 0;
    uint32_t n;   /* joints from anywhere (here the timeline's), turned into codes, then continued */
    const float *joints = sm_motion_joints(b, &n);
    if (failed(sm_ctx_encode_motion(c, r, joints, n / 66, a), c)) return 1;
    sm_request_set_prefix(r, a);
    sm_request_set_text(r, "a person stands up and walks away", -1);
    sm_request_set_float(r, SM_P_SECONDS, 3);
    if (failed(sm_ctx_generate(c, r, &b, 1), c)) return 1;
    printf("encoded %u frames into %u tokens, then went on for %u frames\n", n / 66, sm_motion_frames(a) / 2,
           sm_motion_frames(b));
    return 0;
}

int main(int argc, char **argv) {
    char err[256];
    sm_model *m;
    if (argc != 2) return fprintf(stderr, "usage: %s model.gguf\n", argv[0]), 2;
    if (sm_model_load_file(argv[1], NULL, &m, err, sizeof err) != SM_OK) return fprintf(stderr, "%s\n", err), 1;
    sm_ctx *c;
    sm_request *r;
    sm_motion *a, *b;
    sm_ctx_create(m, &c);
    sm_request_create(m, &r);
    sm_motion_create(m, &a);
    sm_motion_create(m, &b);
    const int bad = run(m, c, r, a, b);
    sm_motion_free(a);
    sm_motion_free(b);
    sm_request_free(r);
    sm_ctx_free(c);
    sm_model_free(m);
    return bad;
}
