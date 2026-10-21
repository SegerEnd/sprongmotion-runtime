/* Text to a BVH file in one call. Usage: 01_hello model.gguf "a person waves" out.bvh */
#include <stdio.h>
#include <stdlib.h>
#include "sm.h"

static int write_bvh(const sm_motion *mo, const char *path) {
    size_t size = 0;
    sm_motion_write_bvh(mo, 100, NULL, 0, &size);   /* a NULL buffer only sizes it; 100 = centimetres */
    char *bvh = malloc(size);
    FILE *f = bvh ? fopen(path, "w") : NULL;
    if (f) {
        sm_motion_write_bvh(mo, 100, bvh, size, NULL);
        fputs(bvh, f);
        fclose(f);
    }
    free(bvh);
    return f != NULL;
}

int main(int argc, char **argv) {
    if (argc != 4) return fprintf(stderr, "usage: %s model.gguf \"a person waves\" out.bvh\n", argv[0]), 2;
    char err[256];
    sm_model *m;
    if (sm_model_load_file(argv[1], NULL, &m, err, sizeof err) != SM_OK) return fprintf(stderr, "%s\n", err), 1;
    sm_motion *mo;
    sm_motion_create(m, &mo);
    int ok = sm_generate_simple(m, argv[2], 4.0, 1, mo, err, sizeof err) == SM_OK;
    if (!ok) fprintf(stderr, "%s\n", err);
    else if (!(ok = write_bvh(mo, argv[3]))) fprintf(stderr, "cannot write %s\n", argv[3]);
    else printf("%s: %u frames at 20 fps\n", argv[3], sm_motion_frames(mo));
    sm_motion_free(mo);
    sm_model_free(m);
    return !ok;
}
