/* SprongMotion: text to 3D human motion. C11, no dependencies.
 *
 * QUICK START (the easy path: one call makes and frees its own context and request)
 *   char err[256];
 *   sm_model *m;
 *   if (sm_model_load_file("SprongMotion-14M-v1.0-Q8_0.gguf", NULL, &m, err, sizeof err) != SM_OK)
 *       { puts(err); return 1; }
 *   sm_motion *out; sm_motion_create(m, &out);
 *   if (sm_generate_simple(m, "a person waves with the left hand", 3.0, 7, out, err, sizeof err) == SM_OK) {
 *       uint32_t n; const float *joints = sm_motion_joints(out, &n);  // frames*22*3 floats: metres, Y up, facing +Z
 *   } else puts(err);
 *   sm_motion_free(out); sm_model_free(m);
 *
 * FULL CONTROL: every parameter, variants, the step API and an error message (checks omitted here)
 *   sm_ctx *c;     sm_ctx_create(m, &c);                      // one per thread; the model is shared
 *   sm_request *r; sm_request_create(m, &r);
 *   sm_request_set_text(r, "a person waves with the left hand", -1);
 *   sm_request_set_float(r, SM_P_SECONDS, 3.0);
 *   if (sm_ctx_generate(c, r, &out, 1) != SM_OK) puts(sm_ctx_last_error(c));
 *   sm_request_free(r); sm_ctx_free(c);
 *
 * RULES (every function follows them)
 *   - The library creates and frees every object (x_create / x_free); handles are opaque; freeing NULL is a no-op.
 *     Free every object made from a model before the model. All objects use the allocator and logger given at load.
 *   - Functions return sm_status: SM_OK (0), a positive progress code (SM_DONE), or a negative error. A context or
 *     request passed first keeps a message saying why (sm_ctx_last_error, ...; valid until the next call on it).
 *     Calls on a model, motion or text return only the status (see sm_status_message); load and
 *     sm_generate_simple errors go to err. A rejected value names itself and its range: "seconds = 12.5 is out of
 *     range 0.5 .. 10".
 *   - Not found (an unknown key or name, past the end, no parent, not generated): -1 for integers, NULL for pointers.
 *   - On error, plain outputs are untouched and motions being generated are left empty (0 frames).
 *   - Inputs are copied when set: a request never points into another object (`prefix = out; generate into out`).
 *   - No global state. A model is immutable after load and shared by any number of threads, and so is a const
 *     request. Everything else (context, motion, a request being changed) is one thread at a time; const
 *     getters on a motion may run on many threads while nothing writes to it. sm_ctx_cancel is safe from any thread.
 *   - After sm_ctx_reserve / sm_motion_reserve (or the first call of a size), generation allocates nothing.
 *   - Same model file (sm_model_id) + same request = bit-identical output on every build (native, wasm SIMD, wasm
 *     plain) at any thread count. Exception: a native FAST=1 build (fused multiply-add, ~1.7x faster maths) matches
 *     the others only closely; sm_build_info() says "fma" then.
 *   - Strings are UTF-8 (paths too, also on Windows). Text lengths are in bytes, -1 = NUL-terminated. Numbers in
 *     strings use the C locale ("3.5") whatever the host's locale.
 *   - Integers across the API are 32-bit (wasm, ctypes); counts are elements, not bytes; byte sizes are size_t.
 *   - Units: metres, seconds, radians. Y up, right-handed, the clip starts at the origin facing +Z. 20 frames/s.
 *
 * ABI: functions are only ever added. sm_hooks starts with its size and only grows at the end. Parameters, info keys
 * and statuses are numbers that are only appended. This header is stable within a major version (libsprongmotion.so.1).
 */
#ifndef SM_H
#define SM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SM_SHARED)
#  ifdef SM_BUILD
#    define SM_API __declspec(dllexport)
#  else
#    define SM_API __declspec(dllimport)
#  endif
#elif defined(SM_SHARED)
#  define SM_API __attribute__((visibility("default")))
#else
#  define SM_API
#endif
#if defined(__GNUC__) || defined(__clang__)
#  define SM_DEPRECATED(hint) __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#  define SM_DEPRECATED(hint) __declspec(deprecated(hint))
#else
#  define SM_DEPRECATED(hint)
#endif

/* Version */
#define SM_VERSION_MAJOR 1
#define SM_VERSION_MINOR 0
#define SM_VERSION_PATCH 0
#define SM_VERSION ((SM_VERSION_MAJOR << 16) | (SM_VERSION_MINOR << 8) | SM_VERSION_PATCH)
SM_API uint32_t    sm_version(void);         /* the library's SM_VERSION; the caller needs the same major */
SM_API const char *sm_build_info(void);      /* "1.0.0 native exact", for bug reports */

/* Status */
typedef int32_t sm_status;
#define SM_OK                 0
#define SM_DONE               1   /* sm_ctx_step: the last pass ran; call sm_ctx_finish */
#define SM_E_ARG             -1   /* NULL where not allowed, a value out of range, an unknown parameter */
#define SM_E_FORMAT          -2   /* not a SprongMotion model file, or a malformed one */
#define SM_E_VERSION         -3   /* the file needs a newer runtime */
#define SM_E_CHECKSUM        -4
#define SM_E_NOMEM           -5
#define SM_E_BUFFER          -6   /* the caller's buffer is too small; *needed holds the size */
#define SM_E_CANCELLED       -7
#define SM_E_NUMERIC         -8   /* NaN or Inf met (a broken model file or input) */
#define SM_E_UNSUPPORTED     -9   /* this model file has no such input (e.g. no generator) */
#define SM_E_IO             -10
#define SM_E_STATE          -11   /* out of order: sm_ctx_step without sm_ctx_begin, ... */
#define SM_E_INTERNAL       -99
SM_API const char *sm_status_name(sm_status s);      /* "SM_E_ARG"; static, never NULL */
SM_API const char *sm_status_message(sm_status s);   /* "invalid argument"; static, never NULL */

/* Objects */
typedef struct sm_model    sm_model;     /* weights, tokenizer, skeleton, defaults; immutable, shareable */
typedef struct sm_ctx      sm_ctx;       /* scratch memory, run state, last error; one thread at a time */
typedef struct sm_request  sm_request;   /* what to generate: text, length, seed, sampling, harness inputs */
typedef struct sm_motion   sm_motion;    /* one result: codes, features, joints, rotations; reusable */
typedef struct sm_text     sm_text;      /* an encoded prompt (text vector + word states), reusable */

/* Model */
typedef void (*sm_log_fn)(void *user, int32_t level /* 0 error .. 3 debug */, const char *msg);
typedef void (*sm_task_fn)(void *arg, uint32_t index);
typedef struct sm_hooks {   /* all optional; start from SM_HOOKS_INIT. struct_size 0 or larger than the library knows
                               = SM_E_ARG; fields past struct_size (an older caller) count as zero */
    uint32_t struct_size;
    void *(*alloc)(void *user, size_t size, size_t align);   /* default: aligned malloc */
    void  (*free)(void *user, void *ptr);
    sm_log_fn log;                                          /* default: silent */
    /* run fn(arg, i) for every i in [0, n) and return when all are done; called from one thread per running context,
       so it must be reentrant. Default: the built-in pool (native) or one thread (wasm). The work split is fixed, so
       results never depend on it. Standalone wasm cannot pass C callbacks: the JS module runs its own workers. */
    void (*parallel_for)(void *user, uint32_t n, sm_task_fn fn, void *arg);
    uint32_t n_threads;                                     /* built-in pool size; 0 = the core count */
    void *user;
} sm_hooks;
#define SM_HOOKS_INIT { .struct_size = sizeof(sm_hooks) }

#define SM_LOAD_BORROW  1u   /* sm_model_load_memory: use the caller's bytes in place (they must outlive the model) */
#define SM_LOAD_NOCHECK 2u   /* skip the whole-file checksum (the structure is still fully validated) */
/* hooks: NULL = defaults; copied. err: optional, always NUL-terminated on return. path: UTF-8. A file is
   memory-mapped read-only and used in place, or read into memory if mapping fails. */
SM_API sm_status sm_model_load_file(const char *path, const sm_hooks *hooks, sm_model **out, char *err, size_t err_len);
SM_API sm_status sm_model_load_memory(const void *bytes, size_t size, uint32_t flags, const sm_hooks *hooks,
                                      sm_model **out, char *err, size_t err_len);
SM_API void      sm_model_free(sm_model *m);

typedef int32_t sm_info;   /* integer facts about a model file */
#define SM_I_FPS              1
#define SM_I_FRAMES_PER_TOKEN 2
#define SM_I_MAX_FRAMES       3
#define SM_I_JOINTS           4   /* 22 */
#define SM_I_FEATURE_DIM      5   /* 263 */
#define SM_I_PARAMS_K         6   /* generator parameters in thousands (exact count: meta general.parameter_count) */
#define SM_I_CODE_LAYERS      9   /* codes per token (6) */
#define SM_I_TEXT_DIM        10   /* text vector size (384) */
#define SM_I_ENCODER         11   /* 1 if the file has the motion encoder (sm_ctx_encode_motion), else 0 */
#define SM_I_PATH_INPUT      12   /* 1 if the model itself follows a path (sm_request_set_path), 0 = the root warp
                                     alone: the codes do not depend on the path, so a path edit only needs a decode */
SM_API int32_t     sm_model_info(const sm_model *m, sm_info key);       /* -1 for an unknown key */
SM_API const char *sm_model_meta(const sm_model *m, const char *key);  /* GGUF string metadata ("general.name") */
SM_API const char *sm_model_id(const sm_model *m);   /* "SprongMotion-14M@1a2b3c4d": name + hash; store with codes */
SM_API const char *sm_model_vocab(const sm_model *m, uint32_t i);  /* known whole words, sorted; NULL past end */

/* Skeleton: sprong-22 (the 22-joint skeleton), the rest pose the rotations are relative to. The offsets are the model's
   reference skeleton from the file: each bone's median over clips the model makes, in metres. */
SM_API const char *sm_skeleton_joint_name(const sm_model *m, uint32_t j);  /* "pelvis", "l_hip", ...; NULL past end */
SM_API int32_t     sm_skeleton_parent(const sm_model *m, uint32_t j);      /* -1 for the root and past the end */
SM_API sm_status   sm_skeleton_offsets(const sm_model *m, float *out, uint32_t n_floats);  /* 22*3 floats: each joint's
                                                         rest offset from its parent; n_floats < 66 = SM_E_BUFFER */

/* Context */
SM_API sm_status   sm_ctx_create(const sm_model *m, sm_ctx **out);
SM_API void        sm_ctx_free(sm_ctx *c);
SM_API sm_status   sm_ctx_reserve(sm_ctx *c, double max_seconds, uint32_t max_variants);  /* allocate up front: no hitch later */
SM_API size_t      sm_ctx_memory_bytes(const sm_ctx *c);
SM_API const char *sm_ctx_last_error(const sm_ctx *c);   /* "" if none; valid until the next call on c */
/* any thread; sticky: every run returns SM_E_CANCELLED ("cancel is set: call sm_ctx_clear_cancel") until cleared */
SM_API void        sm_ctx_cancel(sm_ctx *c);
SM_API void        sm_ctx_clear_cancel(sm_ctx *c);       /* between runs, on the context's own thread */
/* runs on the calling thread; nonzero stops this run with SM_E_CANCELLED (the sticky flag stays clear) */
typedef int32_t (*sm_progress_fn)(void *user, uint32_t done, uint32_t total);
SM_API void        sm_ctx_set_progress(sm_ctx *c, sm_progress_fn fn, void *user);  /* NULL to remove */

/* Parameters: one setter per value (zstd style); bounds and defaults come from the model file */
typedef int32_t sm_param;
#define SM_P_SECONDS        1   /* float, clip length; 0.5 .. max_frames / fps */
#define SM_P_SEED           2   /* int, 0 .. 2^31-1; same seed = same motion; variant i uses (seed + i) mod 2^31 */
#define SM_P_STEPS          3   /* int, base-layer unmasking steps */
#define SM_P_RES_LAYERS     4   /* int, residual layers 0..5 (0 = base codes only: a fast draft) */
#define SM_P_CFG            5   /* float, text guidance of the base layer */
#define SM_P_RES_CFG        6   /* float, text guidance of the residual layers */
#define SM_P_TEMP           7   /* float, sampling temperature */
#define SM_P_TOP_K          8   /* int, 0 = off */
#define SM_P_FOOT_LOCK      9   /* int 0/1 (default 1): planted feet stay put and above the floor (leg IK) */
#define SM_P_IN_PLACE      10   /* int 0/1 (default 0): joints, rotations and BVH without the horizontal root travel;
                                   sm_motion_root keeps the full path (the root-motion curve games apply themselves) */
#define SM_P_LOOP          11   /* int 0/1 (default 0): a seamless cycle. Generating, 1.2 s past the end are made as
                                   the start again (and dropped), so the end leads into the start (seconds then up to
                                   the maximum minus 1.2 s); decoding, the decoder's window wraps around. Pinned codes
                                   may be set, a prefix not. Root travel stays: a walk cycle moves on each repeat */
#define SM_P_PATH_WARP     13   /* int 0/1 (default 1): with a path, the root lands on it before the foot lock; 0 = only
                                   sm_motion_path_error. (12 is reserved for path guidance by models that read paths) */
SM_API sm_status   sm_request_create(const sm_model *m, sm_request **out);   /* all parameters at the model's default */
SM_API void        sm_request_free(sm_request *r);
SM_API sm_status   sm_request_clone(const sm_request *r, sm_request **out);
SM_API sm_status   sm_request_reset(sm_request *r);                          /* defaults again, inputs cleared */
SM_API const char *sm_request_last_error(const sm_request *r);
/* set_int on a float parameter converts; set_float on an int one = SM_E_ARG unless whole; get_int on a float one =
   SM_E_ARG */
SM_API sm_status   sm_request_set_int(sm_request *r, sm_param p, int32_t v);
SM_API sm_status   sm_request_set_float(sm_request *r, sm_param p, double v);
SM_API sm_status   sm_request_get_int(const sm_request *r, sm_param p, int32_t *v);
SM_API sm_status   sm_request_get_float(const sm_request *r, sm_param p, double *v);
/* introspection for bindings, UIs and config files; names are stable snake_case ("seconds", "res_cfg") */
SM_API sm_param    sm_param_next(sm_param p);             /* iterate: start at 0, -1 = the end */
SM_API sm_param    sm_param_from_name(const char *name);  /* -1 if unknown */
SM_API const char *sm_param_name(sm_param p);             /* NULL if unknown */
SM_API sm_status   sm_param_bounds(const sm_model *m, sm_param p, double *min, double *max, double *dflt,
                                   int32_t *is_int);
SM_API sm_status   sm_request_set_by_name(sm_request *r, const char *name, const char *value);   /* "steps", "10" */
SM_API const char *sm_api_schema_json(void);   /* names, types, units and the runtime's ranges; constant */
SM_API const char *sm_model_schema_json(const sm_model *m);  /* the same with this model's ranges; lives with m */

/* Inputs, all copied at the call */
SM_API sm_status sm_request_set_text(sm_request *r, const char *utf8, int32_t len);
SM_API sm_status sm_request_set_text_encoded(sm_request *r, const sm_text *t);   /* one encoding for many seeds */
/* chaining: the new clip starts from the last 3 tokens (0.3 s) of prev and continues it. The result holds only the
   new SM_P_SECONDS (at most the model's maximum minus 0.3 s). Its codes follow prev's: both decoded in one
   sm_ctx_decode_codes call are as smooth at the junction as inside a clip. Its own joints continue prev's in prev's
   frame, fading in from prev's last pose over 0.4 s: playable after prev's, but rougher at the junction than one
   decode. prev needs at least 3 tokens; NULL clears. */
SM_API sm_status sm_request_set_prefix(sm_request *r, const sm_motion *prev);

/* pinned codes: tokens*6 in sm_motion_codes' layout, for the first n_tokens tokens of the clip (at most its length):
   a token whose base code (layer 0) is -1 is generated, any other stays exactly as given on all 6 layers (residual
   -1 = none) and the rest is generated around it (masked models in-paint). One array, so codes read from a motion can
   be edited and passed back: regenerate a range (-1 there, both ends pinned), lock a block, end a clip on a pose.
   Up to 512 tokens; n_tokens 0 clears. */
SM_API sm_status sm_request_set_codes(sm_request *r, const int16_t *codes, uint32_t n_tokens);

/* a root path, copied: n points of 4 floats (t, x, z, heading) in the output's frame (seconds, metres, radians; heading
   0 = facing +Z, pi/2 = facing +X). For sm_ctx_generate_blocks t is timeline time; with a prefix the frame is the
   previous clip's. Every call that decodes uses it (generate, preview, blocks, decode_codes, encode_motion). The root
   starts at the first point and follows a centripetal Catmull-Rom curve through all of them (the first point again at
   the end closes it). Its body keeps the prompt's facing relative to the direction of travel, so a sidestep stays a
   sidestep. t: when the root passes the point, rising, within the clip (checked when it runs); NaN = free, at the
   motion's own speed (with no times the clip's travel is stretched to the curve's length, so pick seconds for the gait:
   sm_motion_speed). heading: NaN, or a facing for models that read the path (SM_I_PATH_INPUT); the warp keeps the
   prompt's. Works best within what the motion already does: about +-10 % of its speed, turns gentler than 45 deg/s
   added, radius 4 m and up at a walk. 2 <= n <= 1024, |x|, |z|, |heading| <= 1000, t <= 600; n = 0 clears. */
SM_API sm_status sm_request_set_path(sm_request *r, const float *points, uint32_t n);

/* Text encoding */
SM_API sm_status sm_text_create(const sm_model *m, sm_text **out);
SM_API void      sm_text_free(sm_text *t);
SM_API sm_status sm_ctx_encode_text(sm_ctx *c, const char *utf8, int32_t len, sm_text *out);
SM_API sm_status sm_text_vector(const sm_text *t, const float **v, uint32_t *dim);   /* borrowed while t lives: the pooled text vector */
SM_API sm_status sm_text_set_vector(sm_text *t, const float *v, uint32_t dim);        /* your own vector (no words) */
/* the words the model does not know (they become an unknown token), one per line, NUL-terminated; *n = how many.
   For UIs: "not understood: pirouette". *needed = bytes incl. the NUL; SM_E_BUFFER if cap is too small (buf NULL =
   only size it). */
SM_API sm_status sm_text_unknown_words(const sm_text *t, char *buf, size_t cap, size_t *needed, uint32_t *n);

/* Generate. outs[i] receives variant i (seed + i), up to 16; the text is encoded once for all of them. */
SM_API sm_status sm_motion_create(const sm_model *m, sm_motion **out);
SM_API void      sm_motion_free(sm_motion *mo);
SM_API sm_status sm_motion_reserve(sm_motion *mo, double max_seconds);
SM_API sm_status sm_motion_copy(sm_motion *dst, const sm_motion *src);
SM_API sm_status sm_ctx_generate(sm_ctx *c, const sm_request *r, sm_motion *const *outs, uint32_t n_variants);
/* step API: the result of sm_ctx_generate, one transformer pass per sm_ctx_step (UI loops, previews, cancel points).
   sm_ctx_begin copies r; outs must stay alive until sm_ctx_finish or sm_ctx_abort. Loop:
   while ((s = sm_ctx_step(c, &d, &t)) == SM_OK) {...}, then s == SM_DONE: sm_ctx_finish.
   sm_ctx_preview decodes the current codes into outs (final at the end). */
SM_API sm_status sm_ctx_begin(sm_ctx *c, const sm_request *r, sm_motion *const *outs, uint32_t n_variants);
SM_API sm_status sm_ctx_step(sm_ctx *c, uint32_t *done, uint32_t *total);
SM_API sm_status sm_ctx_preview(sm_ctx *c);
SM_API sm_status sm_ctx_finish(sm_ctx *c);
SM_API void      sm_ctx_abort(sm_ctx *c);                                     /* end a run; its outs are left empty */
/* stored codes (sm_motion_codes) back to a motion; r gives SM_P_FOOT_LOCK, SM_P_IN_PLACE and SM_P_LOOP (NULL = the
   defaults) */
SM_API sm_status sm_ctx_decode_codes(sm_ctx *c, const sm_request *r, const int16_t *codes, uint32_t n_tokens,
                                     sm_motion *out);
/* existing motion to codes: joints n_frames*22*3 floats (metres, Y up, 20 frames/s, any start and facing: they are
   moved onto the floor at the origin facing +Z and onto the model's skeleton by leg length). out holds
   (n_frames - 1) / 2 tokens and their decode (r as in sm_ctx_decode_codes): use it as a prefix to continue it, or its
   codes as pinned codes to stitch two motions or fix a range. 3 <= n_frames <= 12001. SM_E_UNSUPPORTED if the model
   file has no encoder (sm_model_info SM_I_ENCODER). */
SM_API sm_status sm_ctx_encode_motion(sm_ctx *c, const sm_request *r, const float *joints, uint32_t n_frames,
                                      sm_motion *out);

/* Timeline: prompt blocks back to back in one motion, the host holding all state in plain arrays (n_blocks each).
   Block i is texts[i] (NUL-terminated) or, where texts or texts[i] is NULL, encs[i]; seconds[i] long (with its seams
   within the model's maximum); seeds[i]. keep[i]: its codes from an earlier call (seconds[i] * fps / frames_per_token
   tokens, 6 codes each), kept as they are; keep or keep[i] NULL: the block is made. Made blocks go left to right,
   their seams pinned to the 0.3 s of each neighbour already final (kept, or made earlier in this call); then the whole
   timeline is decoded in one pass into out, whose codes hold every block's codes back to back. first_frames (n_blocks
   + 1 entries, may be NULL) receives where each block starts in out, and the end. Edit a block: pass the others'
   codes as keep and call again; they stay exactly. r gives the sampling and output parameters (its text, seconds,
   seed, prefix and codes are not used); SM_P_LOOP closes the timeline into a cycle. Progress and cancel work as for
   sm_ctx_generate; no step API (a call is a run per made block). 1 <= n_blocks <= 256. */
SM_API sm_status sm_ctx_generate_blocks(sm_ctx *c, const sm_request *r, uint32_t n_blocks, const char *const *texts,
                                        const sm_text *const *encs, const double *seconds, const int32_t *seeds,
                                        const int16_t *const *keep, uint32_t *first_frames, sm_motion *out);
/* the easy path: a temporary context and request, the model's defaults; text NUL-terminated. err: optional, always
   NUL-terminated on return */
SM_API sm_status sm_generate_simple(const sm_model *m, const char *utf8, double seconds, int32_t seed, sm_motion *out,
                                    char *err, size_t err_len);

/* Results: borrowed, valid until the motion is regenerated or freed. *n = element count (may be NULL). */
SM_API double         sm_motion_seconds(const sm_motion *mo);
SM_API uint32_t       sm_motion_frames(const sm_motion *mo);
SM_API int32_t        sm_motion_seed(const sm_motion *mo);   /* -1 if not generated (decoded codes); copied */
SM_API const float   *sm_motion_joints(const sm_motion *mo, uint32_t *n);     /* frames*22*3 floats: positions, m */
/* frames*22*4 floats, x y z w (glTF order): each joint's rotation relative to its parent, turning its children's rest
   offsets (as BVH and glTF do); joint 0 = the pelvis. They aim every bone at its joint, the pelvis and chest by their
   hip and shoulder lines. FK of them lands, as a mean over joints and frames, 9 mm from the joints on the reference
   skeleton (sm_skeleton_offsets, for retargeting) for the median clip and up to ~70 mm on clips whose hips sit
   differently (the training data mixes skeleton conventions); on the clip's own skeleton (sm_motion_offsets, the BVH)
   8 mm for the median clip, up to ~55 mm. The rest is the joints' own bones changing length within a clip, which no
   fixed skeleton follows. */
SM_API const float   *sm_motion_rotations(const sm_motion *mo, uint32_t *n);
/* 22*3 floats: this clip's own skeleton, each bone's median offset in its parent's frame over the clip's frames (the
   reference directions, the clip's bone lengths); the BVH hierarchy uses it */
SM_API const float   *sm_motion_offsets(const sm_motion *mo, uint32_t *n);
SM_API const float   *sm_motion_root(const sm_motion *mo, uint32_t *n);       /* frames*3 floats */
/* frames floats, m/s: the model's own root speed (the root smoothed over 0.25 s, before a path warp). The mean over a
   block's moving frames is its gait speed: seconds = path length / gait speed fits a block to a path */
SM_API const float   *sm_motion_speed(const sm_motion *mo, uint32_t *n);
/* m: the largest distance of the smoothed root from where the path puts it at that time, before the warp (the warp's
   size); -1 without a path */
SM_API float          sm_motion_path_error(const sm_motion *mo);
SM_API const uint8_t *sm_motion_contacts(const sm_motion *mo, uint32_t *n);   /* frames*4 on the ground: l_ankle,
                                                                                 l_foot, r_ankle, r_foot */
SM_API const float   *sm_motion_features(const sm_motion *mo, uint32_t *n);   /* frames*263 floats */
SM_API const int16_t *sm_motion_codes(const sm_motion *mo, uint32_t *n);      /* tokens*6: store with sm_model_id */
SM_API const float   *sm_motion_confidence(const sm_motion *mo, uint32_t *n); /* tokens floats 0..1: certainty */
/* BVH text into buf, with the clip's own skeleton (sm_motion_offsets); *needed = bytes incl. the NUL; SM_E_BUFFER if
   cap is too small (buf NULL = only size it). scale: 100 = centimetres (the usual BVH unit), 1 = metres. glTF export
   lives in the JS and Python layers. */
SM_API sm_status      sm_motion_write_bvh(const sm_motion *mo, double scale, char *buf, size_t cap, size_t *needed);

#ifdef __cplusplus
}
#endif
#endif /* SM_H */
