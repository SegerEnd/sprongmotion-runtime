# Examples

Each builds with `make test` and uses only `sm.h`. Run them with a model file: `01_hello SprongMotion-14M-v1.0-Q8_0.gguf "a person waves" out.bvh`.

- `01_hello.c`: `sm_generate_simple` and `sm_motion_write_bvh`, text to a BVH file (a tiny CLI).
- `02_request.c`: `sm_request_*` parameters, `sm_param_next` / `sm_param_bounds`, variants with `sm_ctx_generate`, error messages.
- `03_steps.c`: the step API (`sm_ctx_begin`, `sm_ctx_step`, `sm_ctx_preview`, `sm_ctx_finish`), progress and cancel.
- `04_edit.c`: pinned codes (`sm_request_set_codes`) to make a range again, a loop (`SM_P_LOOP`), one block of a timeline made again between kept ones (`sm_ctx_generate_blocks`), and with a model file that has the encoder, joints into codes (`sm_ctx_encode_motion`) continued as a prefix.
- `05_wasm.mjs`: the WebAssembly build from JavaScript, in a browser or node: the model loaded from memory, text to BVH.
  `node examples/05_wasm.mjs build/wasm/sprongmotion.wasm SprongMotion-14M-v1.0-Q8_0.gguf "walks forward" > walk.bvh`
