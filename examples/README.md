# Examples

Each builds with `make test` and uses only `sprongmotion.h`. Run them with a model file: `01_hello SprongMotion-14M-v1.0-Q8_0.gguf "a person waves" out.bvh`.

- `01_hello.c`: `sm_generate_motion` and `sm_export_motion_bvh`, text to a BVH file (a tiny CLI).
- `02_options.c`: `sm_default_options` and its fields, one `sm_encode_text` for several seeds, error messages.
- `03_progress.c`: the progress callback, and cancelling from it.
- `04_edit.c`: pinned codes to make a range again, a loop, one block of a timeline made again between kept ones (`sm_generate_timeline`), and with a model file that has the encoder, joints into codes (`sm_encode_motion`) continued as `previous`.
- `05_wasm.mjs`: the WebAssembly build from JavaScript, in a browser or node: the model loaded from memory, text to BVH.
  `node examples/05_wasm.mjs build/wasm/sprongmotion.wasm SprongMotion-14M-v1.0-Q8_0.gguf "walks forward" > walk.bvh`
