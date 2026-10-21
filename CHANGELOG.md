# Changelog

## 1.0.1 (2026-10-21)

Faster, with the same output: codes, features and joints stay bit-identical to 1.0.0 on every exact build at any
thread count.

- The matrix weights are dequantised once at load into f32 panels instead of on every pass: about 6 % less time per
  timeline block and 2 to 3 % per 10 s clip, native and WebAssembly. A model takes 57 MB more memory for it.
- `docs/design.md` lists the faster options that would change the output, with their measured gains.

## 1.0.0 (2026-10-21)

First release. Loads SprongMotion-14M 1.0 (`SprongMotion-14M-v1.0-Q8_0.gguf`, model file format 2).

- C11 library with no dependencies: static and shared builds for macOS, Linux and Windows, and standalone WebAssembly
  (SIMD128 and plain).
- Text to motion: `sm_generate_simple`, or contexts and requests with every sampling parameter, variants, and the step
  API with progress, previews and cancel.
- Editing: pinned codes, in-painting a range, loops, timelines of prompt blocks (`sm_ctx_generate_blocks`), joints into
  codes with the motion encoder (`sm_ctx_encode_motion`).
- Paths: `sm_request_set_path` warps the root onto a curve, with planted feet keeping their yaw. `sm_motion_speed`.
- Outputs: joints, rotations, offsets, root, contacts, features, codes, confidence, BVH. Foot lock and in place.
- Same model file and same request give bit-identical output on every exact build (native, wasm SIMD, wasm plain) at
  any thread count. `FAST=1` trades that for fused multiply-add speed.
- `make dist` and `make dist-wasm` package a release per platform.
