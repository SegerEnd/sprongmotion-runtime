# Design

How the runtime is built and why. The API itself is documented in `include/sprongmotion.h`.

## Architecture

SprongMotion-14M is a masked token model over a residual VQ motion tokenizer, the design of MoMask (Guo et al., "MoMask:
Generative Masked Modeling of 3D Human Motions", CVPR 2024). The runtime runs four parts, all from one model file:

1. **Tokenizer.** CLIP's byte-pair encoding (Radford et al. 2021), lowercased, at most 40 tokens with the start and end
   tokens, then remapped to the model's own vocabulary (unknown words map to one UNK id).
2. **Text encoder.** Two pre-norm transformer layers (width 256, 4 heads, feed-forward 1024) over the token embeddings
   and a learned position table. It gives the word states (one per token) and a pooled 384-d text vector.
3. **Generator.** Six post-norm transformer layers (width 384, 6 heads of 64, feed-forward 1536, exact-erf GELU) shared
   by two heads:
   - the masked head makes the base code layer. The sequence is `[text vector, word states, motion tokens]`. It
     starts fully masked and unmasks over `steps` passes with a cosine schedule (MaskGIT, Chang et al. 2022): each pass
     samples every masked position and re-masks the least confident ones;
   - the residual head makes code layers 1 to 5, one pass each, conditioned on the sum of the layers below.
   Every pass runs a conditional and an unconditional sequence (classifier-free guidance, Ho and Salimans 2022), with
   top-k filtering, temperature and Gumbel-max sampling.
4. **Motion decoder.** A 1D convolutional VQ decoder: the summed codebook vectors of the 6 layers, three dilated residual
   blocks (dilation 9, 3, 1), a 2x nearest upsample, then 263 features per frame. One token is 2 frames at 20 fps. The
   optional motion encoder is the mirror image (a stride-2 convolution down, the same residual blocks, then a residual
   quantizer with nearest-code search) and turns joints back into codes.

The 263 features per frame follow Guo et al., "Generating Diverse and Natural 3D Human Motions From Text" (CVPR 2022):
root angular velocity, root velocity on the ground, root height, local joint positions, 6D joint rotations, local
velocities and four foot contacts, on a 22-joint skeleton. The runtime recovers joint positions from them in double
precision, then joint rotations by analytic inverse kinematics (two-direction frames for the pelvis and chest, the
shortest arc for every other bone), then optional foot locking (leg IK on the contact flags) and BVH.

Paths warp the generated root onto a curve (centripetal Catmull-Rom, walked by arc length) and turn the body with it.
Timelines, loops and in-painting pin codes: a masked model fills in whatever is not pinned, so a block between two kept
blocks is made with both of its seams fixed.

The tests measure both against a gate: jerk and foot slide at a seam, or on a warped path, within +20 % of the inside
of an unedited clip. Paths pass on straight lines, gentle S curves and 4 m circles at a walk: tight circles, corners and
zigzags slide the feet. In-painting between kept codes passes. Seams between blocks with different prompts are above
the gate with SprongMotion-14M 1.0 (the tests print the numbers).

Where the work is: the generator's four matrix products per layer are about 90 % of the time. Attention, the heads, the
decoder and the text encoder are small. So the speed work is one matrix kernel.

## Source files

| File | Holds |
|---|---|
| `src/sm_gguf.c` | the GGUF reader and the CRC-32 |
| `src/sm_bind.c` | binding a model file: tokenizer tables, decoder, skeleton, settings, metadata |
| `src/sm_model.c` | loading a model, file mapping, the settings, errors, the allocator, the model's spare contexts |
| `src/sm_tokenizer.c` | the CLIP tokenizer and its text cleanup |
| `src/sm_layers.c` | the matrix kernel, layer norm, attention, transformer layers, the VQ decoder and encoder |
| `src/sm_generator.c` | binding the text encoder and generator, and their passes |
| `src/sm_sampling.c` | the sampling noise, top-k picks, the mask schedule, exact `exp`, `log` and float16 rounding |
| `src/sm_context.c` | contexts, text encoding, generating, decoding and encoding motions |
| `src/sm_timeline.c` | prompt blocks on a timeline |
| `src/sm_options.c` | the option table, the defaults, options checked into a request |
| `src/sm_pose.c` | features to joints, rotations, the clip's skeleton, foot contacts and foot lock |
| `src/sm_motion.c` | motions and BVH |
| `src/sm_features.c` | joints to features, for the motion encoder |
| `src/sm_path.c` | root paths and the path warp |
| `src/sm_threads.c` | the built-in thread pool |
| `src/sm_internal.h` | the types and functions shared between them |

## Model file

The model is a GGUF v3 file (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): little-endian, a key-value
header, a tensor table and aligned data. The runtime has its own validating reader and does not link ggml. Matrices of
64 x 64 or more are Q8_0 (32 int8 values with one fp16 scale), the rest F32. At load the matrices of the matrix kernel
are dequantised once into f32 panels of 16 columns (57 MB for SprongMotion-14M), so the arithmetic is f32 (weights
8-bit in the file, activations 32-bit) and a pass reads its weights without converting them again.

Keys the runtime reads:

| Key | Meaning |
|---|---|
| `general.architecture` | `sprongmotion` |
| `general.name`, `general.version`, `general.license`, ... | free metadata, readable with `sm_get_model_meta` |
| `sprongmotion.format_version` | the schema version (2). A runtime reads files up to its own version |
| `sprongmotion.data_crc32` | CRC-32 of the data section, checked at load. `model.id` is `general.name@crc` |
| `sprongmotion.fps`, `.frames_per_token`, `.max_frames` | 20, 2, 200 |
| `sprongmotion.tokenizer.max_len`, `.sot`, `.eot` | 40, 49406, 49407 |
| `sprongmotion.skeleton.name` | the skeleton's name. Parents and offsets are tensors |
| `sprongmotion.defaults.<key>` (`.min`, `.max`) | an option's default and the range the model allows; keys `seconds`, `seed`, `steps`, `res_layers`, `cfg`, `res_cfg`, `temp`, `top_k`, `foot_lock`, `in_place`, `loop`, `path_warp` |
| `sprongmotion.vq.encoder` | 1 when the file has the motion encoder |

Tensors, by name (shapes innermost first):

| Prefix | Holds |
|---|---|
| `tok.*` | BPE merges as id pairs, the kept vocabulary, byte and Unicode class tables, normalisation tables |
| `text.*` | embedding (Q8_0, with a low-rank up-projection), positions, `text.L{0,1}.*` layers, final norm, pooled output |
| `trunk.L{0..5}.*` | the shared generator layers: `attn.qkv`, `attn.out`, `ff1`, `ff2`, `ln1`, `ln2` |
| `mask.*` | the masked head: text vector and word projections, token table, output dense, norm and logits |
| `res.*` | the residual head: text vector and layer-index projections, per-layer input and output tables |
| `pe` | the sinusoidal position table of the generator (100 tokens = 10 s) |
| `vq.*` | codebooks (6 x 512 x 256), feature mean and std, decoder convolutions, and `vq.enc.*` (optional encoder) |
| `skeleton.*` | the reference skeleton: parents and offsets in metres |

Convolution weights are stored ready for im2col, so the decoder and encoder run on the same matrix kernel as the
transformer. A matrix whose row count is not a multiple of 16 is padded with zero rows.

Every value is checked before use: sizes and offsets with overflow checks, tensor names and exact shapes, finite floats
and scales, consistent tokenizer tables. A bad file returns an error and a message, never a crash.

## Threads and determinism

Same model file, same request and same seed give bit-identical codes, features and joints on every exact build
(native, WebAssembly SIMD, WebAssembly plain) at any thread count. What makes that hold:

- Built with `-ffp-contract=off` and no fast-math: no fused multiply-add, which rounds differently between targets.
- Own `exp`, `log` and `erf` (polynomials) instead of the platform's libm, whose last bits differ between platforms.
- The vector and scalar code paths add in the same per-lane order (GCC/Clang vector extensions, which lower to the same
  scalar order where there is no SIMD).
- A reduction is never split across threads. Threads own output rows or column panels and take the next task from a
  counter, so which thread does which task does not change the result.
- The sampling noise is a counter-based hash of (seed, pass, position, code), not a stream, so it does not depend on
  evaluation order.
- The mask schedule is computed in double with round-half-even, and NaN or Inf logits fail the call.

`make FAST=1` turns fused multiply-add on for native builds: about 1.7x faster, the same output at any thread count within
that build, close to but not bit-identical with the others. `sm_get_build_info()` says "fma" then.

A model starts its own thread pool at load (`sm_set_thread_count`, 0 = the core count), or uses the host's
`sm_set_parallel_for`, so a game engine or Blender can run the work on its own job system. WebAssembly builds are
single-threaded unless the host gives `parallel_for`.

## Speed

Measured on a Mac mini with an Apple M4 (4 performance and 6 efficiency cores, 16 GB), `SprongMotion-14M-v1.0-Q8_0.gguf`,
the default 10 base steps and 5 residual layers, the exact build, best of 10 runs of `make bench`, from text to joints:

| Threads | 2 s clip | 5 s clip | 10 s clip | one block in a 6 x 2 s timeline |
|---|---|---|---|---|
| 10 (the core count) | 56 ms | 115 ms | 220 ms | 70 ms |
| 4 | 79 ms | 173 ms | 340 ms | 101 ms |
| 1 | 280 ms | 620 ms | 1211 ms | 361 ms |
| WebAssembly SIMD, 1 (node 26) | 319 ms | 707 ms | 1380 ms | 411 ms |

A 2 s clip at 10 threads: text 0.2 ms, a base step 3.8 ms, a residual layer 3.1 ms, decoding 1.4 ms. `FAST=1` is about
1.7x faster in the matrix kernel.

The exact matrix kernel runs at about 90 % of what one core can do without fused multiply-add, so faster than this
means changing the output. Measured on the same machine (1.0.1):

| Option | Gain |
|---|---|
| `FAST=1` (fused multiply-add) | 10 s clip 1211 -> 709 ms at 1 thread, 220 -> 171 ms at 10 |
| int8 activations with int8 dot products (`sdot`) | the matrix kernel 4.3x on a 216 x 384 x 1536 product, a rough prototype |
| 6 base steps instead of 10 | 10 s clip 248 -> 187 ms, a timeline block 78 -> 58 ms (10 threads) |
| 3 residual layers instead of 5 | 10 s clip 248 -> 217 ms, a timeline block 78 -> 68 ms (10 threads) |

## API principles

- **Values with fields.** `sm_model`, `sm_motion` and `sm_text` are small structs returned by value; their fields are
  the data (`motion.joints`, `model.fps`) and one `internal` pointer holds the rest. Load and unload pairs free them.
- **A simple call and an `_ex` call.** `sm_generate_motion(model, prompt, seconds, seed)` takes plain arguments;
  `sm_generate_motion_ex` takes an `sm_options` struct that `sm_default_options` fills from the model file. Every
  setting is a named field with its range in the header, and options are values: copy one, change a field.
- **No status codes.** A failed call returns an empty value and `sm_get_error()` says why, per thread, naming the
  rejected field and its range ("seconds = 12.5 is out of range 0.5 .. 10"). A call given an empty value fails with the
  error that made it, so a chain of calls can be checked once at the end.
- **A model is shared.** Any number of threads may call with one model at once. Each call borrows scratch memory from a
  few contexts the model keeps, so scratch is reused without the caller holding any.
- **Process settings are functions.** `sm_set_memory_functions`, `sm_set_thread_count` and `sm_set_parallel_for` apply
  to models loaded after them.
- **Edits say what the user means.** `previous` continues a motion, `loop` makes a cycle, `kept_codes` keep blocks on a
  timeline, pinned codes use the layout of `motion.codes`. The 2-frames-per-token detail stays inside the library.
- **The ABI only grows.** Exported symbols are exactly the `SM_API` declarations of `sprongmotion.h` (`make test`
  checks it). Within major version 1 functions are only added and structs only grow at the end. In WebAssembly a struct
  is passed and returned through memory (a pointer to it, the result's pointer first); `examples/05_wasm.mjs` shows the
  layout.
