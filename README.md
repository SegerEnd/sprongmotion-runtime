# SprongMotion runtime

Text to 3D human motion in plain C11, with no dependencies. It runs [SprongMotion-14M](https://huggingface.co/SegerEnd/SprongMotion-14M)
on a CPU or in the browser: a prompt in, a 22-joint skeleton clip out (20 fps, up to 10 s), with editing, timelines of
prompt blocks and paths. A 10 s clip takes 0.23 s on an Apple M4, 1.4 s in WebAssembly on one thread
([speed](docs/design.md#speed)). No GPU needed.

## Quick start

```sh
hf download SegerEnd/SprongMotion-14M SprongMotion-14M-v1.0-Q8_0.gguf --local-dir .
make build/examples/01_hello
build/examples/01_hello SprongMotion-14M-v1.0-Q8_0.gguf "waves with the right hand" wave.bvh
```

In C:

```c
sm_model model = sm_load_model("SprongMotion-14M-v1.0-Q8_0.gguf");
sm_motion motion = sm_generate_motion(model, "waves with the left hand", 3.0f, 7);
if (!sm_export_motion_bvh(motion, "wave.bvh")) puts(sm_get_error());
sm_unload_motion(motion);
sm_unload_model(model);
```

In JavaScript, with the WebAssembly build ([examples/05_wasm.mjs](examples/05_wasm.mjs), browser or node):

```js
import { load } from "./05_wasm.mjs";
const generate = await load("sprongmotion.wasm", "SprongMotion-14M-v1.0-Q8_0.gguf");
const bvh = generate("walks forward", 4);
```

The API is documented in [include/sprongmotion.h](include/sprongmotion.h), the design (architecture, model file, determinism, speed) in
[docs/design.md](docs/design.md). [examples/](examples/) has small programs from text to BVH up to editing.

## Build

```sh
make lib            # build/libsprongmotion.a and the shared library
make test           # unit tests and examples (SM_MODEL_DIR=dir with the model file for the model tests)
make wasm           # build/wasm/sprongmotion.wasm (SIMD128) and sprongmotion-plain.wasm, needs emcc
make wasm-test      # the tests built with emcc, run in node
make dist           # build/dist/sprongmotion-<version>-<os>-<arch>.tar.gz
make dist-wasm      # build/dist/sprongmotion-<version>-wasm.tar.gz
```

Only a C11 compiler and make are needed. `FAST=1` builds with fused multiply-add (faster, not bit-identical to other
builds).

Some tests compare against reference tensors in `tests/data/`, which is not public. Without it,
`make symbols build/test_gemm build/test_math && build/test_gemm && build/test_math` runs the tests that need no data.

## Platforms

| Platform | Build | Package |
|---|---|---|
| macOS arm64 | clang, `make dist` | `libsprongmotion.dylib`, `libsprongmotion.a` |
| Linux x86-64 | gcc or clang, `make dist` | `libsprongmotion.so`, `libsprongmotion.a` |
| Windows x86-64 | MinGW-w64 gcc (MSYS2), `make dist` | `libsprongmotion.dll`, import library `libsprongmotion.dll.a`, `libsprongmotion.a` |
| WebAssembly | emcc, `make dist-wasm` | `sprongmotion.wasm` (SIMD128), `sprongmotion-plain.wasm`: standalone, exports `sm_*`, `malloc`, `free`, `memory` |

Each package holds `include/sprongmotion.h`, the libraries, LICENSE, NOTICE, this README and CHANGELOG.md. The header is
stable from 1.1 on within major version 1: functions are only added and structs only grow at the end.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The model has its own license and a request on use: see the
[model card](https://huggingface.co/SegerEnd/SprongMotion-14M).
