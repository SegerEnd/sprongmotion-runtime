// Text to BVH with the WebAssembly build, in a browser or node. Usage: node 05_wasm.mjs sprongmotion.wasm model.gguf "waves"
const read = async (url) =>
  typeof process == "object" ? (await import("node:fs")).readFileSync(url) : new Uint8Array(await (await fetch(url)).arrayBuffer());

// wasm32 passes and returns these structs through memory: sm_model is 52 bytes, sm_motion 68 (frame_count at 0)
const MODEL_SIZE = 52, MOTION_SIZE = 68;

export async function load(wasmUrl, modelUrl) {
  const stub = () => 0;
  const { instance } = await WebAssembly.instantiate(await read(wasmUrl), {
    env: { emscripten_notify_memory_growth: stub },
    wasi_snapshot_preview1: { fd_close: stub, fd_write: stub, fd_read: stub, fd_seek: stub },
  });
  const sm = instance.exports;
  sm._initialize();
  const bytes = () => new Uint8Array(sm.memory.buffer);
  const cstr = (s) => {
    const b = new TextEncoder().encode(s + "\0"), p = sm.malloc(b.length);
    bytes().set(b, p);
    return p;
  };
  const text = (p) => new TextDecoder().decode(bytes().subarray(p, bytes().indexOf(0, p)));
  const fail = () => {
    throw new Error(text(sm.sm_get_error()));
  };

  const file = await read(modelUrl), data = sm.malloc(file.length);
  bytes().set(file, data);
  const model = sm.malloc(MODEL_SIZE), motion = sm.malloc(MOTION_SIZE);
  sm.sm_load_model_from_memory(model, data, file.length);   // copies the bytes
  sm.free(data);
  if (!sm.sm_is_model_valid(model)) fail();

  return function generate(prompt, seconds = 4, seed = 1) {
    const p = cstr(prompt);
    sm.sm_generate_motion(motion, model, p, seconds, seed);
    sm.free(p);
    if (!sm.sm_is_motion_valid(motion)) fail();
    const bvh = sm.sm_export_motion_bvh_text(motion, 100);   // 100 = centimetres
    sm.sm_unload_motion(motion);
    if (!bvh) fail();
    const s = text(bvh);
    sm.sm_free(bvh);
    return s;
  };
}

if (typeof process == "object" && process.argv[1]?.endsWith("05_wasm.mjs")) {
  const [wasm, model, prompt = "waves with the right hand"] = process.argv.slice(2);
  const generate = await load(wasm, model);
  process.stdout.write(generate(prompt));
}
