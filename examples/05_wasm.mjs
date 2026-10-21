// Text to BVH with the WebAssembly build, in a browser or node. Usage: node 05_wasm.mjs sprongmotion.wasm model.gguf "waves"
const read = async (url) =>
  typeof process == "object" ? (await import("node:fs")).readFileSync(url) : new Uint8Array(await (await fetch(url)).arrayBuffer());

export async function load(wasmUrl, modelUrl) {
  const stub = () => 0;
  const { instance } = await WebAssembly.instantiate(await read(wasmUrl), {
    env: { emscripten_notify_memory_growth: stub },
    wasi_snapshot_preview1: { fd_close: stub, fd_write: stub, fd_read: stub, fd_seek: stub },
  });
  const sm = instance.exports;
  sm._initialize();
  const bytes = () => new Uint8Array(sm.memory.buffer);
  const u32 = (p) => new DataView(sm.memory.buffer).getUint32(p, true);
  const cstr = (s) => {
    const b = new TextEncoder().encode(s + "\0"), p = sm.malloc(b.length);
    bytes().set(b, p);
    return p;
  };
  const text = (p) => new TextDecoder().decode(bytes().subarray(p, bytes().indexOf(0, p)));
  const err = sm.malloc(256), out = sm.malloc(4);
  const check = (status) => {
    if (status) throw new Error(text(err) || text(sm.sm_status_message(status)));
  };

  const file = await read(modelUrl), data = sm.malloc(file.length);
  bytes().set(file, data);   // the model is used in place: keep this buffer
  check(sm.sm_model_load_memory(data, file.length, 0, 0, out, err, 256));
  const model = u32(out);
  check(sm.sm_motion_create(model, out));
  const motion = u32(out);

  return function generate(prompt, seconds = 4, seed = 1) {
    const p = cstr(prompt);
    try {
      check(sm.sm_generate_simple(model, p, seconds, seed, motion, err, 256));
    } finally {
      sm.free(p);
    }
    sm.sm_motion_write_bvh(motion, 100, 0, 0, out);   // a NULL buffer only sizes it; 100 = centimetres
    const size = u32(out), buf = sm.malloc(size);
    sm.sm_motion_write_bvh(motion, 100, buf, size, 0);
    const bvh = text(buf);
    sm.free(buf);
    return bvh;
  };
}

if (typeof process == "object" && process.argv[1]?.endsWith("05_wasm.mjs")) {
  const [wasm, model, prompt = "waves with the right hand"] = process.argv.slice(2);
  const generate = await load(wasm, model);
  process.stdout.write(generate(prompt));
}
