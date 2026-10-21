# make test   unit tests and examples (optimised build)      make asan   the tests under AddressSanitizer + UBSan
# make leaks  leak check (macOS leaks, Linux valgrind)      make lib   libsprongmotion.a and the shared library
# make bench  time per phase on $(SM_MODEL_DIR)/SprongMotion-14M-v1.0-Q8_0.gguf,
#             BENCH_ARGS="threads runs" (0 10: all cores, best of 10)
# make dist   build/dist/sprongmotion-<version>-<platform>.tar.gz: header, static + shared library, licenses
# make wasm   build/wasm: sprongmotion.wasm (SIMD) and sprongmotion-plain.wasm, standalone (emcc)
# make wasm-test  the unit tests built with emcc, run in node
# make clean
# SM_MODEL_DIR=dir  where test_api, test_path, test_edit and the examples find SprongMotion-14M-v1.0-Q8_0.gguf
#                   (without it, skipped)
# SM_GOLDEN_F32, SM_GOLDEN_Q8, SM_GATE_MODEL=file  the model files the golden data and the path gate were made with
#                   (not public: without them, test_gen and the path gate are skipped)
# FAST=1      native only: fused multiply-add (~1.7x faster maths), results not bit-identical to other builds,
#             built in build/fast
CC     ?= cc
CFLAGS ?= -O2
FP      = $(if $(filter 1,$(FAST)),-ffp-contract=fast -DSM_FMA,-ffp-contract=off)
WFLAGS  = -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror $(FP) -Iinclude
WARN    = $(WFLAGS) -pthread
SAN     = -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all
SRC     = $(wildcard src/*.c)
B       = $(if $(filter 1,$(FAST)),build/fast,build)
TESTS   = $(patsubst tests/%.c,$(B)/%,$(wildcard tests/test_*.c))
EXAMPLES = $(patsubst examples/%.c,$(B)/examples/%,$(wildcard examples/*.c))
EXMODEL = $(SM_MODEL_DIR)/SprongMotion-14M-v1.0-Q8_0.gguf
MACHINE := $(shell $(CC) -dumpmachine)
OS      = $(if $(findstring apple,$(MACHINE)),macos,$(if $(findstring mingw,$(MACHINE)),windows,linux))
ARCH    = $(subst aarch64,arm64,$(firstword $(subst -, ,$(MACHINE))))
SO      = $(if $(filter macos,$(OS)),dylib,$(if $(filter windows,$(OS)),dll,so))
comma   = ,
SHARED  = $(if $(filter macos,$(OS)),-dynamiclib -install_name @rpath/libsprongmotion.dylib -pthread,$(if \
          $(filter windows,$(OS)),-shared -static-libgcc -Wl$(comma)--out-implib$(comma)$(B)/libsprongmotion.dll.a,-shared -pthread))
PIC     = $(if $(filter windows,$(OS)),,-fPIC)
RPATH   = $(if $(filter windows,$(OS)),,-Wl$(comma)-rpath$(comma)$(abspath $(B)))
EXPORTS = $(if $(filter macos,$(OS)),nm -gU,nm -D --defined-only) $(B)/libsprongmotion.$(SO) | awk '{print $$3}'
VERSION := $(shell awk '/define SM_VERSION_(MAJOR|MINOR|PATCH) /{v = v (v == "" ? "" : ".") $$3} END {print v}' include/sm.h)
DIST    = sprongmotion-$(VERSION)-$(OS)-$(ARCH)
DOCS    = LICENSE NOTICE README.md CHANGELOG.md
EMCC   ?= emcc
LP     := (
WASMLIB = -O2 -sSTANDALONE_WASM --no-entry -sALLOW_MEMORY_GROWTH=1 -sEXPORTED_FUNCTIONS=$(shell awk -F'$(LP)' \
          '/^SM_API/ {n = split($$1, a, /[ *]/); printf "_%s,", a[n]}' include/sm.h)_malloc,_free
WASMTEST = -O2 -msimd128 -sNODERAWFS=1 -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=1MB --pre-js tests/env.js

lib: $(B)/libsprongmotion.a $(B)/libsprongmotion.$(SO)
$(B)/libsprongmotion.a: $(SRC:src/%.c=$(B)/%.o)
	ar rcs $@ $^
$(B)/libsprongmotion.$(SO): $(SRC:src/%.c=$(B)/%.o)
	$(CC) $(SHARED) $^ -o $@ -lm
$(B)/%.o: src/%.c src/sm_internal.h include/sm.h | $(B)
	$(CC) $(WARN) $(CFLAGS) $(PIC) -fvisibility=hidden -DSM_SHARED -DSM_BUILD -c $< -o $@
$(B)/test_api: tests/test_api.c include/sm.h tests/check.h $(B)/libsprongmotion.$(SO)
	$(CC) $(WARN) $(CFLAGS) -DSM_SHARED $< -L$(B) -lsprongmotion $(RPATH) -o $@ -lm
$(B)/test_%: tests/test_%.c $(SRC) src/sm_internal.h tests/check.h | $(B)
	$(CC) $(WARN) $(CFLAGS) $< $(SRC) -o $@ -lm
$(B)/asan_%: tests/test_%.c $(SRC) src/sm_internal.h tests/check.h | $(B)
	$(CC) $(WARN) $(SAN) $< $(SRC) -o $@ -lm
$(B)/examples/%: examples/%.c include/sm.h $(B)/libsprongmotion.a | $(B)/examples
	$(CC) $(WARN) $(CFLAGS) $< $(B)/libsprongmotion.a -o $@ -lm
$(B) $(B)/examples:
	mkdir -p $@

test: $(TESTS) $(EXAMPLES) symbols
	@for t in $(TESTS); do $$t || exit 1; done
	@if [ -f "$(EXMODEL)" ]; then $(B)/examples/01_hello $(EXMODEL) "a person waves" $(B)/hello.bvh && \
	  $(B)/examples/02_request $(EXMODEL) && $(B)/examples/03_steps $(EXMODEL) && $(B)/examples/04_edit $(EXMODEL); \
	  else echo "examples: built, not run (no model in \$$SM_MODEL_DIR)"; fi
symbols: $(B)/libsprongmotion.$(SO)
	@if [ $(OS) = windows ]; then echo "exports: not checked on Windows"; else \
	  $(EXPORTS) | sed 's/^_//' | sort > $(B)/exports.txt && \
	  sed -n 's/^SM_API[^(]*[ *]\(sm_[a-z0-9_]*\)(.*/\1/p' include/sm.h | sort | diff - $(B)/exports.txt && \
	  echo "exports: $$(grep -c . $(B)/exports.txt) symbols, as declared in sm.h"; fi
$(B)/bench: bench/bench.c include/sm.h $(B)/libsprongmotion.a
	$(CC) $(WARN) $(CFLAGS) $< $(B)/libsprongmotion.a -o $@ -lm
bench: $(B)/bench
	$(B)/bench $(EXMODEL) $(BENCH_ARGS)
asan: $(TESTS:$(B)/test_%=$(B)/asan_%)
	@for t in $^; do $$t || exit 1; done
leaks: $(TESTS)
	@for t in $^; do if command -v valgrind >/dev/null; then valgrind -q --leak-check=full --error-exitcode=1 $$t || exit 1; \
	  else leaks --atExit -- $$t | grep -E "leaks for" || exit 1; fi; done
dist: lib
	rm -rf build/dist/$(DIST) && mkdir -p build/dist/$(DIST)/include build/dist/$(DIST)/lib
	cp include/sm.h build/dist/$(DIST)/include && cp $(DOCS) build/dist/$(DIST)
	cp $(B)/libsprongmotion.a $(B)/libsprongmotion.$(SO) $(wildcard $(B)/libsprongmotion.dll.a) build/dist/$(DIST)/lib
	tar -C build/dist -czf build/dist/$(DIST).tar.gz $(DIST) && echo build/dist/$(DIST).tar.gz
wasm: build/wasm/sprongmotion.wasm build/wasm/sprongmotion-plain.wasm
build/wasm/sprongmotion.wasm: $(SRC) src/sm_internal.h include/sm.h | build/wasm
	$(EMCC) $(WFLAGS) $(WASMLIB) -msimd128 $(SRC) -o $@
build/wasm/sprongmotion-plain.wasm: $(SRC) src/sm_internal.h include/sm.h | build/wasm
	$(EMCC) $(WFLAGS) $(WASMLIB) $(SRC) -o $@
build/wasm/test_api.js: tests/test_api.c $(SRC) src/sm_internal.h tests/check.h tests/env.js | build/wasm
	$(EMCC) $(WFLAGS) $(WASMTEST) -pthread -Wno-pthreads-mem-growth -sPROXY_TO_PTHREAD -sEXIT_RUNTIME $< $(SRC) -o $@ -lm
build/wasm/test_%.js: tests/test_%.c $(SRC) src/sm_internal.h tests/check.h tests/env.js | build/wasm
	$(EMCC) $(WFLAGS) $(WASMTEST) $< $(SRC) -o $@ -lm
wasm-test: $(patsubst tests/%.c,build/wasm/%.js,$(wildcard tests/test_*.c))
	@for t in $^; do node $$t || exit 1; done
dist-wasm: wasm
	rm -rf build/dist/sprongmotion-$(VERSION)-wasm && mkdir -p build/dist/sprongmotion-$(VERSION)-wasm
	cp build/wasm/*.wasm include/sm.h $(DOCS) build/dist/sprongmotion-$(VERSION)-wasm
	tar -C build/dist -czf build/dist/sprongmotion-$(VERSION)-wasm.tar.gz sprongmotion-$(VERSION)-wasm \
	  && echo build/dist/sprongmotion-$(VERSION)-wasm.tar.gz
build/wasm:
	mkdir -p $@
clean:
	rm -rf build
.PHONY: lib test symbols bench asan leaks dist wasm wasm-test dist-wasm clean
