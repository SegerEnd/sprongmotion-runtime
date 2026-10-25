# Contributing

Bug reports and small fixes are welcome as issues and pull requests.

- Plain C11, no dependencies, make only. `docs/design.md` lists what each file in `src/` holds.
- `make test` and `make asan` pass (see README for the tests that need no data).
- Output stays bit-identical across exact builds and thread counts: a change that alters generated motion needs a reason.
- Comments only where the code cannot say it: formulas, sources, specs, reasons.

By contributing you agree that your contribution is licensed under Apache-2.0.
