# arbcycle scaffold

This is a Poetry-friendly, CMake + nanobind project scaffold for a single Python import surface, `arbcycle`, with per-ISA C++ stub files and a runtime backend detector.

## Current structure

- `src/arbcycle/` — the Python package users import
- `src/cpp/module.cpp` — the nanobind extension entry point
- `src/cpp/detect/` — runtime backend detection
- `src/cpp/backends/` — one stub translation unit per ISA / platform variant
- `CMakeLists.txt` — target selection and extension build
- `pyproject.toml` — Poetry-friendly metadata + scikit-build-core backend

## Notes

- The package exposes a **single Python module** (`arbcycle`).
- The native extension is `_arbcycle_native`.
- The detector returns the **best currently usable backend** for the running machine.
- The backend stub files are intentionally thin placeholders. They exist so you can drop in real kernels later without redoing the project layout.
- On Linux AArch64, the architectural name is **ASIMD**; many people say **NEON** informally.
- On macOS arm64, this scaffold includes **NEON** and **SME** stub files as requested.

## Suggested development flow

```bash
poetry install --with dev
poetry run pip install --no-build-isolation -ve .
poetry run pytest
```

## Building without Poetry isolation

`poetry build` creates a temporary isolated build environment and tries to
download the backend declared in `[build-system].requires`. For this project,
that means fetching `scikit-build-core`, `nanobind`, `cmake`, and `ninja`
before the native build even starts.

If you are offline, behind a restricted index, or debugging local native build
issues, use the project environment directly instead:

```bash
poetry install --with dev
bash scripts/build-wheel-no-isolation.sh
```

That script runs:

```bash
poetry run python -m build --wheel --no-isolation
```

For editable installs:

```bash
poetry install --with dev
poetry run pip install --no-build-isolation -ve .
```

If you still want to use `poetry build`, make sure the machine can reach your
package index for the build backend dependencies.

## Files to edit first

- `src/arbcycle/__init__.py`
- `src/cpp/module.cpp`
- the specific backend files under `src/cpp/backends/`
