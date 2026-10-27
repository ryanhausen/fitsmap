# Rust Port of the Image Tiling Algorithm: Plan

Goal: learn Rust by porting fitsmap's image tiling path ([fitsmap/convert.py](fitsmap/convert.py)) to a PyO3 extension.

## Setup (done)

- Crate: [rust/](rust/), PyO3 0.30, `crate-type = ["cdylib"]`
- Distribution name: `fitsmap_tiling`. Import name: `_tiling`, set by `[lib] name`, which must match the `#[pymodule]` name.
- Dev loop: run `maturin develop` in `rust/`, then `uv run python -c "import _tiling"`
- maturin is a dev dependency: `uv add --dev maturin`
- Gotcha: `uv sync` uninstalls `_tiling` because it isn't in the lockfile. Run `maturin develop` again afterwards.

## Port order

Work from the leaf functions up to the orchestrator.

| # | Function | Rust concepts | Status |
|---|---|---|---|
| 0 | `add` (stub) | `#[pyfunction]`, `#[pymodule]`, `#[pymodule_export]` | done |
| 1 | [`get_total_tiles`](fitsmap/convert.py#L242) | integer types, ranges, `.map().sum()`, `pow`, overflow | |
| 2 | [`get_zoom_range`](fitsmap/convert.py#L222) | tuples, `as f64` casts, `log2`/`ceil`, truncation | |
| 3 | [`build_path`](fitsmap/convert.py#L77) | `String` vs `&str`, `PathBuf`, `format!` | |
| 4 | [`slice_idx_generator`](fitsmap/convert.py#L97) | structs, `Vec<T>` vs `impl Iterator`, `PyResult` errors | |
| 5 | [`make_tile_pil`](fitsmap/convert.py#L293) / [`mem_safe_make_tile`](fitsmap/convert.py#L322) | `numpy` crate, `PyReadonlyArray2`, `ndarray` slicing, borrowing | |
| 6 | [`tile_img`](fitsmap/convert.py#L419) | `rayon` parallelism instead of ray, image encoding | |

## Questions for steps 1–2

- `4**i` grows quickly. Which integer type do you use? What does Rust do on overflow in debug builds compared with release builds?
- Python's `int()` truncates toward zero. Which Rust operation does the same?
- `get_zoom_range`: take `(usize, usize)` or `&[usize]`? Which is clearer for the caller?
- Python's `get_zoom_range` takes `tile_size` as a tuple but only reads `[0]`. Keep that quirk in Rust or drop it?

## Testing

For every ported function:
- `#[test]` unit tests in Rust (`cargo test`)
- a pytest parity check comparing `_tiling.<fn>` against the Python original over several inputs

## Later

- Package as `fitsmap._tiling`, either with maturin `module-name` or with setuptools-rust.
- Choose the build backend: setuptools-rust keeps `setuptools_scm` versioning; maturin is simpler but takes its version from `Cargo.toml`.
