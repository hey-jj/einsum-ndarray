# Changelog

## 0.1.1 - 2026-10-03

- Add disambiguation note in README paragraph 1.

## 0.1.0 - 2026-08-22

- Add one-shot Einstein summation for dynamic arrays.
- Add reusable shape-bound plans with optimal, greedy, and explicit paths.
- Support ellipses, diagonals, broadcasting, scalars, and empty axes.
- Report typed validation and execution errors.
- Use pure-Rust matrix multiplication by default.
- Add opt-in `blas` and `openblas` matrix multiplication features.
- Make integer overflow wrap consistently in every build profile.
- Return `TooManyLabels` when a subscript names more axes than its operand.
- Require Rust 1.75, the minimum supported by the matrix kernel dependency.

Path costs follow NumPy's reported contraction model.
