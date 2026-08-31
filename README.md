# einsum-ndarray

`einsum-ndarray` evaluates Einstein summation expressions on dynamically
shaped [`ndarray`](https://docs.rs/ndarray) arrays. It supports explicit and
implicit outputs, repeated-axis diagonals, ellipsis broadcasting, scalar
operands, and zero-length axes.

The default planner searches every contraction tree for expressions with up
to eight operands. It uses a greedy path for larger expressions. Contracted
pairs run through `ndarray` matrix multiplication. Pairwise products without
a contracted label use broadcast multiplication.

## Example

```rust
use einsum_ndarray::{einsum, EinsumPlan};
use ndarray::array;

let left = array![[1.0, 2.0], [3.0, 4.0]];
let right = array![[5.0, 6.0], [7.0, 8.0]];
let operands = [left.view().into_dyn(), right.view().into_dyn()];

let result = einsum("ij,jk->ik", &operands)?;
assert_eq!(result, array![[19.0, 22.0], [43.0, 50.0]].into_dyn());

let shapes: [&[usize]; 2] = [left.shape(), right.shape()];
let plan = EinsumPlan::new("ij,jk->ik", &shapes)?;
assert_eq!(plan.output_shape(), &[2, 2]);

# Ok::<(), einsum_ndarray::EinsumError>(())
```

Build an `EinsumPlan` once when the expression and operand shapes stay fixed.
Each execution checks the shapes before it allocates a result.

## Errors

Parsing, rank checks, broadcasting checks, path validation, and shape checks
return `EinsumError`. Error variants carry the operand, label, position, or
shape data needed to diagnose the input.

Integer additions and multiplications wrap at the type boundary in every build
profile. Floating-point and complex values use their `ndarray` arithmetic.

## Matrix multiplication backends

The default feature set uses `ndarray`'s pure-Rust matrix multiplication.
Enable `blas` when another dependency in the final binary selects and links a
compatible BLAS provider. The `openblas` feature enables the same pass-through
for builds that select OpenBLAS in their application dependency graph.

## Compiler support

The crate supports Rust 1.75 and later.

## License

MIT
