use std::panic::{catch_unwind, AssertUnwindSafe};

use einsum_ndarray::{einsum, EinsumError, EinsumPlan, Strategy};
use ndarray::{ArrayD, IxDyn};

#[test]
fn defect_public_inputs_return_errors_without_panicking() {
    let alphabet = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.-> 0123";
    let mut state = 0x6a09_e667_f3bc_c909u64;
    for iteration in 0..100_000usize {
        let length = usize::try_from(next(&mut state) % 17).unwrap();
        let mut text = String::with_capacity(length);
        for _ in 0..length {
            let index = usize::try_from(next(&mut state)).unwrap() % alphabet.len();
            text.push(char::from(alphabet[index]));
        }
        let operand_count = usize::try_from(next(&mut state) % 4).unwrap();
        let mut arrays = Vec::with_capacity(operand_count);
        for _ in 0..operand_count {
            let rank = usize::try_from(next(&mut state) % 4).unwrap();
            let shape: Vec<usize> = (0..rank)
                .map(|_| usize::try_from(next(&mut state) % 4).unwrap())
                .collect();
            arrays.push(ArrayD::<i64>::zeros(IxDyn(&shape)));
        }
        let views = arrays.iter().map(|array| array.view()).collect::<Vec<_>>();
        let outcome = catch_unwind(AssertUnwindSafe(|| einsum(&text, &views)));
        assert!(
            outcome.is_ok(),
            "iteration {iteration} panicked for {text:?}"
        );
    }
}

#[test]
fn output_size_overflow_is_reported_before_allocation() {
    let large = usize::MAX / 2;
    let shapes: [&[usize]; 4] = [&[large], &[large], &[large], &[large]];
    let error = EinsumPlan::new("i,j,k,l->ijkl", &shapes).unwrap_err();
    assert_eq!(error, EinsumError::SizeOverflow);
}

#[test]
fn invalid_explicit_paths_return_typed_errors() {
    let shapes: [&[usize]; 3] = [&[2, 2], &[2, 2], &[2, 2]];
    let repeated = EinsumPlan::with_strategy(
        "ij,jk,kl->il",
        &shapes,
        Strategy::Explicit(vec![vec![0, 0]]),
    )
    .unwrap_err();
    assert!(matches!(repeated, EinsumError::InvalidPath { .. }));

    let incomplete = EinsumPlan::with_strategy(
        "ij,jk,kl->il",
        &shapes,
        Strategy::Explicit(vec![vec![0, 1]]),
    )
    .unwrap_err();
    assert!(matches!(incomplete, EinsumError::InvalidPath { .. }));
}

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}
