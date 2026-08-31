mod common;

use std::hint::black_box;
use std::time::{Duration, Instant};

use einsum_ndarray::{EinsumPlan, Strategy};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

#[test]
#[ignore = "release timing gate"]
fn optimized_chain_is_at_least_five_times_faster_than_left_to_right() {
    let shapes: [&[usize]; 4] = [&[256, 256], &[256, 256], &[256, 256], &[256, 1]];
    let arrays = [
        filled(&[256, 256], 0.001),
        filled(&[256, 256], 0.002),
        filled(&[256, 256], 0.003),
        filled(&[256, 1], 0.004),
    ];
    let views: Vec<ArrayViewD<'_, f64>> = arrays.iter().map(|array| array.view()).collect();
    let optimized = EinsumPlan::new("ab,bc,cd,de->ae", &shapes).unwrap();
    let explicit = EinsumPlan::with_strategy(
        "ab,bc,cd,de->ae",
        &shapes,
        Strategy::Explicit(vec![vec![0, 1], vec![0, 2], vec![0, 1]]),
    )
    .unwrap();

    black_box(optimized.execute(&views).unwrap());
    black_box(explicit.execute(&views).unwrap());
    let optimized_time = minimum_of_five(|| {
        black_box(optimized.execute(&views).unwrap());
    });
    let explicit_time = minimum_of_five(|| {
        black_box(explicit.execute(&views).unwrap());
    });
    assert_ratio(explicit_time, optimized_time, 5.0, "chain order");
}

#[test]
#[ignore = "release timing gate"]
#[cfg(has_reference_evaluator)]
fn matrix_dispatch_is_at_least_four_times_faster_than_direct_evaluation() {
    let left = filled(&[512, 512], 0.001);
    let right = filled(&[512, 512], 0.002);
    let arrays = [left, right];
    let views: Vec<ArrayViewD<'_, f64>> = arrays.iter().map(|array| array.view()).collect();
    let shapes: [&[usize]; 2] = [&[512, 512], &[512, 512]];
    let plan = EinsumPlan::new("ij,jk->ik", &shapes).unwrap();

    black_box(plan.execute(&views).unwrap());
    black_box(common::naive::evaluate("ij,jk->ik", &views));
    let matrix_time = minimum_of_five(|| {
        black_box(plan.execute(&views).unwrap());
    });
    let direct_time = minimum_of_five(|| {
        black_box(common::naive::evaluate("ij,jk->ik", &views));
    });
    assert_ratio(direct_time, matrix_time, 4.0, "matrix multiplication");
}

#[test]
#[ignore = "release timing gate"]
#[cfg(has_reference_evaluator)]
fn batched_matrix_dispatch_is_at_least_four_times_faster_than_direct_evaluation() {
    let left = filled(&[8, 256, 256], 0.001);
    let right = filled(&[8, 256, 256], 0.002);
    let arrays = [left, right];
    let views: Vec<ArrayViewD<'_, f64>> = arrays.iter().map(|array| array.view()).collect();
    let shapes: [&[usize]; 2] = [&[8, 256, 256], &[8, 256, 256]];
    let plan = EinsumPlan::new("bij,bjk->bik", &shapes).unwrap();

    black_box(plan.execute(&views).unwrap());
    black_box(common::naive::evaluate("bij,bjk->bik", &views));
    let matrix_time = minimum_of_five(|| {
        black_box(plan.execute(&views).unwrap());
    });
    let direct_time = minimum_of_five(|| {
        black_box(common::naive::evaluate("bij,bjk->bik", &views));
    });
    assert_ratio(
        direct_time,
        matrix_time,
        4.0,
        "batched matrix multiplication",
    );
}

fn minimum_of_five(mut measured: impl FnMut()) -> Duration {
    (0..5)
        .map(|_| {
            let start = Instant::now();
            measured();
            start.elapsed()
        })
        .min()
        .unwrap()
}

fn assert_ratio(slow: Duration, fast: Duration, floor: f64, label: &str) {
    let ratio = slow.as_secs_f64() / fast.as_secs_f64();
    println!("{label}: {ratio:.2}x; slow={slow:?}, fast={fast:?}");
    assert!(
        ratio >= floor,
        "{label} ratio {ratio:.2} is below {floor:.2}; slow={slow:?}, fast={fast:?}"
    );
}

fn filled(shape: &[usize], value: f64) -> ArrayD<f64> {
    let len = shape.iter().copied().product();
    ArrayD::from_shape_vec(IxDyn(shape), vec![value; len]).unwrap()
}
