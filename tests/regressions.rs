use einsum_ndarray::{einsum, einsum_path, EinsumError, EinsumPlan, Strategy};
use ndarray::{array, ArrayD, ArrayViewD, IxDyn};

#[test]
fn defect_fixed_left_to_right_order_selects_the_lower_cost_chain() {
    let shapes: [&[usize]; 4] = [&[16, 16], &[16, 16], &[16, 16], &[16, 1]];
    let path = einsum_path("ab,bc,cd,de->ae", &shapes).unwrap();
    assert_eq!(path.optimized_flops, 1536);
    assert!(path.optimized_flops < path.naive_flops);
}

#[test]
fn defect_generic_iteration_dispatches_matrix_contractions() {
    let matmul_shapes: [&[usize]; 2] = [&[3, 4], &[4, 5]];
    let matmul = einsum_path("ij,jk->ik", &matmul_shapes).unwrap();
    assert!(matmul.steps.iter().all(|step| step.gemm));

    let batch_shapes: [&[usize]; 2] = [&[2, 3, 4], &[2, 4, 5]];
    let batch = einsum_path("bij,bjk->bik", &batch_shapes).unwrap();
    assert!(batch.steps.iter().all(|step| step.gemm));

    let bilinear_shapes: [&[usize]; 3] = [&[3], &[3, 4], &[4]];
    let bilinear = einsum_path("i,ij,j->", &bilinear_shapes).unwrap();
    assert!(bilinear.steps.iter().any(|step| step.gemm));
}

#[test]
fn defect_missing_ellipsis_support_broadcasts_batch_axes() {
    let left =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right =
        ArrayD::from_shape_vec(IxDyn(&[2, 3, 1]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let operands = [left.view(), right.view()];
    let result = einsum("...ij,...jk->...ik", &operands).unwrap();
    assert_eq!(result.shape(), &[2, 2, 1]);
    assert_eq!(result.as_slice().unwrap(), &[14.0, 32.0, 32.0, 77.0]);
}

#[test]
fn defect_unpinned_diagonal_broadcast_and_empty_axes_are_defined() {
    let matrix = array![[1_i64, 2, 3], [4, 5, 6], [7, 8, 9]].into_dyn();
    let diagonal = einsum("ii->i", &[matrix.view()]).unwrap();
    assert_eq!(diagonal.as_slice().unwrap(), &[1, 5, 9]);

    let left = array![1_i64, 2, 3].into_dyn();
    let one = array![4_i64].into_dyn();
    let broadcast = einsum("i,i->i", &[left.view(), one.view()]).unwrap();
    assert_eq!(broadcast.as_slice().unwrap(), &[4, 8, 12]);

    let empty = ArrayD::<i64>::zeros(IxDyn(&[2, 0]));
    let other = ArrayD::<i64>::zeros(IxDyn(&[0, 3]));
    let product = einsum("ij,jk->ik", &[empty.view(), other.view()]).unwrap();
    assert_eq!(product, ArrayD::<i64>::zeros(IxDyn(&[2, 3])));
}

#[test]
fn plan_rejects_execution_shapes_that_differ_from_the_bound_shapes() {
    let plan = EinsumPlan::new("ij->ji", &[&[2, 3]]).unwrap();
    let wrong = ArrayD::<f64>::zeros(IxDyn(&[3, 2]));
    let error = plan.execute(&[wrong.view()]).unwrap_err();
    assert!(matches!(
        error,
        einsum_ndarray::EinsumError::ShapeMismatch { .. }
    ));
}

#[test]
fn scalar_operands_and_results_keep_zero_rank() {
    let scalar = ArrayD::from_shape_vec(IxDyn(&[]), vec![3_i64]).unwrap();
    let vector = array![2_i64, 4, 6].into_dyn();
    let scaled = einsum(",i->i", &[scalar.view(), vector.view()]).unwrap();
    assert_eq!(scaled.as_slice().unwrap(), &[6, 12, 18]);

    let dot = einsum("i,i->", &[vector.view(), vector.view()]).unwrap();
    assert_eq!(dot.ndim(), 0);
    assert_eq!(dot.as_slice().unwrap(), &[56]);
}

#[test]
fn one_shot_api_accepts_non_owned_dynamic_views() {
    let left = array![[1_i64, 2], [3, 4]];
    let right = array![[5_i64, 6], [7, 8]];
    let views: Vec<ArrayViewD<'_, i64>> = vec![left.view().into_dyn(), right.view().into_dyn()];
    let result = einsum("ij,jk->ik", &views).unwrap();
    assert_eq!(result, array![[19, 22], [43, 50]].into_dyn());
}

#[test]
fn integer_overflow_wraps_in_reduction_broadcast_and_matrix_kernels() {
    let maximum = array![i64::MAX].into_dyn();
    let two = array![2_i64].into_dyn();
    let product = einsum("i,i->i", &[maximum.view(), two.view()]).unwrap();
    assert_eq!(product.as_slice().unwrap(), &[-2]);

    let summands = array![i64::MAX, 1].into_dyn();
    let sum = einsum("i->", &[summands.view()]).unwrap();
    assert_eq!(sum.as_slice().unwrap(), &[i64::MIN]);

    let left = array![[i64::MAX]].into_dyn();
    let right = array![[2_i64]].into_dyn();
    let matrix = einsum("ij,jk->ik", &[left.view(), right.view()]).unwrap();
    assert_eq!(matrix.as_slice().unwrap(), &[-2]);
}

#[test]
fn parser_reports_too_many_labels_for_rank_one_operand() {
    let error = EinsumPlan::new("ij,j->i", &[&[2], &[2]]).unwrap_err();
    assert!(matches!(
        error,
        EinsumError::TooManyLabels { operand: 0, .. }
    ));
}

#[test]
fn explicit_intermediate_size_overflow_is_rejected_during_planning() {
    let large = usize::MAX / 2 + 1;
    let shapes: [&[usize]; 3] = [&[large], &[large], &[1, 1]];
    let error = EinsumPlan::with_strategy(
        "i,j,ij->",
        &shapes,
        Strategy::Explicit(vec![vec![0, 1], vec![0, 1]]),
    )
    .unwrap_err();
    assert_eq!(error, EinsumError::SizeOverflow);
}

#[test]
fn strategy_default_selects_auto() {
    assert_eq!(Strategy::default(), Strategy::Auto);
}

#[test]
fn public_greedy_and_optimal_strategies_execute_without_panicking() {
    let shapes: [&[usize]; 3] = [&[2, 3], &[3, 4], &[4, 2]];
    let arrays = [
        ArrayD::<i64>::ones(IxDyn(shapes[0])),
        ArrayD::<i64>::ones(IxDyn(shapes[1])),
        ArrayD::<i64>::ones(IxDyn(shapes[2])),
    ];
    let views = arrays.iter().map(|array| array.view()).collect::<Vec<_>>();
    for strategy in [Strategy::Greedy, Strategy::Optimal] {
        let plan = EinsumPlan::with_strategy("ij,jk,kl->il", &shapes, strategy).unwrap();
        assert_eq!(
            plan.execute(&views).unwrap(),
            ArrayD::from_elem(IxDyn(&[2, 2]), 12)
        );
    }
}
