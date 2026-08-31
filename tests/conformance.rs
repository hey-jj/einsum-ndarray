mod common;

use einsum_ndarray::{EinsumError, EinsumPlan, Strategy};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

use common::{left_to_right_path, load_cases};

#[test]
fn all_158_vectors_match_resolved_labels_and_values() {
    let cases = load_cases();
    let mut evaluated = 0;
    for case in cases.iter().filter(|case| case.expected.is_some()) {
        let expected = case.expected.as_ref().unwrap();
        let arrays_f64: Vec<ArrayD<f64>> = case
            .operands
            .iter()
            .map(|operand| {
                ArrayD::from_shape_vec(IxDyn(&operand.shape), operand.data.clone()).unwrap()
            })
            .collect();
        let views_f64: Vec<ArrayViewD<'_, f64>> =
            arrays_f64.iter().map(|array| array.view()).collect();
        let shapes: Vec<&[usize]> = arrays_f64.iter().map(ArrayD::shape).collect();
        let plan = EinsumPlan::new(&case.subscripts, &shapes)
            .unwrap_or_else(|error| panic!("{} failed to plan: {error}", case.id));
        assert_eq!(
            plan.output_subscripts(),
            case.resolved_output.as_deref().unwrap(),
            "{} resolved labels",
            case.id
        );
        assert_eq!(plan.output_shape(), expected.shape, "{} shape", case.id);
        let result = plan
            .execute(&views_f64)
            .unwrap_or_else(|error| panic!("{} failed to execute: {error}", case.id));
        assert_close(
            result.as_slice().unwrap(),
            &expected.data,
            1e-12,
            1e-12,
            &case.id,
        );

        #[cfg(has_reference_evaluator)]
        {
            let direct = common::naive::evaluate(&case.subscripts, &views_f64);
            assert_close(
                direct.as_slice().unwrap(),
                &expected.data,
                1e-12,
                1e-12,
                &format!("{} direct", case.id),
            );
        }

        let arrays_f32: Vec<ArrayD<f32>> = case
            .operands
            .iter()
            .map(|operand| {
                ArrayD::from_shape_vec(
                    IxDyn(&operand.shape),
                    operand.data.iter().map(|value| *value as f32).collect(),
                )
                .unwrap()
            })
            .collect();
        let views_f32: Vec<ArrayViewD<'_, f32>> =
            arrays_f32.iter().map(|array| array.view()).collect();
        let result_f32 = plan.execute(&views_f32).unwrap();
        let result_f32_as_f64: Vec<f64> =
            result_f32.iter().map(|value| f64::from(*value)).collect();
        assert_close(
            &result_f32_as_f64,
            &expected.data,
            1e-5,
            1e-6,
            &format!("{} f32", case.id),
        );

        let arrays_i64: Vec<ArrayD<i64>> = case
            .operands
            .iter()
            .map(|operand| {
                ArrayD::from_shape_vec(IxDyn(&operand.shape), operand.data_int.clone()).unwrap()
            })
            .collect();
        let views_i64: Vec<ArrayViewD<'_, i64>> =
            arrays_i64.iter().map(|array| array.view()).collect();
        let result_i64 = plan.execute(&views_i64).unwrap();
        assert_eq!(
            result_i64.as_slice().unwrap(),
            expected.data_int_scaled,
            "{} i64",
            case.id
        );
        evaluated += 1;
    }
    assert_eq!(evaluated, 158);
}

#[test]
fn defect_fixed_contraction_order_matches_all_recorded_path_costs() {
    let cases = load_cases();
    let mut checked = 0;
    for case in cases.iter().filter(|case| case.path.is_some()) {
        let expected = case.path.as_ref().unwrap();
        let shapes: Vec<&[usize]> = case
            .operands
            .iter()
            .map(|operand| operand.shape.as_slice())
            .collect();
        let plan = EinsumPlan::new(&case.subscripts, &shapes).unwrap();
        assert_eq!(
            plan.path().naive_flops,
            expected.naive_flops,
            "{} naive cost",
            case.id
        );
        assert_eq!(
            plan.path().optimized_flops,
            expected.optimal_flops,
            "{} optimal cost",
            case.id
        );
        let explicit = EinsumPlan::with_strategy(
            &case.subscripts,
            &shapes,
            Strategy::Explicit(left_to_right_path(shapes.len())),
        )
        .unwrap();
        assert_eq!(
            explicit.path().optimized_flops,
            expected.naive_left_to_right_flops,
            "{} left-to-right cost",
            case.id
        );
        checked += 1;
    }
    assert_eq!(checked, 114);
}

#[test]
fn defect_string_errors_match_all_malformed_cases() {
    let cases = load_cases();
    let mut checked = 0;
    for case in cases.iter().filter(|case| case.error_kind.is_some()) {
        let shapes: Vec<&[usize]> = case
            .operands
            .iter()
            .map(|operand| operand.shape.as_slice())
            .collect();
        let error = EinsumPlan::new(&case.subscripts, &shapes)
            .expect_err("malformed case unexpectedly planned");
        assert_eq!(
            error_kind(&error),
            case.error_kind.as_deref().unwrap(),
            "{}",
            case.id
        );
        checked += 1;
    }
    assert_eq!(checked, 31);
}

fn assert_close(actual: &[f64], expected: &[f64], relative: f64, absolute: f64, case: &str) {
    assert_eq!(actual.len(), expected.len(), "{case} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = absolute + relative * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{case} element {index}: got {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn error_kind(error: &EinsumError) -> &'static str {
    match error {
        EinsumError::InvalidCharacter { .. } => "InvalidCharacter",
        EinsumError::MalformedSubscripts { .. } => "MalformedSubscripts",
        EinsumError::OperandCountMismatch { .. } => "OperandCountMismatch",
        EinsumError::TooManyLabels { .. } => "TooManyLabels",
        EinsumError::TooFewLabels { .. } => "TooFewLabels",
        EinsumError::DiagonalSizeMismatch { .. } => "DiagonalSizeMismatch",
        EinsumError::BroadcastMismatch { .. } => "BroadcastMismatch",
        EinsumError::OutputLabelRepeated { .. } => "OutputLabelRepeated",
        EinsumError::OutputLabelUnknown { .. } => "OutputLabelUnknown",
        EinsumError::OutputEllipsisMissing { .. } => "OutputEllipsisMissing",
        EinsumError::SizeOverflow => "SizeOverflow",
        EinsumError::ShapeMismatch { .. } => "ShapeMismatch",
        EinsumError::InvalidPath { .. } => "InvalidPath",
        _ => "Unknown",
    }
}
