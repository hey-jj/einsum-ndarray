use std::any::{Any, TypeId};

use ndarray::{Array3, ArrayD, ArrayViewD, Axis, IxDyn, LinalgScalar, ShapeBuilder, Zip};

use crate::parse::{Label, ParsedExpression, ParsedOperand};
use crate::{ContractionPath, EinsumError, EinsumPlan};

#[derive(Clone, Debug)]
pub(crate) struct ExecutionPlan {
    operands: Vec<OperandExecution>,
    steps: Vec<StepExecution>,
}

#[derive(Clone, Debug)]
struct OperandExecution {
    reductions: Vec<Label>,
}

#[derive(Clone, Debug)]
struct StepExecution {
    operands: Vec<usize>,
    pairs: Vec<PairExecution>,
}

#[derive(Clone, Debug)]
struct PairExecution {
    left_reductions: Vec<Label>,
    right_reductions: Vec<Label>,
    batch: Vec<Label>,
    left_only: Vec<Label>,
    contracted: Vec<Label>,
    right_only: Vec<Label>,
    result_labels: Vec<Label>,
    orientation: MatrixOrientation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MatrixOrientation {
    Broadcast,
    LeftThenRight,
    RightThenLeft,
}

#[derive(Clone, Debug)]
struct PlanningTensor {
    labels: Vec<Label>,
    standard_order: Option<Vec<Label>>,
}

enum TensorData<'a, A> {
    View(ArrayViewD<'a, A>),
    Owned(ArrayD<A>),
    DiagonalView {
        base: ArrayViewD<'a, A>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
    DiagonalOwned {
        base: ArrayD<A>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

struct Tensor<'a, A> {
    data: TensorData<'a, A>,
    labels: Vec<Label>,
}

impl<A> Tensor<'_, A> {
    fn view(&self) -> Result<ArrayViewD<'_, A>, EinsumError> {
        match &self.data {
            TensorData::View(view) => Ok(view.view()),
            TensorData::Owned(array) => Ok(array.view()),
            TensorData::DiagonalView {
                base,
                shape,
                strides,
            } => diagonal_view(base.as_slice(), shape, strides),
            TensorData::DiagonalOwned {
                base,
                shape,
                strides,
            } => diagonal_view(base.as_slice(), shape, strides),
        }
    }
}

fn diagonal_view<'a, A>(
    slice: Option<&'a [A]>,
    shape: &[usize],
    strides: &[usize],
) -> Result<ArrayViewD<'a, A>, EinsumError> {
    let slice = slice.ok_or(EinsumError::InvalidPath {
        reason: "diagonal base is not contiguous",
    })?;
    ArrayViewD::from_shape(IxDyn(shape).strides(IxDyn(strides)), slice).map_err(|_| {
        EinsumError::InvalidPath {
            reason: "diagonal view is out of bounds",
        }
    })
}

pub(crate) fn build_execution_plan(
    expression: &ParsedExpression,
    path: &ContractionPath,
) -> Result<ExecutionPlan, EinsumError> {
    let mut operands = Vec::with_capacity(expression.operands.len());
    let mut active = Vec::with_capacity(expression.operands.len());
    for (index, operand) in expression.operands.iter().enumerate() {
        let reductions = operand
            .unique_labels
            .iter()
            .copied()
            .filter(|label| {
                !expression.output.contains(label)
                    && !expression
                        .operands
                        .iter()
                        .enumerate()
                        .any(|(other, operand)| {
                            other != index && operand.unique_labels.contains(label)
                        })
            })
            .collect::<Vec<_>>();
        let labels = without_labels(&operand.unique_labels, &reductions);
        let has_diagonal = operand.axis_labels.len() != operand.unique_labels.len();
        let standard_order = if has_diagonal && reductions.is_empty() {
            None
        } else {
            Some(labels.clone())
        };
        operands.push(OperandExecution { reductions });
        active.push(PlanningTensor {
            labels,
            standard_order,
        });
    }

    let mut steps = Vec::with_capacity(path.steps.len());
    for step in &path.steps {
        let mut removal = step.operands.clone();
        removal.sort_unstable();
        validate_removal(&removal, active.len())?;
        let mut selected = removal
            .iter()
            .map(|&index| {
                active.get(index).cloned().ok_or(EinsumError::InvalidPath {
                    reason: "planned operand is missing",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for &index in removal.iter().rev() {
            active.remove(index);
        }

        let mut accumulator = selected.remove(0);
        let mut pairs = Vec::with_capacity(selected.len());
        while !selected.is_empty() {
            let next = selected.remove(0);
            let mut needed = expression.output.clone();
            for tensor in active.iter().chain(selected.iter()) {
                extend_unique(&mut needed, &tensor.labels);
            }
            let (pair, result) = plan_pair(accumulator, next, &needed, &expression.sizes)?;
            pairs.push(pair);
            accumulator = result;
        }
        active.push(accumulator);
        steps.push(StepExecution {
            operands: step.operands.clone(),
            pairs,
        });
    }

    if active.len() != 1 {
        return Err(EinsumError::InvalidPath {
            reason: "execution plan does not produce one result",
        });
    }
    checked_label_product(&expression.output, &expression.sizes)?;
    Ok(ExecutionPlan { operands, steps })
}

fn plan_pair(
    left: PlanningTensor,
    right: PlanningTensor,
    needed: &[Label],
    sizes: &[usize],
) -> Result<(PairExecution, PlanningTensor), EinsumError> {
    let left_reductions = left
        .labels
        .iter()
        .copied()
        .filter(|label| !right.labels.contains(label) && !needed.contains(label))
        .collect::<Vec<_>>();
    let left_labels = without_labels(&left.labels, &left_reductions);
    let right_reductions = right
        .labels
        .iter()
        .copied()
        .filter(|label| !left_labels.contains(label) && !needed.contains(label))
        .collect::<Vec<_>>();
    let right_labels = without_labels(&right.labels, &right_reductions);
    let left_order = reduced_standard_order(left.standard_order, &left_labels, &left_reductions);
    let right_order =
        reduced_standard_order(right.standard_order, &right_labels, &right_reductions);

    let mut batch = Vec::new();
    let mut contracted = Vec::new();
    let mut left_only = Vec::new();
    let mut right_only = Vec::new();
    for &label in &left_labels {
        if right_labels.contains(&label) {
            if needed.contains(&label) {
                batch.push(label);
            } else {
                contracted.push(label);
            }
        } else {
            left_only.push(label);
        }
    }
    for &label in &right_labels {
        if !left_labels.contains(&label) {
            right_only.push(label);
        }
    }

    let mut result_labels = batch.clone();
    result_labels.extend_from_slice(&left_only);
    result_labels.extend_from_slice(&right_only);
    let (orientation, standard_order) = if contracted.is_empty() {
        result_labels = left_labels.clone();
        extend_unique(&mut result_labels, &right_labels);
        (MatrixOrientation::Broadcast, Some(result_labels.clone()))
    } else {
        let mut normal_left = batch.clone();
        normal_left.extend_from_slice(&left_only);
        normal_left.extend_from_slice(&contracted);
        let mut normal_right = batch.clone();
        normal_right.extend_from_slice(&contracted);
        normal_right.extend_from_slice(&right_only);
        let mut reverse_right = batch.clone();
        reverse_right.extend_from_slice(&right_only);
        reverse_right.extend_from_slice(&contracted);
        let mut reverse_left = batch.clone();
        reverse_left.extend_from_slice(&contracted);
        reverse_left.extend_from_slice(&left_only);
        let normal_cost = estimated_copy_cost(left_order.as_deref(), &normal_left, sizes)?
            .checked_add(estimated_copy_cost(
                right_order.as_deref(),
                &normal_right,
                sizes,
            )?)
            .ok_or(EinsumError::SizeOverflow)?;
        let reverse_cost = estimated_copy_cost(right_order.as_deref(), &reverse_right, sizes)?
            .checked_add(estimated_copy_cost(
                left_order.as_deref(),
                &reverse_left,
                sizes,
            )?)
            .ok_or(EinsumError::SizeOverflow)?;
        if reverse_cost < normal_cost {
            let mut storage = batch.clone();
            storage.extend_from_slice(&right_only);
            storage.extend_from_slice(&left_only);
            (MatrixOrientation::RightThenLeft, Some(storage))
        } else {
            (
                MatrixOrientation::LeftThenRight,
                Some(result_labels.clone()),
            )
        }
    };
    checked_label_product(&result_labels, sizes)?;
    checked_label_product(&batch, sizes)?;
    checked_label_product(&left_only, sizes)?;
    checked_label_product(&contracted, sizes)?;
    checked_label_product(&right_only, sizes)?;

    let pair = PairExecution {
        left_reductions,
        right_reductions,
        batch,
        left_only,
        contracted,
        right_only,
        result_labels: result_labels.clone(),
        orientation,
    };
    Ok((
        pair,
        PlanningTensor {
            labels: result_labels,
            standard_order,
        },
    ))
}

fn reduced_standard_order(
    order: Option<Vec<Label>>,
    labels: &[Label],
    reductions: &[Label],
) -> Option<Vec<Label>> {
    if reductions.is_empty() {
        order
    } else {
        Some(labels.to_vec())
    }
}

fn estimated_copy_cost(
    standard_order: Option<&[Label]>,
    target: &[Label],
    sizes: &[usize],
) -> Result<usize, EinsumError> {
    if standard_order == Some(target) {
        Ok(0)
    } else {
        checked_label_product(target, sizes)
    }
}

pub(crate) fn execute<A: LinalgScalar>(
    plan: &EinsumPlan,
    operands: &[ArrayViewD<'_, A>],
) -> Result<ArrayD<A>, EinsumError> {
    if operands.len() != plan.expression.shapes.len() {
        return Err(EinsumError::OperandCountMismatch {
            subscripts: plan.expression.shapes.len(),
            operands: operands.len(),
        });
    }
    for (operand, (array, expected)) in operands
        .iter()
        .zip(plan.expression.shapes.iter())
        .enumerate()
    {
        if array.shape() != expected {
            return Err(EinsumError::ShapeMismatch {
                operand,
                expected: expected.clone(),
                actual: array.shape().to_vec(),
            });
        }
    }

    let mut tensors = Vec::with_capacity(operands.len());
    for ((array, parsed), execution) in operands
        .iter()
        .zip(plan.expression.operands.iter())
        .zip(plan.execution.operands.iter())
    {
        tensors.push(simplify_operand(array.view(), parsed, execution)?);
    }
    for step in &plan.execution.steps {
        execute_step(&mut tensors, step, &plan.expression.sizes)?;
    }
    let result = tensors.pop().ok_or(EinsumError::InvalidPath {
        reason: "execution result is missing",
    })?;
    if !tensors.is_empty() {
        return Err(EinsumError::InvalidPath {
            reason: "execution produced more than one result",
        });
    }
    let aligned = align_axes(&result, &plan.expression.output)?;
    let broadcast = aligned
        .broadcast(IxDyn(&plan.expression.output_shape))
        .ok_or(EinsumError::InvalidPath {
            reason: "result cannot broadcast to the output shape",
        })?;
    Ok(broadcast.as_standard_layout().into_owned())
}

fn simplify_operand<'a, A: LinalgScalar>(
    array: ArrayViewD<'a, A>,
    parsed: &ParsedOperand,
    execution: &OperandExecution,
) -> Result<Tensor<'a, A>, EinsumError> {
    let has_diagonal = parsed.axis_labels.len() != parsed.unique_labels.len();
    let tensor = if has_diagonal {
        diagonal_tensor(array, &parsed.axis_labels, &parsed.unique_labels)?
    } else {
        Tensor {
            data: TensorData::View(array),
            labels: parsed.unique_labels.clone(),
        }
    };
    reduce_labels(tensor, &execution.reductions)
}

fn diagonal_tensor<'a, A: LinalgScalar>(
    array: ArrayViewD<'a, A>,
    axis_labels: &[Label],
    unique_labels: &[Label],
) -> Result<Tensor<'a, A>, EinsumError> {
    let data = if array.is_standard_layout() {
        let (shape, strides) =
            diagonal_metadata(axis_labels, unique_labels, array.shape(), array.strides())?;
        diagonal_view(array.as_slice(), &shape, &strides)?;
        TensorData::DiagonalView {
            base: array,
            shape,
            strides,
        }
    } else {
        let base = array.as_standard_layout().into_owned();
        let (shape, strides) =
            diagonal_metadata(axis_labels, unique_labels, base.shape(), base.strides())?;
        diagonal_view(base.as_slice(), &shape, &strides)?;
        TensorData::DiagonalOwned {
            base,
            shape,
            strides,
        }
    };
    Ok(Tensor {
        data,
        labels: unique_labels.to_vec(),
    })
}

fn diagonal_metadata(
    axis_labels: &[Label],
    unique_labels: &[Label],
    source_shape: &[usize],
    source_strides: &[isize],
) -> Result<(Vec<usize>, Vec<usize>), EinsumError> {
    let mut shape = vec![0usize; unique_labels.len()];
    let mut strides = vec![0usize; unique_labels.len()];
    for ((&label, &axis_size), &axis_stride) in axis_labels
        .iter()
        .zip(source_shape.iter())
        .zip(source_strides.iter())
    {
        let position = unique_labels
            .iter()
            .position(|current| *current == label)
            .ok_or(EinsumError::InvalidPath {
                reason: "diagonal label is missing",
            })?;
        *shape.get_mut(position).ok_or(EinsumError::InvalidPath {
            reason: "diagonal shape is missing",
        })? = axis_size;
        let stride = usize::try_from(axis_stride).map_err(|_| EinsumError::InvalidPath {
            reason: "standard array has a negative stride",
        })?;
        let slot = strides.get_mut(position).ok_or(EinsumError::InvalidPath {
            reason: "diagonal stride is missing",
        })?;
        *slot = slot.checked_add(stride).ok_or(EinsumError::SizeOverflow)?;
    }
    Ok((shape, strides))
}

fn execute_step<'a, A: LinalgScalar>(
    tensors: &mut Vec<Tensor<'a, A>>,
    step: &StepExecution,
    sizes: &[usize],
) -> Result<(), EinsumError> {
    let mut removal = step.operands.clone();
    removal.sort_unstable();
    validate_removal(&removal, tensors.len())?;
    let mut selected = Vec::with_capacity(removal.len());
    for &index in removal.iter().rev() {
        selected.push((index, tensors.remove(index)));
    }
    selected.sort_by_key(|entry| entry.0);
    let mut selected = selected
        .into_iter()
        .map(|(_, tensor)| tensor)
        .collect::<Vec<_>>();
    let mut accumulator = selected.remove(0);
    if step.pairs.len() != selected.len() {
        return Err(EinsumError::InvalidPath {
            reason: "execution pair plan has the wrong length",
        });
    }
    for (next, pair) in selected.into_iter().zip(&step.pairs) {
        accumulator = contract_pair(accumulator, next, pair, sizes)?;
    }
    tensors.push(accumulator);
    Ok(())
}

fn contract_pair<'a, A: LinalgScalar>(
    left: Tensor<'a, A>,
    right: Tensor<'a, A>,
    plan: &PairExecution,
    sizes: &[usize],
) -> Result<Tensor<'a, A>, EinsumError> {
    let left = reduce_labels(left, &plan.left_reductions)?;
    let right = reduce_labels(right, &plan.right_reductions)?;
    match plan.orientation {
        MatrixOrientation::Broadcast => broadcast_product(left, right, plan, sizes),
        MatrixOrientation::LeftThenRight => matrix_product(left, right, plan, sizes, false),
        MatrixOrientation::RightThenLeft => matrix_product(right, left, plan, sizes, true),
    }
}

fn reduce_labels<'a, A: LinalgScalar>(
    mut tensor: Tensor<'a, A>,
    reductions: &[Label],
) -> Result<Tensor<'a, A>, EinsumError> {
    for &label in reductions {
        let axis = tensor
            .labels
            .iter()
            .position(|current| *current == label)
            .ok_or(EinsumError::InvalidPath {
                reason: "planned reduction label is missing",
            })?;
        let owned = sum_axis_consistent(tensor.view()?, Axis(axis));
        tensor.labels.remove(axis);
        tensor.data = TensorData::Owned(owned);
    }
    Ok(tensor)
}

fn sum_axis_consistent<A: LinalgScalar>(array: ArrayViewD<'_, A>, axis: Axis) -> ArrayD<A> {
    if uses_wrapping_integer::<A>() {
        array.fold_axis(axis, A::zero(), |sum, value| wrapping_add(sum, value))
    } else {
        array.sum_axis(axis)
    }
}

fn broadcast_product<'a, A: LinalgScalar>(
    left: Tensor<'a, A>,
    right: Tensor<'a, A>,
    plan: &PairExecution,
    sizes: &[usize],
) -> Result<Tensor<'a, A>, EinsumError> {
    let shape = shape_for_labels(&plan.result_labels, sizes)?;
    let left_aligned = align_axes(&left, &plan.result_labels)?;
    let right_aligned = align_axes(&right, &plan.result_labels)?;
    let left_view = left_aligned
        .broadcast(IxDyn(&shape))
        .ok_or(EinsumError::InvalidPath {
            reason: "left operand cannot broadcast",
        })?;
    let right_view = right_aligned
        .broadcast(IxDyn(&shape))
        .ok_or(EinsumError::InvalidPath {
            reason: "right operand cannot broadcast",
        })?;
    let product = if uses_wrapping_integer::<A>() {
        Zip::from(&left_view)
            .and(&right_view)
            .map_collect(wrapping_mul)
    } else {
        &left_view * &right_view
    };
    Ok(Tensor {
        data: TensorData::Owned(product),
        labels: plan.result_labels.clone(),
    })
}

fn matrix_product<'a, A: LinalgScalar>(
    first: Tensor<'a, A>,
    second: Tensor<'a, A>,
    plan: &PairExecution,
    sizes: &[usize],
    reversed: bool,
) -> Result<Tensor<'a, A>, EinsumError> {
    let (first_outer, second_outer) = if reversed {
        (&plan.right_only, &plan.left_only)
    } else {
        (&plan.left_only, &plan.right_only)
    };
    let mut first_order = plan.batch.clone();
    first_order.extend_from_slice(first_outer);
    first_order.extend_from_slice(&plan.contracted);
    let mut second_order = plan.batch.clone();
    second_order.extend_from_slice(&plan.contracted);
    second_order.extend_from_slice(second_outer);
    let first_shape = shape_for_labels(&first_order, sizes)?;
    let second_shape = shape_for_labels(&second_order, sizes)?;
    let first_aligned = align_axes(&first, &first_order)?;
    let second_aligned = align_axes(&second, &second_order)?;
    let first_broadcast =
        first_aligned
            .broadcast(IxDyn(&first_shape))
            .ok_or(EinsumError::InvalidPath {
                reason: "first matrix operand cannot broadcast",
            })?;
    let second_broadcast =
        second_aligned
            .broadcast(IxDyn(&second_shape))
            .ok_or(EinsumError::InvalidPath {
                reason: "second matrix operand cannot broadcast",
            })?;
    let first_standard = first_broadcast.as_standard_layout();
    let second_standard = second_broadcast.as_standard_layout();

    let batches = checked_label_product(&plan.batch, sizes)?;
    let rows = checked_label_product(first_outer, sizes)?;
    let shared = checked_label_product(&plan.contracted, sizes)?;
    let columns = checked_label_product(second_outer, sizes)?;
    let first_3d = first_standard
        .to_shape((batches, rows, shared))
        .map_err(|_| EinsumError::InvalidPath {
            reason: "first matrix reshape failed",
        })?;
    let second_3d = second_standard
        .to_shape((batches, shared, columns))
        .map_err(|_| EinsumError::InvalidPath {
            reason: "second matrix reshape failed",
        })?;
    let mut product = Array3::<A>::zeros((batches, rows, columns));
    for ((mut output, first_matrix), second_matrix) in product
        .outer_iter_mut()
        .zip(first_3d.outer_iter())
        .zip(second_3d.outer_iter())
    {
        if uses_wrapping_integer::<A>() {
            for ((row, column), value) in output.indexed_iter_mut() {
                let left_row = first_matrix.row(row);
                let right_column = second_matrix.column(column);
                *value = left_row.iter().zip(right_column.iter()).fold(
                    A::zero(),
                    |sum, (left, right)| {
                        let term = wrapping_mul(left, right);
                        wrapping_add(&sum, &term)
                    },
                );
            }
        } else {
            output.assign(&first_matrix.dot(&second_matrix));
        }
    }

    let mut storage_labels = plan.batch.clone();
    storage_labels.extend_from_slice(first_outer);
    storage_labels.extend_from_slice(second_outer);
    let storage_shape = shape_for_labels(&storage_labels, sizes)?;
    let product = product
        .into_shape_with_order(IxDyn(&storage_shape))
        .map_err(|_| EinsumError::InvalidPath {
            reason: "matrix result reshape failed",
        })?;
    let product = permute_owned(product, &storage_labels, &plan.result_labels)?;
    Ok(Tensor {
        data: TensorData::Owned(product),
        labels: plan.result_labels.clone(),
    })
}

fn permute_owned<A>(
    array: ArrayD<A>,
    current: &[Label],
    target: &[Label],
) -> Result<ArrayD<A>, EinsumError> {
    if current == target {
        return Ok(array);
    }
    let permutation = target
        .iter()
        .map(|label| {
            current
                .iter()
                .position(|candidate| candidate == label)
                .ok_or(EinsumError::InvalidPath {
                    reason: "matrix result label is missing",
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(array.permuted_axes(IxDyn(&permutation)))
}

fn align_axes<'a, A>(
    tensor: &'a Tensor<'_, A>,
    target: &[Label],
) -> Result<ArrayViewD<'a, A>, EinsumError> {
    let mut ordered_existing = tensor
        .labels
        .iter()
        .enumerate()
        .map(|(axis, label)| {
            target
                .iter()
                .position(|candidate| candidate == label)
                .map(|position| (position, axis))
                .ok_or(EinsumError::InvalidPath {
                    reason: "operand has a label absent from its target",
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    ordered_existing.sort_unstable_by_key(|entry| entry.0);
    let permutation = ordered_existing
        .iter()
        .map(|entry| entry.1)
        .collect::<Vec<_>>();
    let mut view = tensor.view()?.permuted_axes(IxDyn(&permutation));
    for (axis, label) in target.iter().enumerate() {
        if !tensor.labels.contains(label) {
            view = view.insert_axis(Axis(axis));
        }
    }
    Ok(view)
}

fn validate_removal(removal: &[usize], active_len: usize) -> Result<(), EinsumError> {
    if removal.len() < 2 {
        return Err(EinsumError::InvalidPath {
            reason: "each execution step needs at least two operands",
        });
    }
    if removal.windows(2).any(|pair| pair.first() == pair.get(1)) {
        return Err(EinsumError::InvalidPath {
            reason: "an execution step repeats an operand",
        });
    }
    if removal.iter().any(|&index| index >= active_len) {
        return Err(EinsumError::InvalidPath {
            reason: "an execution step index is out of range",
        });
    }
    Ok(())
}

fn shape_for_labels(labels: &[Label], sizes: &[usize]) -> Result<Vec<usize>, EinsumError> {
    checked_label_product(labels, sizes)?;
    labels
        .iter()
        .map(|&label| sizes.get(label).copied().ok_or(EinsumError::SizeOverflow))
        .collect()
}

fn checked_label_product(labels: &[Label], sizes: &[usize]) -> Result<usize, EinsumError> {
    labels.iter().try_fold(1usize, |product, &label| {
        let size = sizes.get(label).copied().ok_or(EinsumError::SizeOverflow)?;
        product.checked_mul(size).ok_or(EinsumError::SizeOverflow)
    })
}

fn without_labels(labels: &[Label], removed: &[Label]) -> Vec<Label> {
    labels
        .iter()
        .copied()
        .filter(|label| !removed.contains(label))
        .collect()
}

fn extend_unique(target: &mut Vec<Label>, labels: &[Label]) {
    for &label in labels {
        if !target.contains(&label) {
            target.push(label);
        }
    }
}

fn uses_wrapping_integer<A: 'static>() -> bool {
    let id = TypeId::of::<A>();
    id == TypeId::of::<i8>()
        || id == TypeId::of::<i16>()
        || id == TypeId::of::<i32>()
        || id == TypeId::of::<i64>()
        || id == TypeId::of::<i128>()
        || id == TypeId::of::<isize>()
        || id == TypeId::of::<u8>()
        || id == TypeId::of::<u16>()
        || id == TypeId::of::<u32>()
        || id == TypeId::of::<u64>()
        || id == TypeId::of::<u128>()
        || id == TypeId::of::<usize>()
}

fn wrapping_add<A: LinalgScalar>(left: &A, right: &A) -> A {
    macro_rules! wrapped {
        ($type:ty) => {
            if let (Some(left), Some(right)) = (
                (left as &dyn Any).downcast_ref::<$type>(),
                (right as &dyn Any).downcast_ref::<$type>(),
            ) {
                if let Some(value) = into_scalar::<A, $type>(left.wrapping_add(*right)) {
                    return value;
                }
            }
        };
    }
    wrapped!(i8);
    wrapped!(i16);
    wrapped!(i32);
    wrapped!(i64);
    wrapped!(i128);
    wrapped!(isize);
    wrapped!(u8);
    wrapped!(u16);
    wrapped!(u32);
    wrapped!(u64);
    wrapped!(u128);
    wrapped!(usize);
    *left + *right
}

fn wrapping_mul<A: LinalgScalar>(left: &A, right: &A) -> A {
    macro_rules! wrapped {
        ($type:ty) => {
            if let (Some(left), Some(right)) = (
                (left as &dyn Any).downcast_ref::<$type>(),
                (right as &dyn Any).downcast_ref::<$type>(),
            ) {
                if let Some(value) = into_scalar::<A, $type>(left.wrapping_mul(*right)) {
                    return value;
                }
            }
        };
    }
    wrapped!(i8);
    wrapped!(i16);
    wrapped!(i32);
    wrapped!(i64);
    wrapped!(i128);
    wrapped!(isize);
    wrapped!(u8);
    wrapped!(u16);
    wrapped!(u32);
    wrapped!(u64);
    wrapped!(u128);
    wrapped!(usize);
    *left * *right
}

fn into_scalar<A: 'static, T: 'static>(value: T) -> Option<A> {
    let value: Box<dyn Any> = Box::new(value);
    value.downcast::<A>().ok().map(|value| *value)
}

#[cfg(test)]
mod unit_tests {
    use super::{build_execution_plan, diagonal_tensor, MatrixOrientation, TensorData};
    use crate::{parse, plan, Strategy};
    use ndarray::Array2;

    #[test]
    fn copy_minimizing_plan_reverses_a_large_transposed_right_operand() {
        let expression = parse::parse("ij,kj->ik", &[&[1, 1024], &[1024, 1024]]).unwrap();
        let path = plan::build_path(&expression, Strategy::Auto).unwrap();
        let execution = build_execution_plan(&expression, &path).unwrap();
        assert_eq!(
            execution.steps[0].pairs[0].orientation,
            MatrixOrientation::RightThenLeft
        );
    }

    #[test]
    fn standard_diagonal_uses_a_borrowed_view_and_nonstandard_input_copies_once() {
        let expression = parse::parse("ii->i", &[&[4, 4]]).unwrap();
        let path = plan::build_path(&expression, Strategy::Auto).unwrap();
        let execution = build_execution_plan(&expression, &path).unwrap();
        assert!(execution.operands[0].reductions.is_empty());

        let array = Array2::<i64>::zeros((4, 4));
        let parsed = &expression.operands[0];
        let standard = diagonal_tensor(
            array.view().into_dyn(),
            &parsed.axis_labels,
            &parsed.unique_labels,
        )
        .unwrap();
        assert!(matches!(standard.data, TensorData::DiagonalView { .. }));

        let transposed = array.view().reversed_axes().into_dyn();
        let nonstandard =
            diagonal_tensor(transposed, &parsed.axis_labels, &parsed.unique_labels).unwrap();
        assert!(matches!(nonstandard.data, TensorData::DiagonalOwned { .. }));
    }

    #[test]
    fn reusable_plan_precomputes_operand_and_pair_partitions() {
        let expression = parse::parse("ij,jk->ik", &[&[2, 3], &[3, 4]]).unwrap();
        let path = plan::build_path(&expression, Strategy::Auto).unwrap();
        let execution = build_execution_plan(&expression, &path).unwrap();
        let pair = &execution.steps[0].pairs[0];
        assert_eq!(pair.contracted.len(), 1);
        assert_eq!(pair.left_only.len(), 1);
        assert_eq!(pair.right_only.len(), 1);
    }
}
