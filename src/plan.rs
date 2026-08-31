use crate::parse::{label_char, Label, ParsedExpression, NAMED_LABELS};
use crate::{ContractionPath, ContractionStep, EinsumError, Strategy};

#[derive(Clone, Debug)]
struct PlanTensor {
    labels: Vec<Label>,
    members: usize,
}

#[derive(Clone, Debug)]
struct SearchState {
    cost: u128,
    split: Option<(usize, usize)>,
}

#[derive(Clone, Copy, Debug)]
struct GreedyCandidate {
    left: usize,
    right: usize,
    cost: u128,
    input_size: u128,
}

pub(crate) fn build_path(
    expression: &ParsedExpression,
    strategy: Strategy,
) -> Result<ContractionPath, EinsumError> {
    let naive_flops = naive_cost(expression);
    let mut path = match strategy {
        Strategy::Auto | Strategy::Optimal if expression.operands.len() <= 8 => {
            optimal_path(expression)?
        }
        Strategy::Auto | Strategy::Optimal | Strategy::Greedy => greedy_path(expression)?,
        Strategy::Explicit(steps) => explicit_path(expression, steps)?,
    };
    path.naive_flops = naive_flops;
    Ok(path)
}

fn naive_cost(expression: &ParsedExpression) -> u128 {
    let groups: Vec<&[Label]> = expression
        .operands
        .iter()
        .map(|operand| operand.unique_labels.as_slice())
        .collect();
    let union = union_in_order(&groups);
    let distinct_total: usize = groups.iter().map(|labels| labels.len()).sum();
    let inner = distinct_total > union.len();
    let factor = expression
        .operands
        .len()
        .saturating_sub(1)
        .saturating_add(usize::from(inner));
    saturating_mul(product_labels(&union, &expression.sizes), factor as u128)
}

fn optimal_path(expression: &ParsedExpression) -> Result<ContractionPath, EinsumError> {
    let count = expression.operands.len();
    if count <= 1 {
        return Ok(empty_path(expression));
    }

    let all_input_labels: Vec<&[Label]> = expression
        .operands
        .iter()
        .map(|operand| operand.unique_labels.as_slice())
        .collect();
    let union = union_in_order(&all_input_labels);
    if union.iter().all(|label| expression.output.contains(label)) {
        return direct_path(expression);
    }

    let total = 1usize
        .checked_shl(u32::try_from(count).map_err(|_| EinsumError::SizeOverflow)?)
        .ok_or(EinsumError::SizeOverflow)?;
    let full = total - 1;
    let mut labels = Vec::with_capacity(total);
    labels.push(Vec::new());
    for mask in 1..total {
        labels.push(labels_for_subset(expression, mask));
    }

    let mut states: Vec<Option<SearchState>> = vec![None; total];
    for operand in 0..count {
        if let Some(slot) = states.get_mut(1usize << operand) {
            *slot = Some(SearchState {
                cost: 0,
                split: None,
            });
        }
    }

    for mask in 1..total {
        if mask.is_power_of_two() {
            continue;
        }
        let lowest = mask & mask.wrapping_neg();
        let mut left = (mask - 1) & mask;
        let mut best: Option<SearchState> = None;
        while left != 0 {
            let right = mask ^ left;
            if right != 0 && left & lowest != 0 {
                let left_cost = states
                    .get(left)
                    .and_then(Option::as_ref)
                    .map(|state| state.cost);
                let right_cost = states
                    .get(right)
                    .and_then(Option::as_ref)
                    .map(|state| state.cost);
                if let (Some(left_cost), Some(right_cost)) = (left_cost, right_cost) {
                    let step_cost = cost_for_groups(
                        &[
                            labels.get(left).map(Vec::as_slice).unwrap_or(&[]),
                            labels.get(right).map(Vec::as_slice).unwrap_or(&[]),
                        ],
                        labels.get(mask).map(Vec::as_slice).unwrap_or(&[]),
                        &expression.sizes,
                    );
                    let cost = saturating_add(saturating_add(left_cost, right_cost), step_cost);
                    if best.as_ref().map_or(true, |current| cost < current.cost) {
                        best = Some(SearchState {
                            cost,
                            split: Some((left, right)),
                        });
                    }
                }
            }
            left = (left - 1) & mask;
        }
        if let Some(slot) = states.get_mut(mask) {
            *slot = best;
        }
    }

    let optimized_flops = states
        .get(full)
        .and_then(Option::as_ref)
        .map(|state| state.cost)
        .ok_or(EinsumError::InvalidPath {
            reason: "no contraction tree was found",
        })?;
    let mut active: Vec<PlanTensor> = expression
        .operands
        .iter()
        .enumerate()
        .map(|(index, operand)| PlanTensor {
            labels: operand.unique_labels.clone(),
            members: 1usize << index,
        })
        .collect();
    let mut steps = Vec::new();
    let mut largest = 0;
    emit_subset(
        full,
        &states,
        &labels,
        &expression.sizes,
        expression.broadcast_rank,
        &mut active,
        &mut steps,
        &mut largest,
    )?;
    Ok(ContractionPath {
        steps,
        naive_flops: 0,
        optimized_flops,
        largest_intermediate: largest,
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_subset(
    mask: usize,
    states: &[Option<SearchState>],
    labels: &[Vec<Label>],
    sizes: &[usize],
    broadcast_rank: usize,
    active: &mut Vec<PlanTensor>,
    steps: &mut Vec<ContractionStep>,
    largest: &mut u128,
) -> Result<(), EinsumError> {
    if mask.is_power_of_two() {
        return Ok(());
    }
    let (left, right) = states
        .get(mask)
        .and_then(Option::as_ref)
        .and_then(|state| state.split)
        .ok_or(EinsumError::InvalidPath {
            reason: "contraction tree is incomplete",
        })?;
    emit_subset(
        left,
        states,
        labels,
        sizes,
        broadcast_rank,
        active,
        steps,
        largest,
    )?;
    emit_subset(
        right,
        states,
        labels,
        sizes,
        broadcast_rank,
        active,
        steps,
        largest,
    )?;

    let left_index = active
        .iter()
        .position(|tensor| tensor.members == left)
        .ok_or(EinsumError::InvalidPath {
            reason: "left intermediate is missing",
        })?;
    let right_index = active
        .iter()
        .position(|tensor| tensor.members == right)
        .ok_or(EinsumError::InvalidPath {
            reason: "right intermediate is missing",
        })?;
    let mut operands = vec![left_index, right_index];
    operands.sort_unstable();
    let result = labels.get(mask).cloned().unwrap_or_default();
    let groups = [
        active
            .get(left_index)
            .map(|tensor| tensor.labels.as_slice())
            .unwrap_or(&[]),
        active
            .get(right_index)
            .map(|tensor| tensor.labels.as_slice())
            .unwrap_or(&[]),
    ];
    let flops = cost_for_groups(&groups, &result, sizes);
    let gemm = step_uses_gemm(&groups, &result);
    let subscripts = format_step(&groups, &result, broadcast_rank);
    for &index in operands.iter().rev() {
        if index >= active.len() {
            return Err(EinsumError::InvalidPath {
                reason: "intermediate index is out of range",
            });
        }
        active.remove(index);
    }
    active.push(PlanTensor {
        labels: result.clone(),
        members: mask,
    });
    *largest = (*largest).max(product_labels(&result, sizes));
    steps.push(ContractionStep {
        operands,
        subscripts,
        flops,
        gemm,
    });
    Ok(())
}

fn greedy_path(expression: &ParsedExpression) -> Result<ContractionPath, EinsumError> {
    if expression.operands.len() <= 1 {
        return Ok(empty_path(expression));
    }
    let mut active = initial_tensors(expression);
    let mut steps = Vec::new();
    let mut cost = 0u128;
    let mut largest = 0u128;
    while active.len() > 1 {
        let mut best: Option<GreedyCandidate> = None;
        for left in 0..active.len() {
            for right in left + 1..active.len() {
                let selection = [left, right];
                let result = result_labels(&active, &selection, &expression.output);
                let groups = [
                    active
                        .get(left)
                        .map(|tensor| tensor.labels.as_slice())
                        .unwrap_or(&[]),
                    active
                        .get(right)
                        .map(|tensor| tensor.labels.as_slice())
                        .unwrap_or(&[]),
                ];
                let candidate = GreedyCandidate {
                    left,
                    right,
                    cost: cost_for_groups(&groups, &result, &expression.sizes),
                    input_size: saturating_add(
                        product_labels(groups[0], &expression.sizes),
                        product_labels(groups[1], &expression.sizes),
                    ),
                };
                if best.map_or(true, |current| greedy_is_better(candidate, current)) {
                    best = Some(candidate);
                }
            }
        }
        let best = best.ok_or(EinsumError::InvalidPath {
            reason: "greedy search found no pair",
        })?;
        let flops = apply_step(
            &mut active,
            &[best.left, best.right],
            &expression.output,
            &expression.sizes,
            expression.broadcast_rank,
            &mut steps,
            &mut largest,
        )?;
        cost = saturating_add(cost, flops);
    }
    let direct = direct_path(expression)?;
    if cost > direct.optimized_flops {
        return Ok(direct);
    }
    Ok(ContractionPath {
        steps,
        naive_flops: 0,
        optimized_flops: cost,
        largest_intermediate: largest,
    })
}

fn greedy_is_better(candidate: GreedyCandidate, current: GreedyCandidate) -> bool {
    let candidate_side = saturating_add(candidate.cost, current.input_size);
    let current_side = saturating_add(current.cost, candidate.input_size);
    candidate_side < current_side
        || (candidate_side == current_side
            && (candidate.left, candidate.right) < (current.left, current.right))
}

fn explicit_path(
    expression: &ParsedExpression,
    requested: Vec<Vec<usize>>,
) -> Result<ContractionPath, EinsumError> {
    let mut active = initial_tensors(expression);
    let mut steps = Vec::new();
    let mut cost = 0u128;
    let mut largest = 0u128;
    for operands in requested {
        if operands.len() < 2 {
            return Err(EinsumError::InvalidPath {
                reason: "each step needs at least two operands",
            });
        }
        let mut sorted = operands.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair.first() == pair.get(1)) {
            return Err(EinsumError::InvalidPath {
                reason: "a step repeats an operand index",
            });
        }
        if sorted.iter().any(|&index| index >= active.len()) {
            return Err(EinsumError::InvalidPath {
                reason: "a step index is out of range",
            });
        }
        let flops = apply_step(
            &mut active,
            &operands,
            &expression.output,
            &expression.sizes,
            expression.broadcast_rank,
            &mut steps,
            &mut largest,
        )?;
        cost = saturating_add(cost, flops);
    }
    if active.len() != 1 {
        return Err(EinsumError::InvalidPath {
            reason: "path must leave exactly one operand",
        });
    }
    Ok(ContractionPath {
        steps,
        naive_flops: 0,
        optimized_flops: cost,
        largest_intermediate: largest,
    })
}

fn direct_path(expression: &ParsedExpression) -> Result<ContractionPath, EinsumError> {
    if expression.operands.len() <= 1 {
        return Ok(empty_path(expression));
    }
    let mut active = initial_tensors(expression);
    let operands: Vec<usize> = (0..active.len()).collect();
    let mut steps = Vec::new();
    let mut largest = 0;
    let cost = apply_step(
        &mut active,
        &operands,
        &expression.output,
        &expression.sizes,
        expression.broadcast_rank,
        &mut steps,
        &mut largest,
    )?;
    Ok(ContractionPath {
        steps,
        naive_flops: 0,
        optimized_flops: cost,
        largest_intermediate: largest,
    })
}

fn empty_path(expression: &ParsedExpression) -> ContractionPath {
    ContractionPath {
        steps: Vec::new(),
        naive_flops: 0,
        optimized_flops: 0,
        largest_intermediate: product_labels(&expression.output, &expression.sizes),
    }
}

fn initial_tensors(expression: &ParsedExpression) -> Vec<PlanTensor> {
    expression
        .operands
        .iter()
        .enumerate()
        .map(|(index, operand)| PlanTensor {
            labels: operand.unique_labels.clone(),
            members: 1usize.checked_shl(index as u32).unwrap_or(0),
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn apply_step(
    active: &mut Vec<PlanTensor>,
    operands: &[usize],
    output: &[Label],
    sizes: &[usize],
    broadcast_rank: usize,
    steps: &mut Vec<ContractionStep>,
    largest: &mut u128,
) -> Result<u128, EinsumError> {
    let result = result_labels(active, operands, output);
    let selected: Vec<Vec<Label>> = operands
        .iter()
        .filter_map(|&index| active.get(index).map(|tensor| tensor.labels.clone()))
        .collect();
    if selected.len() != operands.len() {
        return Err(EinsumError::InvalidPath {
            reason: "a step index is out of range",
        });
    }
    let groups: Vec<&[Label]> = selected.iter().map(Vec::as_slice).collect();
    let flops = cost_for_groups(&groups, &result, sizes);
    let gemm = step_uses_gemm(&groups, &result);
    let subscripts = format_step(&groups, &result, broadcast_rank);
    let members = operands
        .iter()
        .filter_map(|&index| active.get(index))
        .fold(0usize, |mask, tensor| mask | tensor.members);
    let mut removal = operands.to_vec();
    removal.sort_unstable();
    for &index in removal.iter().rev() {
        if index >= active.len() {
            return Err(EinsumError::InvalidPath {
                reason: "a step index is out of range",
            });
        }
        active.remove(index);
    }
    active.push(PlanTensor {
        labels: result.clone(),
        members,
    });
    *largest = (*largest).max(product_labels(&result, sizes));
    steps.push(ContractionStep {
        operands: operands.to_vec(),
        subscripts,
        flops,
        gemm,
    });
    Ok(flops)
}

fn result_labels(active: &[PlanTensor], operands: &[usize], output: &[Label]) -> Vec<Label> {
    let selected: Vec<&[Label]> = operands
        .iter()
        .filter_map(|&index| active.get(index).map(|tensor| tensor.labels.as_slice()))
        .collect();
    let union = union_in_order(&selected);
    let remaining: Vec<Label> = active
        .iter()
        .enumerate()
        .filter(|(index, _)| !operands.contains(index))
        .flat_map(|(_, tensor)| tensor.labels.iter().copied())
        .collect();
    union
        .into_iter()
        .filter(|label| output.contains(label) || remaining.contains(label))
        .collect()
}

fn labels_for_subset(expression: &ParsedExpression, mask: usize) -> Vec<Label> {
    let selected: Vec<&[Label]> = expression
        .operands
        .iter()
        .enumerate()
        .filter_map(|(index, operand)| {
            ((mask & (1usize << index)) != 0).then_some(operand.unique_labels.as_slice())
        })
        .collect();
    let union = union_in_order(&selected);
    if mask.is_power_of_two() {
        return union;
    }
    let remaining: Vec<Label> = expression
        .operands
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1usize << index) == 0)
        .flat_map(|(_, operand)| operand.unique_labels.iter().copied())
        .collect();
    union
        .into_iter()
        .filter(|label| expression.output.contains(label) || remaining.contains(label))
        .collect()
}

fn cost_for_groups(groups: &[&[Label]], result: &[Label], sizes: &[usize]) -> u128 {
    let union = union_in_order(groups);
    let removes_label = union.iter().any(|label| !result.contains(label));
    let factor = groups
        .len()
        .saturating_sub(1)
        .saturating_add(usize::from(removes_label));
    saturating_mul(product_labels(&union, sizes), factor as u128)
}

fn step_uses_gemm(groups: &[&[Label]], result: &[Label]) -> bool {
    let union = union_in_order(groups);
    union.iter().any(|label| {
        !result.contains(label) && groups.iter().filter(|group| group.contains(label)).count() >= 2
    })
}

fn union_in_order(groups: &[&[Label]]) -> Vec<Label> {
    let mut union = Vec::new();
    for group in groups {
        for &label in *group {
            if !union.contains(&label) {
                union.push(label);
            }
        }
    }
    union
}

fn product_labels(labels: &[Label], sizes: &[usize]) -> u128 {
    labels.iter().fold(1u128, |product, &label| {
        let size = sizes.get(label).copied().unwrap_or(1) as u128;
        saturating_mul(product, size)
    })
}

fn saturating_mul(left: u128, right: u128) -> u128 {
    left.saturating_mul(right)
}

fn saturating_add(left: u128, right: u128) -> u128 {
    left.saturating_add(right)
}

fn format_step(groups: &[&[Label]], result: &[Label], broadcast_rank: usize) -> String {
    let mut text = String::new();
    for (index, group) in groups.iter().enumerate() {
        if index > 0 {
            text.push(',');
        }
        text.push_str(&format_labels(group, broadcast_rank));
    }
    text.push_str("->");
    text.push_str(&format_labels(result, broadcast_rank));
    text
}

fn format_labels(labels: &[Label], broadcast_rank: usize) -> String {
    let mut text = String::new();
    let mut wrote_ellipsis = false;
    for &label in labels {
        if label >= NAMED_LABELS && label < NAMED_LABELS + broadcast_rank {
            if !wrote_ellipsis {
                text.push_str("...");
                wrote_ellipsis = true;
            }
        } else if let Some(character) = label_char(label) {
            text.push(character);
        }
    }
    text
}
