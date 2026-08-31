use ndarray::{ArrayD, ArrayViewD, IxDyn, LinalgScalar};

const NAMED_LABELS: usize = 52;

pub fn evaluate<A: LinalgScalar>(subscripts: &str, operands: &[ArrayViewD<'_, A>]) -> ArrayD<A> {
    let compact: String = subscripts
        .chars()
        .filter(|character| *character != ' ')
        .collect();
    let mut sides = compact.split("->");
    let inputs = sides.next().unwrap();
    let explicit_output = sides.next();
    let groups: Vec<&str> = inputs.split(',').collect();
    let ranks: Vec<usize> = groups
        .iter()
        .zip(operands)
        .map(|(group, operand)| {
            if group.contains("...") {
                operand.ndim()
                    - group
                        .chars()
                        .filter(|character| character.is_ascii_alphabetic())
                        .count()
            } else {
                0
            }
        })
        .collect();
    let broadcast_rank = ranks.iter().copied().max().unwrap_or(0);
    let axis_labels: Vec<Vec<usize>> = groups
        .iter()
        .zip(ranks.iter().copied())
        .map(|(group, rank)| expand_group(group, broadcast_rank, rank))
        .collect();
    let mut sizes = vec![1usize; NAMED_LABELS + broadcast_rank];
    for (labels, operand) in axis_labels.iter().zip(operands) {
        for (&label, &size) in labels.iter().zip(operand.shape()) {
            if sizes[label] == 1 {
                sizes[label] = size;
            }
        }
    }
    let output = match explicit_output {
        Some(group) => expand_group(group, broadcast_rank, broadcast_rank),
        None => {
            let mut counts = vec![0usize; NAMED_LABELS];
            for labels in &axis_labels {
                for &label in labels {
                    if label < NAMED_LABELS {
                        counts[label] += 1;
                    }
                }
            }
            let mut labels: Vec<usize> = (0..broadcast_rank)
                .map(|axis| NAMED_LABELS + axis)
                .collect();
            labels.extend(
                counts
                    .iter()
                    .enumerate()
                    .filter_map(|(label, &count)| (count == 1).then_some(label)),
            );
            labels
        }
    };
    let mut union = Vec::new();
    for labels in &axis_labels {
        for &label in labels {
            if !union.contains(&label) {
                union.push(label);
            }
        }
    }
    let summed: Vec<usize> = union
        .into_iter()
        .filter(|label| !output.contains(label))
        .collect();
    let output_shape: Vec<usize> = output.iter().map(|&label| sizes[label]).collect();
    let output_len = output_shape.iter().copied().product();
    let mut values = Vec::with_capacity(output_len);
    for flat in 0..output_len {
        let mut assignment = vec![0usize; sizes.len()];
        let mut remaining = flat;
        for (&label, &size) in output.iter().zip(output_shape.iter()).rev() {
            if size != 0 {
                assignment[label] = remaining % size;
                remaining /= size;
            }
        }
        values.push(sum_assignments(
            0,
            &summed,
            &sizes,
            &mut assignment,
            operands,
            &axis_labels,
        ));
    }
    ArrayD::from_shape_vec(IxDyn(&output_shape), values).unwrap()
}

fn sum_assignments<A: LinalgScalar>(
    depth: usize,
    summed: &[usize],
    sizes: &[usize],
    assignment: &mut [usize],
    operands: &[ArrayViewD<'_, A>],
    axis_labels: &[Vec<usize>],
) -> A {
    if depth == summed.len() {
        return operands
            .iter()
            .zip(axis_labels)
            .fold(A::one(), |product, (operand, labels)| {
                let indices: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .map(|(axis, &label)| {
                        if operand.shape()[axis] == 1 {
                            0
                        } else {
                            assignment[label]
                        }
                    })
                    .collect();
                product * *operand.get(IxDyn(&indices)).unwrap()
            });
    }
    let label = summed[depth];
    let mut sum = A::zero();
    for index in 0..sizes[label] {
        assignment[label] = index;
        sum = sum + sum_assignments(depth + 1, summed, sizes, assignment, operands, axis_labels);
    }
    sum
}

fn expand_group(group: &str, broadcast_rank: usize, rank: usize) -> Vec<usize> {
    let mut labels = Vec::new();
    let bytes = group.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor..].starts_with(b"...") {
            labels.extend((broadcast_rank - rank..broadcast_rank).map(|axis| NAMED_LABELS + axis));
            cursor += 3;
        } else {
            labels.push(named_label(bytes[cursor]));
            cursor += 1;
        }
    }
    labels
}

fn named_label(byte: u8) -> usize {
    match byte {
        b'A'..=b'Z' => usize::from(byte - b'A'),
        b'a'..=b'z' => 26 + usize::from(byte - b'a'),
        _ => unreachable!(),
    }
}
