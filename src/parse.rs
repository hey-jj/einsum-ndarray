use std::collections::BTreeMap;

use crate::EinsumError;

pub(crate) type Label = usize;

pub(crate) const NAMED_LABELS: usize = 52;

#[derive(Clone, Debug)]
pub(crate) struct ParsedOperand {
    pub(crate) axis_labels: Vec<Label>,
    pub(crate) unique_labels: Vec<Label>,
}

#[derive(Clone, Debug)]
pub(crate) struct ParsedExpression {
    pub(crate) operands: Vec<ParsedOperand>,
    pub(crate) shapes: Vec<Vec<usize>>,
    pub(crate) output: Vec<Label>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) sizes: Vec<usize>,
    pub(crate) broadcast_rank: usize,
}

#[derive(Clone, Copy, Debug)]
enum Token {
    Label(Label),
    Ellipsis,
}

#[derive(Clone, Debug)]
struct Group {
    tokens: Vec<Token>,
    has_ellipsis: bool,
}

pub(crate) fn parse(
    subscripts: &str,
    shapes: &[&[usize]],
) -> Result<ParsedExpression, EinsumError> {
    let compact: Vec<(u8, usize)> = subscripts
        .bytes()
        .enumerate()
        .filter_map(|(position, byte)| (byte != b' ').then_some((byte, position)))
        .collect();
    let arrow = find_arrow(&compact)?;
    let input_end = arrow.unwrap_or(compact.len());
    let output_start = arrow.map(|position| position + 2);
    let input = compact
        .get(..input_end)
        .ok_or(EinsumError::MalformedSubscripts {
            position: subscripts.len(),
            reason: "output arrow is out of bounds",
        })?;
    let input_groups = parse_input_groups(input)?;

    if input_groups.len() != shapes.len() {
        return Err(EinsumError::OperandCountMismatch {
            subscripts: input_groups.len(),
            operands: shapes.len(),
        });
    }

    let output_group = match output_start {
        Some(start) => Some(parse_group(
            compact
                .get(start..)
                .ok_or(EinsumError::MalformedSubscripts {
                    position: subscripts.len(),
                    reason: "output group is out of bounds",
                })?,
            false,
        )?),
        None => None,
    };

    let mut ellipsis_ranks = Vec::with_capacity(shapes.len());
    for (operand, (group, shape)) in input_groups.iter().zip(shapes).enumerate() {
        let labels = group
            .tokens
            .iter()
            .filter(|token| matches!(token, Token::Label(_)))
            .count();
        if labels > shape.len() {
            return Err(EinsumError::TooManyLabels {
                operand,
                labels,
                ndim: shape.len(),
            });
        }
        if !group.has_ellipsis && labels < shape.len() {
            return Err(EinsumError::TooFewLabels {
                operand,
                labels,
                ndim: shape.len(),
            });
        }
        ellipsis_ranks.push(if group.has_ellipsis {
            shape.len() - labels
        } else {
            0
        });
    }

    let broadcast_rank = ellipsis_ranks.iter().copied().max().unwrap_or(0);
    let label_count = NAMED_LABELS
        .checked_add(broadcast_rank)
        .ok_or(EinsumError::SizeOverflow)?;
    let mut parsed_operands = Vec::with_capacity(shapes.len());
    for (operand, ((group, shape), ellipsis_rank)) in input_groups
        .iter()
        .zip(shapes)
        .zip(ellipsis_ranks.iter().copied())
        .enumerate()
    {
        let axis_labels = expand_input_group(group, broadcast_rank, ellipsis_rank);
        validate_diagonals(operand, &axis_labels, shape)?;
        parsed_operands.push(ParsedOperand {
            unique_labels: unique_in_order(&axis_labels),
            axis_labels,
        });
    }

    let mut resolved = vec![None; label_count];
    for (operand, shape) in parsed_operands.iter().zip(shapes) {
        for (&label, &size) in operand.axis_labels.iter().zip(shape.iter()) {
            let slot = resolved.get_mut(label).ok_or(EinsumError::SizeOverflow)?;
            match *slot {
                None => *slot = Some(size),
                Some(current) if current == size || size == 1 => {}
                Some(1) => *slot = Some(size),
                Some(current) => {
                    return Err(EinsumError::BroadcastMismatch {
                        label: label_char(label),
                        sizes: (current, size),
                    });
                }
            }
        }
    }
    let sizes: Vec<usize> = resolved.into_iter().map(|size| size.unwrap_or(1)).collect();

    let output = match output_group {
        Some(group) => resolve_explicit_output(&group, &parsed_operands, broadcast_rank)?,
        None => resolve_implicit_output(&parsed_operands, broadcast_rank),
    };
    let output_shape: Vec<usize> = output
        .iter()
        .map(|&label| sizes.get(label).copied().unwrap_or(1))
        .collect();
    output_shape.iter().try_fold(1usize, |size, &axis| {
        size.checked_mul(axis).ok_or(EinsumError::SizeOverflow)
    })?;

    Ok(ParsedExpression {
        operands: parsed_operands,
        shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
        output,
        output_shape,
        sizes,
        broadcast_rank,
    })
}

pub(crate) fn format_labels(labels: &[Label], broadcast_rank: usize) -> String {
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

fn find_arrow(compact: &[(u8, usize)]) -> Result<Option<usize>, EinsumError> {
    let mut arrow = None;
    let mut cursor = 0;
    while let Some(&(byte, position)) = compact.get(cursor) {
        if byte == b'-' {
            if compact.get(cursor + 1).map(|entry| entry.0) != Some(b'>') {
                return Err(EinsumError::MalformedSubscripts {
                    position,
                    reason: "'-' must be followed by '>'",
                });
            }
            if arrow.is_some() {
                return Err(EinsumError::MalformedSubscripts {
                    position,
                    reason: "more than one output arrow",
                });
            }
            arrow = Some(cursor);
            cursor += 2;
            continue;
        }
        if byte == b'>' {
            return Err(EinsumError::MalformedSubscripts {
                position,
                reason: "'>' must follow '-'",
            });
        }
        cursor += 1;
    }
    Ok(arrow)
}

fn parse_input_groups(input: &[(u8, usize)]) -> Result<Vec<Group>, EinsumError> {
    let mut groups = Vec::new();
    let mut start = 0;
    for (index, &(byte, _)) in input.iter().enumerate() {
        if byte == b',' {
            let group = input
                .get(start..index)
                .ok_or(EinsumError::MalformedSubscripts {
                    position: input.get(index).map(|entry| entry.1).unwrap_or(0),
                    reason: "input group is out of bounds",
                })?;
            groups.push(parse_group(group, true)?);
            start = index + 1;
        }
    }
    let group = input.get(start..).ok_or(EinsumError::MalformedSubscripts {
        position: input.last().map(|entry| entry.1 + 1).unwrap_or(0),
        reason: "input group is out of bounds",
    })?;
    groups.push(parse_group(group, true)?);
    Ok(groups)
}

fn parse_group(bytes: &[(u8, usize)], input: bool) -> Result<Group, EinsumError> {
    let mut tokens = Vec::new();
    let mut has_ellipsis = false;
    let mut cursor = 0;
    while let Some(&(byte, position)) = bytes.get(cursor) {
        if let Some(label) = named_label(byte) {
            tokens.push(Token::Label(label));
            cursor += 1;
            continue;
        }
        if byte == b'.' {
            let run = bytes
                .get(cursor..)
                .unwrap_or(&[])
                .iter()
                .take_while(|entry| entry.0 == b'.')
                .count();
            if run != 3 || has_ellipsis {
                return Err(EinsumError::InvalidCharacter { position, byte });
            }
            has_ellipsis = true;
            tokens.push(Token::Ellipsis);
            cursor += 3;
            continue;
        }
        if byte == b',' && !input {
            return Err(EinsumError::InvalidCharacter { position, byte });
        }
        return Err(EinsumError::InvalidCharacter { position, byte });
    }
    Ok(Group {
        tokens,
        has_ellipsis,
    })
}

fn expand_input_group(group: &Group, broadcast_rank: usize, rank: usize) -> Vec<Label> {
    let mut labels = Vec::with_capacity(group.tokens.len().saturating_sub(1).saturating_add(rank));
    for token in &group.tokens {
        match token {
            Token::Label(label) => labels.push(*label),
            Token::Ellipsis => {
                let start = broadcast_rank.saturating_sub(rank);
                labels.extend((start..broadcast_rank).map(|axis| NAMED_LABELS + axis));
            }
        }
    }
    labels
}

fn validate_diagonals(
    operand: usize,
    labels: &[Label],
    shape: &[usize],
) -> Result<(), EinsumError> {
    let mut first_sizes = BTreeMap::new();
    for (&label, &size) in labels.iter().zip(shape.iter()) {
        match first_sizes.get(&label).copied() {
            Some(first) if first != size => {
                return Err(EinsumError::DiagonalSizeMismatch {
                    operand,
                    label: label_char(label).unwrap_or('?'),
                    sizes: (first, size),
                });
            }
            Some(_) => {}
            None => {
                first_sizes.insert(label, size);
            }
        }
    }
    Ok(())
}

fn resolve_explicit_output(
    group: &Group,
    operands: &[ParsedOperand],
    broadcast_rank: usize,
) -> Result<Vec<Label>, EinsumError> {
    let known: Vec<Label> = operands
        .iter()
        .flat_map(|operand| operand.unique_labels.iter().copied())
        .collect();
    let mut output = Vec::new();
    for token in &group.tokens {
        match token {
            Token::Label(label) => {
                if output.contains(label) {
                    return Err(EinsumError::OutputLabelRepeated {
                        label: label_char(*label).unwrap_or('?'),
                    });
                }
                if !known.contains(label) {
                    return Err(EinsumError::OutputLabelUnknown {
                        label: label_char(*label).unwrap_or('?'),
                    });
                }
                output.push(*label);
            }
            Token::Ellipsis => output.extend((0..broadcast_rank).map(|axis| NAMED_LABELS + axis)),
        }
    }
    if broadcast_rank > 0 && !group.has_ellipsis {
        return Err(EinsumError::OutputEllipsisMissing { broadcast_rank });
    }
    Ok(output)
}

fn resolve_implicit_output(operands: &[ParsedOperand], broadcast_rank: usize) -> Vec<Label> {
    let mut counts = vec![0usize; NAMED_LABELS];
    for operand in operands {
        for &label in &operand.axis_labels {
            if let Some(count) = counts.get_mut(label) {
                *count += 1;
            }
        }
    }
    let mut output: Vec<Label> = (0..broadcast_rank)
        .map(|axis| NAMED_LABELS + axis)
        .collect();
    output.extend(
        counts
            .iter()
            .enumerate()
            .filter_map(|(label, &count)| (count == 1).then_some(label)),
    );
    output
}

fn unique_in_order(labels: &[Label]) -> Vec<Label> {
    let mut unique = Vec::new();
    for &label in labels {
        if !unique.contains(&label) {
            unique.push(label);
        }
    }
    unique
}

pub(crate) fn named_label(byte: u8) -> Option<Label> {
    match byte {
        b'A'..=b'Z' => Some(usize::from(byte - b'A')),
        b'a'..=b'z' => Some(26 + usize::from(byte - b'a')),
        _ => None,
    }
}

pub(crate) fn label_char(label: Label) -> Option<char> {
    if label < 26 {
        u8::try_from(label)
            .ok()
            .map(|value| char::from(b'A' + value))
    } else if label < NAMED_LABELS {
        u8::try_from(label - 26)
            .ok()
            .map(|value| char::from(b'a' + value))
    } else {
        None
    }
}
