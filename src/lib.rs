#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Einstein summation for dynamically shaped `ndarray` arrays.
//!
//! Integer additions and multiplications use wrapping arithmetic in every
//! build profile. Other element types use their `ndarray` arithmetic.

mod execute;
mod parse;
mod plan;

use std::error::Error;
use std::fmt;

use ndarray::{ArrayD, ArrayViewD, LinalgScalar};

use parse::ParsedExpression;

/// Selects how a contraction path is built.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum Strategy {
    /// Searches all contraction trees for up to eight operands and uses a
    /// greedy search above that limit.
    #[default]
    Auto,
    /// Searches all contraction trees without the eight-operand cutoff.
    ///
    /// Expressions above eight operands currently use the greedy search to
    /// keep planning time bounded.
    Optimal,
    /// Repeatedly contracts the pair with the lowest greedy score.
    Greedy,
    /// Uses caller-supplied current-list indices.
    ///
    /// Each step removes its operands and appends its intermediate.
    Explicit(Vec<Vec<usize>>),
}

/// Describes an ordered contraction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractionPath {
    /// Steps in execution order.
    pub steps: Vec<ContractionStep>,
    /// Cost of evaluating every operand in one expression.
    pub naive_flops: u128,
    /// Sum of the selected step costs.
    pub optimized_flops: u128,
    /// Largest intermediate element count.
    pub largest_intermediate: u128,
}

/// Describes one contraction step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractionStep {
    /// Indices into the operand list as it exists before this step.
    pub operands: Vec<usize>,
    /// Fully resolved input and output labels for this step.
    pub subscripts: String,
    /// Cost assigned by the contraction model.
    pub flops: u128,
    /// Whether execution uses `ndarray` matrix multiplication for this step.
    pub gemm: bool,
}

/// Reports invalid expressions, shapes, and paths.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EinsumError {
    /// A byte outside the subscript grammar was found.
    InvalidCharacter {
        /// Byte position in the supplied string.
        position: usize,
        /// Invalid byte.
        byte: u8,
    },
    /// An arrow or another grammar element is malformed.
    MalformedSubscripts {
        /// Byte position in the supplied string.
        position: usize,
        /// Short explanation of the grammar error.
        reason: &'static str,
    },
    /// The number of input groups differs from the operand count.
    OperandCountMismatch {
        /// Number of input groups.
        subscripts: usize,
        /// Number of operand shapes or arrays.
        operands: usize,
    },
    /// An operand group names more axes than the operand has.
    TooManyLabels {
        /// Zero-based operand index.
        operand: usize,
        /// Number of explicit labels.
        labels: usize,
        /// Operand rank.
        ndim: usize,
    },
    /// An operand without an ellipsis names fewer axes than its rank.
    TooFewLabels {
        /// Zero-based operand index.
        operand: usize,
        /// Number of labels.
        labels: usize,
        /// Operand rank.
        ndim: usize,
    },
    /// Repeated axes in one operand have different lengths.
    DiagonalSizeMismatch {
        /// Zero-based operand index.
        operand: usize,
        /// Repeated label.
        label: char,
        /// First and conflicting lengths.
        sizes: (usize, usize),
    },
    /// Two axes cannot broadcast to one label size.
    BroadcastMismatch {
        /// Named label, or `None` for an ellipsis axis.
        label: Option<char>,
        /// Conflicting lengths.
        sizes: (usize, usize),
    },
    /// An explicit output repeats a label.
    OutputLabelRepeated {
        /// Repeated label.
        label: char,
    },
    /// An explicit output names a label absent from every input.
    OutputLabelUnknown {
        /// Unknown label.
        label: char,
    },
    /// An explicit output omits broadcast axes.
    OutputEllipsisMissing {
        /// Number of omitted broadcast axes.
        broadcast_rank: usize,
    },
    /// A required array size does not fit in `usize`.
    SizeOverflow,
    /// Execution operands differ from the shapes bound to a plan.
    ShapeMismatch {
        /// Zero-based operand index.
        operand: usize,
        /// Shape stored in the plan.
        expected: Vec<usize>,
        /// Shape supplied for execution.
        actual: Vec<usize>,
    },
    /// A supplied path cannot reduce the operand list to one value.
    InvalidPath {
        /// Short explanation of the path error.
        reason: &'static str,
    },
}

impl fmt::Display for EinsumError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCharacter { position, byte } => {
                write!(formatter, "invalid byte {byte:#04x} at position {position}")
            }
            Self::MalformedSubscripts { position, reason } => {
                write!(formatter, "malformed subscripts at position {position}: {reason}")
            }
            Self::OperandCountMismatch {
                subscripts,
                operands,
            } => write!(
                formatter,
                "subscripts contain {subscripts} operands but {operands} were supplied"
            ),
            Self::TooManyLabels {
                operand,
                labels,
                ndim,
            } => write!(
                formatter,
                "operand {operand} has rank {ndim} but its subscript has {labels} labels"
            ),
            Self::TooFewLabels {
                operand,
                labels,
                ndim,
            } => write!(
                formatter,
                "operand {operand} has rank {ndim} but its subscript has {labels} labels and no ellipsis"
            ),
            Self::DiagonalSizeMismatch {
                operand,
                label,
                sizes,
            } => write!(
                formatter,
                "operand {operand} repeats label {label} on axes of lengths {} and {}",
                sizes.0, sizes.1
            ),
            Self::BroadcastMismatch { label, sizes } => match label {
                Some(label) => write!(
                    formatter,
                    "label {label} cannot broadcast lengths {} and {}",
                    sizes.0, sizes.1
                ),
                None => write!(
                    formatter,
                    "ellipsis axis cannot broadcast lengths {} and {}",
                    sizes.0, sizes.1
                ),
            },
            Self::OutputLabelRepeated { label } => {
                write!(formatter, "output label {label} appears more than once")
            }
            Self::OutputLabelUnknown { label } => {
                write!(formatter, "output label {label} does not appear in an input")
            }
            Self::OutputEllipsisMissing { broadcast_rank } => write!(
                formatter,
                "output omits an ellipsis with rank {broadcast_rank}"
            ),
            Self::SizeOverflow => formatter.write_str("array size exceeds usize"),
            Self::ShapeMismatch {
                operand,
                expected,
                actual,
            } => write!(
                formatter,
                "operand {operand} has shape {actual:?}, expected {expected:?}"
            ),
            Self::InvalidPath { reason } => write!(formatter, "invalid path: {reason}"),
        }
    }
}

impl Error for EinsumError {}

/// A parsed contraction bound to fixed operand shapes.
#[derive(Clone, Debug)]
pub struct EinsumPlan {
    expression: ParsedExpression,
    path: ContractionPath,
    execution: execute::ExecutionPlan,
    output_subscripts: String,
}

impl EinsumPlan {
    /// Parses and plans an expression for fixed operand shapes.
    pub fn new(subscripts: &str, shapes: &[&[usize]]) -> Result<Self, EinsumError> {
        Self::with_strategy(subscripts, shapes, Strategy::Auto)
    }

    /// Parses and plans an expression with a selected path strategy.
    pub fn with_strategy(
        subscripts: &str,
        shapes: &[&[usize]],
        strategy: Strategy,
    ) -> Result<Self, EinsumError> {
        let expression = parse::parse(subscripts, shapes)?;
        let path = plan::build_path(&expression, strategy)?;
        let execution = execute::build_execution_plan(&expression, &path)?;
        let output_subscripts = parse::format_labels(&expression.output, expression.broadcast_rank);
        Ok(Self {
            expression,
            path,
            execution,
            output_subscripts,
        })
    }

    /// Executes the plan on arrays with the planned shapes.
    pub fn execute<A: LinalgScalar>(
        &self,
        operands: &[ArrayViewD<'_, A>],
    ) -> Result<ArrayD<A>, EinsumError> {
        execute::execute(self, operands)
    }

    /// Returns the result shape.
    pub fn output_shape(&self) -> &[usize] {
        &self.expression.output_shape
    }

    /// Returns the selected contraction path.
    pub fn path(&self) -> &ContractionPath {
        &self.path
    }

    /// Returns the resolved result labels.
    ///
    /// A broadcast block is written as `...`.
    pub fn output_subscripts(&self) -> &str {
        &self.output_subscripts
    }
}

/// Parses, plans, and evaluates one expression.
pub fn einsum<A: LinalgScalar>(
    subscripts: &str,
    operands: &[ArrayViewD<'_, A>],
) -> Result<ArrayD<A>, EinsumError> {
    let shapes: Vec<&[usize]> = operands.iter().map(ArrayViewD::shape).collect();
    EinsumPlan::new(subscripts, &shapes)?.execute(operands)
}

/// Parses and plans one expression from shapes alone.
pub fn einsum_path(subscripts: &str, shapes: &[&[usize]]) -> Result<ContractionPath, EinsumError> {
    Ok(EinsumPlan::new(subscripts, shapes)?.path)
}
