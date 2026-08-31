#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

#[cfg(has_reference_evaluator)]
pub mod naive;

#[derive(Clone, Debug)]
pub struct OperandCase {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub data_int: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct ExpectedCase {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub data_int_scaled: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct PathCase {
    pub naive_flops: u128,
    pub naive_left_to_right_flops: u128,
    pub optimal_flops: u128,
}

#[derive(Clone, Debug)]
pub struct ConformanceCase {
    pub id: String,
    pub category: String,
    pub subscripts: String,
    pub operands: Vec<OperandCase>,
    pub expected: Option<ExpectedCase>,
    pub error_kind: Option<String>,
    pub resolved_output: Option<String>,
    pub path: Option<PathCase>,
}

pub fn load_cases() -> Vec<ConformanceCase> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/conformance-vectors.json");
    let text = fs::read_to_string(path).unwrap();
    let document: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(document["count"].as_u64(), Some(189));
    document["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(parse_case)
        .collect()
}

fn parse_case(value: &Value) -> ConformanceCase {
    let operands = value["operands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|operand| OperandCase {
            shape: usize_array(&operand["shape"]),
            data: f64_array(&operand["data"]),
            data_int: i64_array(&operand["data_int"]),
        })
        .collect();
    let expected = value.get("expected").map(|expected| ExpectedCase {
        shape: usize_array(&expected["shape"]),
        data: f64_array(&expected["data"]),
        data_int_scaled: i64_array(&expected["data_int_scaled"]),
    });
    let path = value.get("path").map(|path| PathCase {
        naive_flops: u128::from(path["naive_flops"].as_u64().unwrap()),
        naive_left_to_right_flops: u128::from(path["naive_left_to_right_flops"].as_u64().unwrap()),
        optimal_flops: u128::from(path["optimal"]["flops"].as_u64().unwrap()),
    });
    ConformanceCase {
        id: value["id"].as_str().unwrap().to_owned(),
        category: value["category"].as_str().unwrap().to_owned(),
        subscripts: value["subscripts"].as_str().unwrap().to_owned(),
        operands,
        expected,
        error_kind: value
            .get("error")
            .map(|error| error["kind"].as_str().unwrap().to_owned()),
        resolved_output: value
            .get("resolved_output")
            .map(|output| output.as_str().unwrap().to_owned()),
        path,
    }
}

fn usize_array(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|number| usize::try_from(number.as_u64().unwrap()).unwrap())
        .collect()
}

fn i64_array(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|number| number.as_i64().unwrap())
        .collect()
}

fn f64_array(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|number| number.as_f64().unwrap())
        .collect()
}

pub fn left_to_right_path(operand_count: usize) -> Vec<Vec<usize>> {
    if operand_count < 2 {
        return Vec::new();
    }
    let mut active: Vec<Option<usize>> = (0..operand_count).map(Some).collect();
    let mut path = Vec::new();
    let mut accumulated = 0;
    for next in 1..operand_count {
        let left = active
            .iter()
            .position(|entry| *entry == Some(accumulated))
            .unwrap();
        let right = active
            .iter()
            .position(|entry| *entry == Some(next))
            .unwrap();
        path.push(vec![left, right]);
        let mut removal = vec![left, right];
        removal.sort_unstable();
        for index in removal.into_iter().rev() {
            active.remove(index);
        }
        active.push(Some(accumulated));
        accumulated = 0;
    }
    path
}
