use std::path::Path;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_reference_evaluator)");
    if Path::new("tests/common/naive.rs").is_file() {
        println!("cargo:rustc-cfg=has_reference_evaluator");
    }
}
