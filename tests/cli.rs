//! End-to-end tests of the `options-lab` binary.

use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_options-lab"))
        .args(args)
        .output()
        .expect("binary runs")
}

fn stdout(args: &[&str]) -> String {
    let out = run(args);
    assert!(
        out.status.success(),
        "options-lab {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn price_reports_bsm_greeks_and_model_comparison() {
    let out = stdout(&[
        "price", "--spot", "100", "--strike", "105", "--vol", "0.2", "--rate", "0.03", "--days",
        "30", "--type", "call",
    ]);
    assert!(out.contains("price   0.695442"), "{out}");
    assert!(out.contains("delta   0.217917"), "{out}");
    assert!(out.contains("CRR tree, American"), "{out}");
    assert!(out.contains("Monte Carlo"), "{out}");
}

#[test]
fn bsm_only_skips_comparison() {
    let out = stdout(&[
        "price",
        "--spot",
        "42",
        "--strike",
        "40",
        "--vol",
        "0.2",
        "--rate",
        "0.1",
        "--years",
        "0.5",
        "--type",
        "put",
        "--bsm-only",
    ]);
    assert!(out.contains("price   0.808599"), "{out}");
    assert!(!out.contains("Model comparison"));
}

#[test]
fn iv_recovers_volatility() {
    let out = stdout(&[
        "iv", "--spot", "100", "--strike", "105", "--price", "0.695442", "--rate", "0.03",
        "--days", "30", "--type", "call",
    ]);
    assert!(out.contains("implied vol  20.0000%"), "{out}");
}

#[test]
fn parity_flags_violations() {
    let out = stdout(&[
        "parity", "--spot", "100", "--strike", "100", "--days", "365", "--call", "9", "--put", "5",
    ]);
    assert!(out.contains("VIOLATED: calls rich vs puts"), "{out}");
}

#[test]
fn arbitrage_violation_is_a_clean_error() {
    let out = run(&[
        "iv", "--spot", "100", "--strike", "50", "--price", "10", "--days", "30", "--type", "call",
    ]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no-arbitrage range"), "{err}");
}

#[test]
fn expiry_is_required() {
    let out = run(&[
        "price", "--spot", "100", "--strike", "100", "--vol", "0.2", "--type", "call",
    ]);
    assert!(!out.status.success());
}

#[test]
fn smile_fits_the_bundled_chain() {
    let chain = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/chain.csv");
    let out = stdout(&["smile", chain]);
    // The generating smile is 26% ATM, skew -0.42, curvature 1.10.
    assert!(out.contains("ATM vol 26.0"), "{out}");
    assert!(out.contains("skew -0.41"), "{out}");
    assert!(out.contains("0 outside the bid-ask"), "{out}");
    assert!(out.contains("(zero bid)"), "{out}");
    assert!(out.contains("implied vol %"), "{out}");
}

#[test]
fn smile_flag_overrides_file_metadata() {
    let chain = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/chain.csv");
    // A wrong dividend yield shifts the forward and breaks parity.
    let out = stdout(&["smile", chain, "--div", "0.05", "--no-chart"]);
    assert!(!out.contains("0 outside the bid-ask"), "{out}");
    assert!(!out.contains("implied vol %"));
}

#[test]
fn smile_reports_missing_file() {
    let out = run(&["smile", "does-not-exist.csv"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read"));
}
