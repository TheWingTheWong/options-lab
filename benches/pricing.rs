//! Criterion benchmarks for the pricing kernels.
//!
//! ```text
//! cargo bench
//! cargo bench -- implied_vol    # one group
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use options_lab::binomial::BinomialTree;
use options_lab::monte_carlo::MonteCarlo;
use options_lab::smile::{self, Market, Selection};
use options_lab::{OptionParams, OptionType, bsm, iv};

fn atm() -> OptionParams {
    OptionParams::new(100.0, 100.0, 0.5)
        .with_rate(0.03)
        .with_dividend(0.01)
        .with_vol(0.25)
}

fn closed_form(c: &mut Criterion) {
    let p = atm();
    let mut g = c.benchmark_group("bsm");
    g.bench_function("price", |b| {
        b.iter(|| bsm::price(OptionType::Call, black_box(&p)))
    });
    g.bench_function("greeks", |b| {
        b.iter(|| bsm::greeks(OptionType::Call, black_box(&p)))
    });
    g.finish();
}

fn implied_vol(c: &mut Criterion) {
    let mut g = c.benchmark_group("implied_vol");
    let cases = [
        ("atm_6m", OptionParams::new(100.0, 100.0, 0.5), 0.25),
        (
            "otm_call_30d",
            OptionParams::new(100.0, 110.0, 30.0 / 365.0),
            0.22,
        ),
        ("deep_otm_put_1y", OptionParams::new(100.0, 40.0, 1.0), 0.45),
        (
            "deep_itm_call_3m",
            OptionParams::new(100.0, 60.0, 0.25),
            0.35,
        ),
    ];
    for (name, p, vol) in cases {
        let p = p.with_rate(0.03);
        let kind = if name.contains("put") {
            OptionType::Put
        } else {
            OptionType::Call
        };
        let quote = bsm::price(kind, &p.with_vol(vol)).unwrap();
        g.bench_function(name, |b| {
            b.iter(|| iv::implied_vol(kind, black_box(quote), black_box(&p)))
        });
    }
    g.finish();
}

fn binomial(c: &mut Criterion) {
    let p = atm();
    let mut g = c.benchmark_group("crr_american_put");
    for steps in [100, 500, 2000] {
        g.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, &n| {
            b.iter(|| BinomialTree::american(n).price(OptionType::Put, black_box(&p)))
        });
    }
    g.finish();
}

fn monte_carlo(c: &mut Criterion) {
    let p = atm();
    let mut g = c.benchmark_group("monte_carlo_100k");
    g.sample_size(20);
    g.bench_function("plain", |b| {
        b.iter(|| MonteCarlo::plain(100_000, 42).price(OptionType::Call, black_box(&p)))
    });
    g.bench_function("antithetic_cv", |b| {
        let mc = MonteCarlo {
            paths: 100_000,
            ..MonteCarlo::default()
        };
        b.iter(|| mc.price(OptionType::Call, black_box(&p)))
    });
    g.finish();
}

fn smile_build(c: &mut Criterion) {
    let text = include_str!("../examples/chain.csv");
    let chain = smile::parse_chain(text).unwrap();
    let market = Market {
        spot: 187.4,
        rate: 0.043,
        dividend: 0.006,
        expiry: 45.0 / 365.0,
    };
    c.bench_function("smile/parse_and_fit_42_quotes", |b| {
        b.iter(|| {
            let chain = smile::parse_chain(black_box(text)).unwrap();
            smile::build_smile(&chain, market, Selection::OutOfTheMoney)
        })
    });
    c.bench_function("smile/fit_only", |b| {
        b.iter(|| smile::build_smile(black_box(&chain), market, Selection::OutOfTheMoney))
    });
}

criterion_group!(
    benches,
    closed_form,
    implied_vol,
    binomial,
    monte_carlo,
    smile_build
);
criterion_main!(benches);
