//! Property-based tests: model-free identities and no-arbitrage bounds that
//! must hold for every valid input, not just hand-picked ones.

use options_lab::binomial::BinomialTree;
use options_lab::monte_carlo::MonteCarlo;
use options_lab::{OptionParams, OptionType, bsm, iv, parity};
use proptest::prelude::*;

/// Inputs spanning short-dated to long-dated, low to very high vol, and
/// strikes from deep ITM to deep OTM.
fn params() -> impl Strategy<Value = OptionParams> {
    (
        10.0..500.0_f64, // spot
        0.4..2.5_f64,    // strike / spot
        0.01..5.0_f64,   // expiry (years)
        -0.01..0.10_f64, // rate
        0.0..0.06_f64,   // dividend
        0.03..1.5_f64,   // vol
    )
        .prop_map(|(s, m, t, r, q, v)| {
            OptionParams::new(s, s * m, t)
                .with_rate(r)
                .with_dividend(q)
                .with_vol(v)
        })
}

/// Narrower region for the statistical Monte Carlo check (see below).
fn mc_params() -> impl Strategy<Value = OptionParams> {
    (
        10.0..500.0_f64,
        0.7..1.4_f64,
        0.05..3.0_f64,
        -0.01..0.10_f64,
        0.0..0.06_f64,
        0.1..0.8_f64,
    )
        .prop_map(|(s, m, t, r, q, v)| {
            OptionParams::new(s, s * m, t)
                .with_rate(r)
                .with_dividend(q)
                .with_vol(v)
        })
}

fn kind() -> impl Strategy<Value = OptionType> {
    prop_oneof![Just(OptionType::Call), Just(OptionType::Put)]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn put_call_parity_holds(p in params()) {
        let c = bsm::price(OptionType::Call, &p).unwrap();
        let q = bsm::price(OptionType::Put, &p).unwrap();
        let chk = parity::check(c, q, &p, 1e-10 * p.spot.max(p.strike)).unwrap();
        prop_assert!(chk.holds(), "gap {:e}", chk.gap);
    }

    #[test]
    fn price_respects_static_bounds(p in params(), k in kind()) {
        let v = bsm::price(k, &p).unwrap();
        let (pv_s, pv_k) = (p.spot * p.dividend_discount(), p.strike * p.discount());
        let lower = (k.sign() * (pv_s - pv_k)).max(0.0);
        let upper = match k { OptionType::Call => pv_s, OptionType::Put => pv_k };
        let eps = 1e-12 * upper;
        prop_assert!(v >= lower - eps && v <= upper + eps, "{v} not in [{lower}, {upper}]");
    }

    #[test]
    fn greeks_have_the_right_signs(p in params(), k in kind()) {
        let g = bsm::greeks(k, &p).unwrap();
        let dq = p.dividend_discount();
        match k {
            OptionType::Call => prop_assert!((-1e-15..=dq + 1e-15).contains(&g.delta)),
            OptionType::Put => prop_assert!((-dq - 1e-15..=1e-15).contains(&g.delta)),
        }
        prop_assert!(g.gamma >= 0.0);
        prop_assert!(g.vega >= 0.0);
        match k {
            OptionType::Call => prop_assert!(g.rho >= 0.0),
            OptionType::Put => prop_assert!(g.rho <= 0.0),
        }
    }

    #[test]
    fn price_is_monotone_in_vol_and_strike(p in params(), k in kind()) {
        let v0 = bsm::price(k, &p).unwrap();
        let v_up = bsm::price(k, &p.with_vol(p.vol * 1.05)).unwrap();
        prop_assert!(v_up >= v0 - 1e-12 * p.spot);
        let mut higher_k = p;
        higher_k.strike *= 1.02;
        let vk = bsm::price(k, &higher_k).unwrap();
        match k {
            OptionType::Call => prop_assert!(vk <= v0 + 1e-12 * p.spot),
            OptionType::Put => prop_assert!(vk >= v0 - 1e-12 * p.spot),
        }
    }

    #[test]
    fn implied_vol_round_trips(p in params(), k in kind()) {
        let vega = bsm::greeks(k, &p).unwrap().vega;
        // Skip quotes whose time value is below double-precision resolution.
        prop_assume!(vega > 1e-6 * p.spot);
        let quote = bsm::price(k, &p).unwrap();
        let sol = iv::implied_vol(k, quote, &p).unwrap();
        let tol = (1e-12 * p.spot / vega).max(1e-9);
        prop_assert!((sol.vol - p.vol).abs() < tol, "{} vs {} via {} in {}", sol.vol, p.vol, sol.method, sol.iterations);
    }

    #[test]
    fn american_dominates_european(p in params(), k in kind()) {
        let am = BinomialTree::american(150).price(k, &p);
        let eu = BinomialTree::european(150).price(k, &p);
        // Large carry with coarse steps can make p_up leave (0, 1); that is a
        // reported error, not a wrong price.
        if let (Ok(am), Ok(eu)) = (am, eu) {
            prop_assert!(am >= eu - 1e-12 * p.spot);
            prop_assert!(am >= k.payoff(p.spot, p.strike) - 1e-12 * p.spot);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn monte_carlo_agrees_with_bsm(p in mc_params(), k in kind(), seed in any::<u64>()) {
        let exact = bsm::price(k, &p).unwrap();
        // The estimator only sees the payoff kink if a decent share of paths
        // land on each side of the strike. When one side is a rare event
        // (far-OTM option, or the far-OTM tail of a deep-ITM one) a handful
        // of hits make the sample standard error itself unreliable, so the
        // statistical check applies only when both the call and the put have
        // real value.
        let other = bsm::price(k.flip(), &p).unwrap();
        prop_assume!(exact.min(other) > 1e-3 * p.spot);
        let mc = MonteCarlo { paths: 40_000, seed, ..MonteCarlo::default() }
            .price(k, &p)
            .unwrap();
        // Five standard errors: a false alarm is a ~1-in-1.7-million event.
        let slack = 5.0 * mc.std_error;
        prop_assert!((mc.price - exact).abs() <= slack,
            "mc {} ± {} vs bsm {}", mc.price, mc.std_error, exact);
    }
}
