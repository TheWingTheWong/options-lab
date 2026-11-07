//! A tour of the library API.
//!
//! ```text
//! cargo run --example quickstart
//! ```

use options_lab::binomial::BinomialTree;
use options_lab::monte_carlo::MonteCarlo;
use options_lab::{OptionParams, OptionType, bsm, iv, parity};

fn main() -> options_lab::Result<()> {
    // 30-day call, 5% out of the money.
    let p = OptionParams::new(100.0, 105.0, 30.0 / 365.0)
        .with_rate(0.03)
        .with_vol(0.20);

    let call = bsm::price(OptionType::Call, &p)?;
    let g = bsm::greeks(OptionType::Call, &p)?;
    println!("BSM call        {call:.6}");
    println!(
        "delta {:.4}  gamma {:.4}  vega/pt {:.4}  theta/day {:.4}",
        g.delta,
        g.gamma,
        g.vega_per_point(),
        g.theta_per_day()
    );

    // Back out the vol from a market quote.
    let sol = iv::implied_vol(OptionType::Call, 0.85, &p)?;
    println!(
        "IV of 0.85      {:.4}% ({} after {} iterations)",
        100.0 * sol.vol,
        sol.method,
        sol.iterations
    );

    // American put on a tree, European put by parity.
    let put_p = p.with_expiry(1.0);
    let american = BinomialTree::american(1000).valuation(OptionType::Put, &put_p)?;
    let european_call = bsm::price(OptionType::Call, &put_p)?;
    let european = parity::put_from_call(european_call, &put_p);
    println!(
        "1y put          American {:.4} vs European {:.4} (early exercise worth {:.4})",
        american.price,
        european,
        american.price - european
    );

    // Monte Carlo with variance reduction.
    let mc = MonteCarlo::default().price(OptionType::Call, &p)?;
    let (lo, hi) = mc.confidence_95();
    println!(
        "MC call         {:.6} ± {:.6}  95% CI [{lo:.4}, {hi:.4}]",
        mc.price, mc.std_error
    );
    Ok(())
}
