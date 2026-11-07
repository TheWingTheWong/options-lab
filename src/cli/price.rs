use clap::Args;
use options_lab::binomial::BinomialTree;
use options_lab::monte_carlo::MonteCarlo;
use options_lab::{OptionType, bsm, parity};

use super::table::{Align, render};
use super::{MarketArgs, describe, parse_kind};

/// Price an option and report Greeks and a cross-model comparison.
#[derive(Args, Debug)]
pub struct PriceArgs {
    #[command(flatten)]
    pub market: MarketArgs,
    /// Volatility (0.2 = 20%).
    #[arg(long)]
    pub vol: f64,
    /// Option type: call or put.
    #[arg(long = "type", value_parser = parse_kind)]
    pub kind: OptionType,
    /// Binomial tree steps.
    #[arg(long, default_value_t = 1000)]
    pub steps: usize,
    /// Monte Carlo paths.
    #[arg(long, default_value_t = 200_000)]
    pub paths: usize,
    /// Monte Carlo seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Only print the Black-Scholes-Merton section.
    #[arg(long)]
    pub bsm_only: bool,
}

pub fn run(args: &PriceArgs) -> super::CliResult {
    let p = args.market.params(args.vol);
    let kind = args.kind;
    let price = bsm::price(kind, &p)?;
    let g = bsm::greeks(kind, &p)?;

    let mut out = describe(kind, &p, true);
    out.push_str(&format!(
        "\nforward {:.4}   ln(K/F) {:+.4}   d1/d2 {}\n",
        p.forward(),
        p.log_moneyness(),
        bsm::d1_d2(&p)
            .map(|d| format!("{:+.4} / {:+.4}", d.d1, d.d2))
            .unwrap_or_else(|| "n/a (zero total variance)".into())
    ));

    out.push_str("\nBlack-Scholes-Merton, European\n");
    let rows: Vec<Vec<String>> = [
        ("price", price, ""),
        ("delta", g.delta, "per 1.00 spot"),
        ("gamma", g.gamma, "delta per 1.00 spot"),
        ("vega", g.vega_per_point(), "per vol point"),
        ("theta", g.theta_per_day(), "per calendar day"),
        ("rho", g.rho_per_point(), "per 1% rate"),
        ("vanna", g.vanna / 100.0, "delta per vol point"),
        (
            "volga",
            g.volga / 10_000.0,
            "vega per vol point, per vol point",
        ),
    ]
    .iter()
    .map(|(name, v, unit)| vec![name.to_string(), format!("{v:.6}"), unit.to_string()])
    .collect();
    out.push_str(&render(
        &["greek", "value", "unit"],
        &[Align::Left, Align::Right, Align::Left],
        &rows,
    ));
    out.push('\n');

    if args.bsm_only {
        return Ok(out);
    }

    let eu = BinomialTree::european(args.steps).valuation(kind, &p)?;
    let am = BinomialTree::american(args.steps).valuation(kind, &p)?;
    let mc_cfg = MonteCarlo {
        paths: args.paths,
        seed: args.seed,
        ..MonteCarlo::default()
    };
    let mc = mc_cfg.price(kind, &p)?;
    let plain = MonteCarlo::plain(args.paths, args.seed).price(kind, &p)?;
    let other = match kind {
        OptionType::Call => parity::put_from_call(price, &p),
        OptionType::Put => parity::call_from_put(price, &p),
    };

    let rows = vec![
        vec![
            "Black-Scholes-Merton".into(),
            format!("{price:.6}"),
            "closed form".into(),
        ],
        vec![
            format!("CRR tree, European, {} steps", args.steps),
            format!("{:.6}", eu.price),
            format!("{:+.6} vs BSM", eu.price - price),
        ],
        vec![
            format!("CRR tree, American, {} steps", args.steps),
            format!("{:.6}", am.price),
            format!("early-exercise value {:+.6}", am.price - eu.price),
        ],
        vec![
            format!("Monte Carlo, {} paths, plain", args.paths),
            format!("{:.6}", plain.price),
            format!("s.e. {:.6}", plain.std_error),
        ],
        vec![
            format!("Monte Carlo, {} paths, AV+CV", args.paths),
            format!("{:.6}", mc.price),
            format!(
                "s.e. {:.6} ({:.1}x smaller)",
                mc.std_error,
                plain.std_error / mc.std_error
            ),
        ],
        vec![
            format!("{} via put-call parity", kind.flip()),
            format!("{other:.6}"),
            format!("from the BSM {kind}"),
        ],
    ];
    out.push_str("\nModel comparison\n");
    out.push_str(&render(
        &["model", "price", "note"],
        &[Align::Left, Align::Right, Align::Left],
        &rows,
    ));
    out.push('\n');
    Ok(out)
}
