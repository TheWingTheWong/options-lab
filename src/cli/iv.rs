use clap::Args;
use options_lab::{OptionType, bsm, iv};

use super::table::{Align, render};
use super::{MarketArgs, describe, parse_kind};

/// Solve for the implied volatility of a quoted option price.
#[derive(Args, Debug)]
pub struct IvArgs {
    #[command(flatten)]
    pub market: MarketArgs,
    /// Quoted option price.
    #[arg(long)]
    pub price: f64,
    /// Option type: call or put.
    #[arg(long = "type", value_parser = parse_kind)]
    pub kind: OptionType,
}

pub fn run(args: &IvArgs) -> super::CliResult {
    let p = args.market.params(0.0);
    let sol = iv::implied_vol(args.kind, args.price, &p)?;
    let solved = p.with_vol(sol.vol);
    let g = bsm::greeks(args.kind, &solved)?;
    let reprice = bsm::price(args.kind, &solved)?;

    let mut out = describe(args.kind, &p, false);
    out.push_str(&format!("  price={}\n\n", args.price));
    let rows = vec![
        vec!["implied vol".into(), format!("{:.4}%", 100.0 * sol.vol)],
        vec![
            "solver".into(),
            format!("{} ({} iterations)", sol.method, sol.iterations),
        ],
        vec!["repriced".into(), format!("{reprice:.10}")],
        vec!["residual".into(), format!("{:.2e}", sol.residual)],
        vec![
            "vega".into(),
            format!("{:.6} per vol point", g.vega_per_point()),
        ],
        vec!["delta".into(), format!("{:.6}", g.delta)],
    ];
    out.push_str(
        &render(&["", ""], &[Align::Left, Align::Left], &rows)
            .lines()
            .skip(2)
            .collect::<Vec<_>>()
            .join("\n"),
    );
    out.push('\n');
    Ok(out)
}
