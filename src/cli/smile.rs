use std::path::PathBuf;

use clap::Args;
use options_lab::DAYS_PER_YEAR;
use options_lab::smile::{self, ChartOptions, Market, Selection};

use super::table::{Align, render};
use super::{num, pct};

/// Fit an implied-vol smile to a quoted chain and draw it.
#[derive(Args, Debug)]
pub struct SmileArgs {
    /// Chain CSV: `strike,type,bid,ask` (or `mid`), with optional
    /// `# spot=..., rate=..., dividend=..., days=...` lines.
    pub file: PathBuf,
    /// Spot price (overrides the file).
    #[arg(long)]
    pub spot: Option<f64>,
    /// Risk-free rate (overrides the file).
    #[arg(long, allow_negative_numbers = true)]
    pub rate: Option<f64>,
    /// Dividend yield (overrides the file).
    #[arg(
        long = "div",
        visible_alias = "dividend",
        allow_negative_numbers = true
    )]
    pub dividend: Option<f64>,
    /// Calendar days to expiry (overrides the file).
    #[arg(long, conflicts_with = "years")]
    pub days: Option<f64>,
    /// Years to expiry (overrides the file).
    #[arg(long)]
    pub years: Option<f64>,
    /// Use calls and puts at every strike instead of the OTM side only.
    #[arg(long)]
    pub all: bool,
    /// Chart width in columns.
    #[arg(long, default_value_t = 64)]
    pub width: usize,
    /// Chart height in rows (approximate; ticks snap to whole rows).
    #[arg(long, default_value_t = 16)]
    pub height: usize,
    /// Skip the chart.
    #[arg(long)]
    pub no_chart: bool,
}

pub fn run(args: &SmileArgs) -> super::CliResult {
    let text = std::fs::read_to_string(&args.file)
        .map_err(|e| format!("cannot read {}: {e}", args.file.display()))?;
    let chain = smile::parse_chain(&text)?;
    let meta = chain.meta;
    let market = Market {
        spot: args
            .spot
            .or(meta.spot)
            .ok_or("spot not given: use --spot or a '# spot=' line")?,
        rate: args.rate.or(meta.rate).unwrap_or(0.0),
        dividend: args.dividend.or(meta.dividend).unwrap_or(0.0),
        expiry: args
            .days
            .map(|d| d / DAYS_PER_YEAR)
            .or(args.years)
            .or(meta.expiry)
            .ok_or("expiry not given: use --days or a '# days=' line")?,
    };
    let selection = if args.all {
        Selection::All
    } else {
        Selection::OutOfTheMoney
    };
    let s = smile::build_smile(&chain, market, selection)?;

    let mut out = format!(
        "{}: {} quotes, {} days ({:.4}y), spot {}, r {}, q {}, forward {:.2}\n\n",
        args.file.display(),
        chain.quotes.len(),
        num(market.expiry * DAYS_PER_YEAR),
        market.expiry,
        num(market.spot),
        pct(market.rate),
        pct(market.dividend),
        s.forward
    );

    let price = |v: Option<f64>| v.map(|x| format!("{x:.2}")).unwrap_or_else(|| "-".into());
    let rows: Vec<Vec<String>> = s
        .points
        .iter()
        .map(|p| {
            vec![
                num(p.quote.strike),
                p.quote.kind.to_string(),
                price(p.quote.bid),
                price(p.quote.ask),
                format!("{:.3}", p.quote.mid),
                format!("{:+.4}", p.log_moneyness),
                pct(p.iv),
                p.fitted.map(pct).unwrap_or_else(|| "-".into()),
                p.fitted
                    .map(|f| format!("{:+.2}", 100.0 * (p.iv - f)))
                    .unwrap_or_else(|| "-".into()),
            ]
        })
        .collect();
    out.push_str(&render(
        &[
            "strike", "side", "bid", "ask", "mid", "ln(K/F)", "iv", "fit", "resid",
        ],
        &[
            Align::Right,
            Align::Left,
            Align::Right,
            Align::Right,
            Align::Right,
            Align::Right,
            Align::Right,
            Align::Right,
            Align::Right,
        ],
        &rows,
    ));
    out.push_str("\n\n");

    match s.fit {
        Some(f) => out.push_str(&format!(
            "fit     vol(k) = a + b*k + c*k^2 on k = ln(K/F), vega-weighted, {} points\n        \
             ATM vol {}   skew {:+.4}   curvature {:+.4}   RMSE {:.2} vol pts\n",
            f.points,
            pct(f.atm_vol),
            f.skew,
            f.curvature,
            100.0 * f.rmse
        )),
        None => out.push_str("fit     not enough implied vols for a quadratic fit\n"),
    }

    let violations: Vec<_> = s.parity.iter().filter(|p| !p.check.holds()).collect();
    out.push_str(&format!(
        "parity  {} strikes quoted on both sides, {} outside the bid-ask",
        s.parity.len(),
        violations.len()
    ));
    for v in &violations {
        out.push_str(&format!(
            "\n        K={} gap {:+.4} vs tolerance {:.4}",
            num(v.strike),
            v.check.gap,
            v.check.tolerance
        ));
    }
    out.push('\n');

    if !s.rejected.is_empty() {
        out.push_str(&format!("skipped {} quotes:", s.rejected.len()));
        for r in &s.rejected {
            out.push_str(&format!(
                "\n        line {}: {} {} ({})",
                r.quote.line,
                num(r.quote.strike),
                r.quote.kind,
                r.reason
            ));
        }
        out.push('\n');
    }

    if !args.no_chart {
        out.push('\n');
        out.push_str(&smile::render_chart(
            &s,
            &ChartOptions {
                width: args.width,
                height: args.height,
            },
        ));
    }
    Ok(out)
}
