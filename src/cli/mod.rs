//! Command-line front end. Everything numeric lives in the library; this
//! module only parses arguments and formats reports.

pub mod iv;
pub mod parity;
pub mod price;
pub mod smile;
pub mod table;

use clap::Args;
use options_lab::{DAYS_PER_YEAR, OptionParams, OptionType};

/// What every subcommand returns: a report to print, or an error message.
pub type CliResult = Result<String, Box<dyn std::error::Error>>;

/// Market inputs shared by the single-option subcommands.
#[derive(Args, Debug, Clone)]
pub struct MarketArgs {
    /// Spot price of the underlying.
    #[arg(long)]
    pub spot: f64,
    /// Strike price.
    #[arg(long)]
    pub strike: f64,
    /// Risk-free rate, continuously compounded (0.03 = 3%).
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub rate: f64,
    /// Continuous dividend yield (0.01 = 1%).
    #[arg(
        long = "div",
        visible_alias = "dividend",
        default_value_t = 0.0,
        allow_negative_numbers = true
    )]
    pub dividend: f64,
    #[command(flatten)]
    pub expiry: ExpiryArgs,
}

/// Time to expiry, either in calendar days or in years.
#[derive(Args, Debug, Clone)]
#[group(required = true, multiple = false)]
pub struct ExpiryArgs {
    /// Calendar days to expiry (converted with a 365-day year).
    #[arg(long)]
    pub days: Option<f64>,
    /// Years to expiry.
    #[arg(long)]
    pub years: Option<f64>,
}

impl ExpiryArgs {
    pub fn years(&self) -> f64 {
        match (self.days, self.years) {
            (Some(d), _) => d / DAYS_PER_YEAR,
            (None, Some(y)) => y,
            (None, None) => unreachable!("clap enforces one of --days/--years"),
        }
    }
}

impl MarketArgs {
    pub fn params(&self, vol: f64) -> OptionParams {
        OptionParams::new(self.spot, self.strike, self.expiry.years())
            .with_rate(self.rate)
            .with_dividend(self.dividend)
            .with_vol(vol)
    }
}

/// clap value parser for `--type`.
pub fn parse_kind(s: &str) -> Result<OptionType, String> {
    s.parse().map_err(|e: options_lab::Error| e.to_string())
}

/// One-line description of the contract and market.
pub fn describe(kind: OptionType, p: &OptionParams, show_vol: bool) -> String {
    let days = p.expiry * DAYS_PER_YEAR;
    let mut s = format!(
        "{kind}  S={}  K={}  T={}d ({:.4}y)",
        num(p.spot),
        num(p.strike),
        num(days),
        p.expiry
    );
    if show_vol {
        s.push_str(&format!("  vol={}", pct(p.vol)));
    }
    s.push_str(&format!("  r={}  q={}", pct(p.rate), pct(p.dividend)));
    s
}

/// Format a percentage with two decimals.
pub fn pct(x: f64) -> String {
    format!("{:.2}%", 100.0 * x)
}

/// Shortest faithful rendering of a user-supplied number (100, 102.5, 0.25).
pub fn num(x: f64) -> String {
    let s = format!("{x:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}
