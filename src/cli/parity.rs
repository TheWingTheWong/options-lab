use clap::Args;
use options_lab::parity;

use super::MarketArgs;

/// Check a call/put pair against put-call parity.
#[derive(Args, Debug)]
pub struct ParityArgs {
    #[command(flatten)]
    pub market: MarketArgs,
    /// Call price.
    #[arg(long)]
    pub call: f64,
    /// Put price.
    #[arg(long)]
    pub put: f64,
    /// Absolute tolerance for the parity gap.
    #[arg(long, default_value_t = 0.01)]
    pub tol: f64,
}

pub fn run(args: &ParityArgs) -> options_lab::Result<String> {
    let p = args.market.params(0.0);
    let chk = parity::check(args.call, args.put, &p, args.tol)?;
    let q = parity::implied_dividend(args.call, args.put, &p.with_dividend(0.0))?;
    Ok(format!(
        "C - P                {:+.6}\n\
         S e^-qT - K e^-rT    {:+.6}\n\
         gap                  {:+.6}  ({})\n\
         implied forward      {:.6}  (model forward {:.6})\n\
         implied div yield    {:.4}%\n",
        chk.lhs,
        chk.rhs,
        chk.gap,
        if chk.holds() {
            format!("within tolerance {}", args.tol)
        } else if chk.gap > 0.0 {
            "VIOLATED: calls rich vs puts".to_string()
        } else {
            "VIOLATED: puts rich vs calls".to_string()
        },
        chk.implied_forward,
        p.forward(),
        100.0 * q
    ))
}
