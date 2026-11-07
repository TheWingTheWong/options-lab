//! Put-call parity for European options.
//!
//! ```text
//! C - P = S e^{-qT} - K e^{-rT}
//! ```
//!
//! Parity is model-free: it holds for any arbitrage-free price, which makes it
//! a cheap consistency check on quoted chains and a way to back out the
//! forward (or the market-implied dividend yield) from a call/put pair.

use crate::error::{Error, Result};
use crate::types::OptionParams;

/// Outcome of checking a call/put pair against parity. `vol` is ignored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityCheck {
    /// Call price.
    pub call: f64,
    /// Put price.
    pub put: f64,
    /// `C - P`.
    pub lhs: f64,
    /// `S e^{-qT} - K e^{-rT}`.
    pub rhs: f64,
    /// `lhs - rhs`. Positive means calls are rich relative to puts.
    pub gap: f64,
    /// Absolute tolerance used for [`ParityCheck::holds`].
    pub tolerance: f64,
    /// Forward implied by the pair: `K + (C - P) e^{rT}`.
    pub implied_forward: f64,
}

impl ParityCheck {
    /// Whether `|gap| <= tolerance`.
    pub fn holds(&self) -> bool {
        self.gap.abs() <= self.tolerance
    }
}

/// Compare a call and a put with the same strike and expiry against parity.
///
/// ```
/// use options_lab::{bsm, parity, OptionParams, OptionType};
/// let p = OptionParams::new(100.0, 100.0, 0.5).with_rate(0.03).with_vol(0.25);
/// let c = bsm::price(OptionType::Call, &p).unwrap();
/// let q = bsm::price(OptionType::Put, &p).unwrap();
/// assert!(parity::check(c, q, &p, 1e-10).unwrap().holds());
/// ```
pub fn check(call: f64, put: f64, p: &OptionParams, tolerance: f64) -> Result<ParityCheck> {
    p.validate()?;
    if !(call.is_finite() && call >= 0.0) {
        return Err(Error::invalid("call", call, "must be a non-negative price"));
    }
    if !(put.is_finite() && put >= 0.0) {
        return Err(Error::invalid("put", put, "must be a non-negative price"));
    }
    let lhs = call - put;
    let rhs = p.spot * p.dividend_discount() - p.strike * p.discount();
    Ok(ParityCheck {
        call,
        put,
        lhs,
        rhs,
        gap: lhs - rhs,
        tolerance,
        implied_forward: p.strike + lhs / p.discount(),
    })
}

/// Put price implied by a call price through parity.
pub fn put_from_call(call: f64, p: &OptionParams) -> f64 {
    call - p.spot * p.dividend_discount() + p.strike * p.discount()
}

/// Call price implied by a put price through parity.
pub fn call_from_put(put: f64, p: &OptionParams) -> f64 {
    put + p.spot * p.dividend_discount() - p.strike * p.discount()
}

/// Continuous dividend yield implied by a call/put pair, given the rate.
///
/// Solves `S e^{-qT} = C - P + K e^{-rT}` for `q`.
pub fn implied_dividend(call: f64, put: f64, p: &OptionParams) -> Result<f64> {
    p.validate()?;
    if p.expiry <= 0.0 {
        return Err(Error::invalid("expiry", p.expiry, "must be positive"));
    }
    let pv_spot = call - put + p.strike * p.discount();
    if pv_spot <= 0.0 {
        return Err(Error::invalid(
            "call - put",
            call - put,
            "implies a non-positive dividend-discounted spot",
        ));
    }
    Ok(-(pv_spot / p.spot).ln() / p.expiry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptionType;
    use crate::bsm;

    fn params() -> OptionParams {
        OptionParams::new(100.0, 110.0, 0.75)
            .with_rate(0.04)
            .with_dividend(0.015)
            .with_vol(0.3)
    }

    #[test]
    fn bsm_prices_satisfy_parity() {
        let p = params();
        let c = bsm::price(OptionType::Call, &p).unwrap();
        let q = bsm::price(OptionType::Put, &p).unwrap();
        let chk = check(c, q, &p, 1e-12).unwrap();
        assert!(chk.holds(), "gap = {}", chk.gap);
        assert!((chk.implied_forward - p.forward()).abs() < 1e-10);
        assert!((put_from_call(c, &p) - q).abs() < 1e-12);
        assert!((call_from_put(q, &p) - c).abs() < 1e-12);
    }

    #[test]
    fn detects_mispriced_pair() {
        let p = params();
        let c = bsm::price(OptionType::Call, &p).unwrap();
        let q = bsm::price(OptionType::Put, &p).unwrap();
        let chk = check(c + 0.25, q, &p, 0.01).unwrap();
        assert!(!chk.holds());
        assert!((chk.gap - 0.25).abs() < 1e-12);
    }

    #[test]
    fn recovers_dividend_yield() {
        let p = params();
        let c = bsm::price(OptionType::Call, &p).unwrap();
        let q = bsm::price(OptionType::Put, &p).unwrap();
        let q_hat = implied_dividend(c, q, &p.with_dividend(0.0)).unwrap();
        assert!((q_hat - 0.015).abs() < 1e-12);
    }
}
