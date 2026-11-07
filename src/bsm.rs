//! Black-Scholes-Merton closed-form pricing with a continuous dividend yield.
//!
//! With `φ = +1` for calls and `-1` for puts:
//!
//! ```text
//! d1 = [ln(S/K) + (r - q + σ²/2) T] / (σ √T)
//! d2 = d1 - σ √T
//! V  = φ [ S e^{-qT} N(φ d1) - K e^{-rT} N(φ d2) ]
//! ```
//!
//! When `σ √T` is (numerically) zero the option is priced at its
//! deterministic limit `max(φ (S e^{-qT} - K e^{-rT}), 0)`, so expiring or
//! zero-vol contracts never produce NaN.

use crate::error::Result;
use crate::math::norm_cdf;
use crate::types::{OptionParams, OptionType};

/// Below this total volatility `σ√T` the model collapses to its deterministic limit.
pub(crate) const MIN_TOTAL_VOL: f64 = 1e-12;

/// The `d1` and `d2` terms of the Black-Scholes-Merton formula.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct D1D2 {
    /// `d1`.
    pub d1: f64,
    /// `d2 = d1 - σ√T`.
    pub d2: f64,
}

/// Compute `d1` and `d2`. Returns `None` in the degenerate `σ√T → 0` limit.
pub fn d1_d2(p: &OptionParams) -> Option<D1D2> {
    let total_vol = p.vol * p.expiry.sqrt();
    if total_vol < MIN_TOTAL_VOL {
        return None;
    }
    let d1 = ((p.spot / p.strike).ln() + (p.rate - p.dividend + 0.5 * p.vol * p.vol) * p.expiry)
        / total_vol;
    Some(D1D2 {
        d1,
        d2: d1 - total_vol,
    })
}

/// Black-Scholes-Merton price of a European option.
///
/// ```
/// use options_lab::{bsm, OptionParams, OptionType};
/// // Hull, "Options, Futures and Other Derivatives", Example 15.6.
/// let p = OptionParams::new(42.0, 40.0, 0.5).with_rate(0.10).with_vol(0.20);
/// let call = bsm::price(OptionType::Call, &p).unwrap();
/// assert!((call - 4.7594).abs() < 5e-5);
/// ```
pub fn price(kind: OptionType, p: &OptionParams) -> Result<f64> {
    p.validate()?;
    Ok(price_unchecked(kind, p))
}

/// [`price`] without input validation, for hot loops that validated once up front.
pub(crate) fn price_unchecked(kind: OptionType, p: &OptionParams) -> f64 {
    let phi = kind.sign();
    let fwd_spot = p.spot * p.dividend_discount();
    let pv_strike = p.strike * p.discount();
    match d1_d2(p) {
        Some(D1D2 { d1, d2 }) => {
            phi * (fwd_spot * norm_cdf(phi * d1) - pv_strike * norm_cdf(phi * d2))
        }
        None => (phi * (fwd_spot - pv_strike)).max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn hull_example_15_6() {
        let p = OptionParams::new(42.0, 40.0, 0.5)
            .with_rate(0.10)
            .with_vol(0.20);
        assert!(close(
            price(OptionType::Call, &p).unwrap(),
            4.759_422_392_871_532,
            1e-12
        ));
        assert!(close(
            price(OptionType::Put, &p).unwrap(),
            0.808_599_372_900_092_2,
            1e-12
        ));
    }

    #[test]
    fn haug_index_put_with_dividend_yield() {
        // Haug, "The Complete Guide to Option Pricing Formulas", Merton (1973) example.
        let p = OptionParams::new(100.0, 95.0, 0.5)
            .with_rate(0.10)
            .with_dividend(0.05)
            .with_vol(0.20);
        assert!(close(
            price(OptionType::Put, &p).unwrap(),
            2.464_787_646_755_830_5,
            1e-12
        ));
    }

    #[test]
    fn expired_option_is_intrinsic() {
        let p = OptionParams::new(110.0, 100.0, 0.0).with_vol(0.3);
        assert_eq!(price(OptionType::Call, &p).unwrap(), 10.0);
        assert_eq!(price(OptionType::Put, &p).unwrap(), 0.0);
    }

    #[test]
    fn zero_vol_is_discounted_forward_intrinsic() {
        let p = OptionParams::new(100.0, 100.0, 1.0)
            .with_rate(0.05)
            .with_vol(0.0);
        let expected = 100.0 - 100.0 * (-0.05_f64).exp();
        assert!(close(price(OptionType::Call, &p).unwrap(), expected, 1e-12));
        assert_eq!(price(OptionType::Put, &p).unwrap(), 0.0);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let p = OptionParams::new(-1.0, 100.0, 1.0);
        assert!(price(OptionType::Call, &p).is_err());
    }
}
