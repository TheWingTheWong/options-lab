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
//!
//! ## Greek conventions
//!
//! All sensitivities are raw partial derivatives in model units:
//! `vega` and `rho` are per 1.00 change in `σ` / `r` (divide by 100 for
//! "per vol point" / "per 1%"), and `theta` is `∂V/∂t` per year (divide by
//! 365 for a calendar-day theta). [`Greeks::vega_per_point`] and
//! [`Greeks::theta_per_day`] do the common conversions.

use crate::error::Result;
use crate::math::{norm_cdf, norm_pdf};
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

/// First- and second-order sensitivities of a European option.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Greeks {
    /// `∂V/∂S`.
    pub delta: f64,
    /// `∂²V/∂S²`.
    pub gamma: f64,
    /// `∂V/∂σ`, per 1.00 of volatility.
    pub vega: f64,
    /// `∂V/∂t = -∂V/∂T`, per year.
    pub theta: f64,
    /// `∂V/∂r`, per 1.00 of rate.
    pub rho: f64,
    /// `∂²V/∂S∂σ` (equivalently `∂Δ/∂σ`).
    pub vanna: f64,
    /// `∂²V/∂σ²`, also known as vomma.
    pub volga: f64,
}

impl Greeks {
    /// Vega for a one vol-point (0.01) move.
    pub fn vega_per_point(&self) -> f64 {
        self.vega / 100.0
    }

    /// Theta per calendar day.
    pub fn theta_per_day(&self) -> f64 {
        self.theta / crate::types::DAYS_PER_YEAR
    }

    /// Rho for a one percentage point (0.01) move in rates.
    pub fn rho_per_point(&self) -> f64 {
        self.rho / 100.0
    }
}

/// Analytic Black-Scholes-Merton Greeks.
///
/// ```
/// use options_lab::{bsm, OptionParams, OptionType};
/// let p = OptionParams::new(100.0, 100.0, 1.0).with_rate(0.05).with_dividend(0.02).with_vol(0.25);
/// let g = bsm::greeks(OptionType::Call, &p).unwrap();
/// assert!((g.delta - 0.584_954_911).abs() < 1e-8);
/// assert!((g.vega - 37.948_089_225).abs() < 1e-8);
/// ```
pub fn greeks(kind: OptionType, p: &OptionParams) -> Result<Greeks> {
    p.validate()?;
    Ok(greeks_unchecked(kind, p))
}

pub(crate) fn greeks_unchecked(kind: OptionType, p: &OptionParams) -> Greeks {
    let phi = kind.sign();
    let dq = p.dividend_discount();
    let dr = p.discount();
    let fwd_spot = p.spot * dq;
    let pv_strike = p.strike * dr;

    let Some(D1D2 { d1, d2 }) = d1_d2(p) else {
        // Deterministic limit: the option is either a discounted forward or worthless.
        let in_the_money = phi * (fwd_spot - pv_strike) > 0.0;
        if !in_the_money {
            return Greeks::default();
        }
        return Greeks {
            delta: phi * dq,
            theta: phi * (p.dividend * fwd_spot - p.rate * pv_strike),
            rho: phi * p.expiry * pv_strike,
            ..Greeks::default()
        };
    };

    let sqrt_t = p.expiry.sqrt();
    let pdf_d1 = norm_pdf(d1);
    let n_d1 = norm_cdf(phi * d1);
    let n_d2 = norm_cdf(phi * d2);
    let vega = fwd_spot * pdf_d1 * sqrt_t;

    Greeks {
        delta: phi * dq * n_d1,
        gamma: dq * pdf_d1 / (p.spot * p.vol * sqrt_t),
        vega,
        theta: -fwd_spot * pdf_d1 * p.vol / (2.0 * sqrt_t) - phi * p.rate * pv_strike * n_d2
            + phi * p.dividend * fwd_spot * n_d1,
        rho: phi * p.expiry * pv_strike * n_d2,
        vanna: -dq * pdf_d1 * d2 / p.vol,
        volga: vega * d1 * d2 / p.vol,
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
    #[test]
    fn greeks_match_reference_values() {
        // Reference values computed independently with scipy.
        let p = OptionParams::new(100.0, 100.0, 1.0)
            .with_rate(0.05)
            .with_dividend(0.02)
            .with_vol(0.25);
        let g = greeks(OptionType::Call, &p).unwrap();
        assert!(close(g.delta, 0.584_954_911_257_883, 1e-12));
        assert!(close(g.gamma, 0.015_179_235_690_178_282, 1e-12));
        assert!(close(g.vega, 37.948_089_225_445_706, 1e-10));
        assert!(close(g.theta, -5.942_187_790_551_456, 1e-10));
        assert!(close(g.rho, 47.371_729_197_730_18, 1e-10));
        assert!(close(g.vanna, 0.007_589_617_845_089_147, 1e-12));
        assert!(close(g.volga, -0.185_945_637_204_684_13, 1e-10));
    }

    #[test]
    fn greeks_match_finite_differences() {
        let base = OptionParams::new(95.0, 105.0, 0.75)
            .with_rate(0.04)
            .with_dividend(0.015)
            .with_vol(0.32);
        for kind in [OptionType::Call, OptionType::Put] {
            let g = greeks(kind, &base).unwrap();
            let v = |p: OptionParams| price_unchecked(kind, &p);
            let h = 1e-4;

            let up = |f: fn(&mut OptionParams, f64)| {
                let mut a = base;
                f(&mut a, h);
                let mut b = base;
                f(&mut b, -h);
                (a, b)
            };
            let (s_up, s_dn) = up(|p, h| p.spot += h);
            let (v_up, v_dn) = up(|p, h| p.vol += h);
            let (r_up, r_dn) = up(|p, h| p.rate += h);
            let (t_up, t_dn) = up(|p, h| p.expiry += h);

            let fd_delta = (v(s_up) - v(s_dn)) / (2.0 * h);
            let fd_gamma = (v(s_up) - 2.0 * v(base) + v(s_dn)) / (h * h);
            let fd_vega = (v(v_up) - v(v_dn)) / (2.0 * h);
            let fd_rho = (v(r_up) - v(r_dn)) / (2.0 * h);
            let fd_theta = -(v(t_up) - v(t_dn)) / (2.0 * h);
            let fd_volga = (v(v_up) - 2.0 * v(base) + v(v_dn)) / (h * h);
            let delta_at = |p: OptionParams| greeks_unchecked(kind, &p).delta;
            let fd_vanna = (delta_at(v_up) - delta_at(v_dn)) / (2.0 * h);

            assert!(close(g.delta, fd_delta, 1e-7), "{kind} delta");
            assert!(close(g.gamma, fd_gamma, 1e-5), "{kind} gamma");
            assert!(close(g.vega, fd_vega, 1e-6), "{kind} vega");
            assert!(close(g.rho, fd_rho, 1e-6), "{kind} rho");
            assert!(close(g.theta, fd_theta, 1e-6), "{kind} theta");
            assert!(close(g.vanna, fd_vanna, 1e-7), "{kind} vanna");
            assert!(close(g.volga, fd_volga, 1e-3), "{kind} volga");
        }
    }

    #[test]
    fn call_and_put_share_gamma_vega_vanna_volga() {
        let p = OptionParams::new(100.0, 90.0, 0.3)
            .with_rate(0.02)
            .with_dividend(0.01)
            .with_vol(0.4);
        let c = greeks(OptionType::Call, &p).unwrap();
        let q = greeks(OptionType::Put, &p).unwrap();
        assert!(close(c.gamma, q.gamma, 1e-14));
        assert!(close(c.vega, q.vega, 1e-12));
        assert!(close(c.vanna, q.vanna, 1e-14));
        assert!(close(c.volga, q.volga, 1e-12));
        // Delta parity: Δc - Δp = e^{-qT}.
        assert!(close(c.delta - q.delta, p.dividend_discount(), 1e-14));
    }

    #[test]
    fn degenerate_greeks_are_finite() {
        let p = OptionParams::new(120.0, 100.0, 0.0).with_vol(0.2);
        let g = greeks(OptionType::Call, &p).unwrap();
        assert_eq!(g.delta, 1.0);
        assert_eq!(g.gamma, 0.0);
        let otm = greeks(OptionType::Put, &p).unwrap();
        assert_eq!(otm, Greeks::default());
    }
}
