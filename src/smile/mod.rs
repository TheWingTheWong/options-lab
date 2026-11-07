//! Volatility smile construction from a quoted option chain.
//!
//! Pipeline: [`parse_chain`] → [`build_smile`].
//!
//! For every strike the builder solves the implied vol of the mid price.
//! By default it keeps the out-of-the-money side only (puts below the
//! forward, calls at or above it), the same convention used for VIX-style
//! calculations: OTM options are the liquid ones, and their prices are pure
//! time value. Where a strike has both a call and a put, put-call parity is
//! checked against the combined half-spreads, which flags stale quotes and
//! a mis-specified rate or dividend yield.

mod chain;
mod fit;

pub use chain::{Chain, ChainMeta, Quote, parse_chain};
pub use fit::{QuadraticFit, fit_quadratic};

use crate::bsm;
use crate::error::{Error, Result};
use crate::iv;
use crate::parity::{self, ParityCheck};
use crate::types::{OptionParams, OptionType};

/// Market data needed to turn prices into implied vols.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Market {
    /// Spot price.
    pub spot: f64,
    /// Risk-free rate.
    pub rate: f64,
    /// Continuous dividend yield.
    pub dividend: f64,
    /// Time to expiry in years.
    pub expiry: f64,
}

impl Market {
    fn params(&self, strike: f64) -> OptionParams {
        OptionParams::new(self.spot, strike, self.expiry)
            .with_rate(self.rate)
            .with_dividend(self.dividend)
    }

    /// Forward price `S e^{(r-q)T}`.
    pub fn forward(&self) -> f64 {
        self.params(self.spot).forward()
    }
}

/// Which quotes feed the smile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Selection {
    /// Out-of-the-money side at each strike (falls back to the other side
    /// if only one is quoted).
    #[default]
    OutOfTheMoney,
    /// Every quote, calls and puts.
    All,
}

/// An implied vol on the smile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmilePoint {
    /// The source quote.
    pub quote: Quote,
    /// `ln(K / F)`.
    pub log_moneyness: f64,
    /// Implied volatility of the mid.
    pub iv: f64,
    /// Vega (per 1.00 vol) at the implied vol; used as the fit weight.
    pub vega: f64,
    /// Fitted vol at this strike, when a fit was possible.
    pub fitted: Option<f64>,
}

/// A quote that could not be placed on the smile.
#[derive(Debug, Clone, PartialEq)]
pub struct Rejected {
    /// The source quote.
    pub quote: Quote,
    /// Why it was skipped.
    pub reason: String,
}

/// Parity check at one strike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrikeParity {
    /// Strike.
    pub strike: f64,
    /// Parity result; tolerance is the sum of the two half-spreads.
    pub check: ParityCheck,
}

/// A built smile.
#[derive(Debug, Clone, PartialEq)]
pub struct Smile {
    /// Market inputs used.
    pub market: Market,
    /// Forward price.
    pub forward: f64,
    /// Points sorted by strike.
    pub points: Vec<SmilePoint>,
    /// Quotes that were skipped, with reasons.
    pub rejected: Vec<Rejected>,
    /// Quadratic fit, if at least three points were available.
    pub fit: Option<QuadraticFit>,
    /// Parity checks for strikes quoted on both sides.
    pub parity: Vec<StrikeParity>,
}

/// Solve implied vols for a chain, fit a quadratic smile and check parity.
pub fn build_smile(chain: &Chain, market: Market, selection: Selection) -> Result<Smile> {
    market.params(market.spot).validate()?;
    if market.expiry <= 0.0 {
        return Err(Error::invalid("expiry", market.expiry, "must be positive"));
    }
    let forward = market.forward();

    let mut points = Vec::new();
    let mut rejected = Vec::new();
    for q in &chain.quotes {
        let otm_side = if q.strike < forward {
            OptionType::Put
        } else {
            OptionType::Call
        };
        if selection == Selection::OutOfTheMoney && q.kind != otm_side {
            let other_quoted = chain
                .quotes
                .iter()
                .any(|o| o.strike == q.strike && o.kind == otm_side);
            if other_quoted {
                continue;
            }
        }
        if q.bid == Some(0.0) {
            rejected.push(Rejected {
                quote: *q,
                reason: "zero bid".into(),
            });
            continue;
        }
        let p = market.params(q.strike);
        match iv::implied_vol(q.kind, q.mid, &p) {
            Ok(sol) => points.push(SmilePoint {
                quote: *q,
                log_moneyness: p.log_moneyness(),
                iv: sol.vol,
                vega: bsm::vega_unchecked(&p.with_vol(sol.vol)),
                fitted: None,
            }),
            Err(e) => rejected.push(Rejected {
                quote: *q,
                reason: e.to_string(),
            }),
        }
    }
    points.sort_by(|a, b| {
        a.quote
            .strike
            .total_cmp(&b.quote.strike)
            .then((a.quote.kind == OptionType::Call).cmp(&(b.quote.kind == OptionType::Call)))
    });

    let triples: Vec<_> = points
        .iter()
        .map(|p| (p.log_moneyness, p.iv, p.vega))
        .collect();
    let fit = fit_quadratic(&triples).ok();
    if let Some(f) = fit {
        for p in &mut points {
            p.fitted = Some(f.vol_at(p.log_moneyness));
        }
    }

    let mut parity_checks = Vec::new();
    for c in chain.quotes.iter().filter(|q| q.kind == OptionType::Call) {
        let Some(p) = chain
            .quotes
            .iter()
            .find(|q| q.kind == OptionType::Put && q.strike == c.strike)
        else {
            continue;
        };
        let tol = 0.5 * (c.spread().unwrap_or(0.0) + p.spread().unwrap_or(0.0));
        if let Ok(check) = parity::check(c.mid, p.mid, &market.params(c.strike), tol) {
            parity_checks.push(StrikeParity {
                strike: c.strike,
                check,
            });
        }
    }
    parity_checks.sort_by(|a, b| a.strike.total_cmp(&b.strike));

    Ok(Smile {
        market,
        forward,
        points,
        rejected,
        fit,
        parity: parity_checks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn market() -> Market {
        Market {
            spot: 100.0,
            rate: 0.03,
            dividend: 0.01,
            expiry: 0.25,
        }
    }

    /// Chain priced off a known quadratic smile, both sides at every strike.
    fn synthetic_chain(a: f64, b: f64, c: f64) -> Chain {
        let m = market();
        let mut quotes = Vec::new();
        for (i, strike) in (70..=130).step_by(5).enumerate() {
            let strike = f64::from(strike);
            let p = m.params(strike);
            let vol = a + b * p.log_moneyness() + c * p.log_moneyness().powi(2);
            for (j, kind) in [OptionType::Call, OptionType::Put].into_iter().enumerate() {
                let mid = bsm::price(kind, &p.with_vol(vol)).unwrap();
                quotes.push(Quote {
                    strike,
                    kind,
                    bid: Some(mid * 0.99),
                    ask: Some(mid * 1.01),
                    mid,
                    line: 2 * i + j + 2,
                });
            }
        }
        Chain {
            meta: ChainMeta::default(),
            quotes,
        }
    }

    #[test]
    fn recovers_the_generating_smile() {
        let chain = synthetic_chain(0.22, -0.35, 0.8);
        let smile = build_smile(&chain, market(), Selection::OutOfTheMoney).unwrap();
        assert_eq!(smile.points.len(), 13, "one OTM quote per strike");
        assert!(smile.rejected.is_empty());
        let fit = smile.fit.unwrap();
        assert!((fit.atm_vol - 0.22).abs() < 1e-8, "{fit:?}");
        assert!((fit.skew + 0.35).abs() < 1e-7, "{fit:?}");
        assert!((fit.curvature - 0.8).abs() < 1e-6, "{fit:?}");
        assert!(smile.parity.iter().all(|p| p.check.holds()));
        assert_eq!(smile.parity.len(), 13);
        // OTM convention: puts strictly below the forward, calls above.
        for p in &smile.points {
            let expected = if p.quote.strike < smile.forward {
                OptionType::Put
            } else {
                OptionType::Call
            };
            assert_eq!(p.quote.kind, expected);
        }
    }

    #[test]
    fn all_selection_keeps_both_sides() {
        let chain = synthetic_chain(0.2, 0.0, 0.0);
        let smile = build_smile(&chain, market(), Selection::All).unwrap();
        assert_eq!(smile.points.len(), 26);
        for p in &smile.points {
            assert!((p.iv - 0.2).abs() < 1e-9);
        }
    }

    #[test]
    fn rejects_unsolvable_and_zero_bid_quotes() {
        let mut chain = synthetic_chain(0.2, 0.0, 0.0);
        chain.quotes[0].mid = 1e-3; // 70 call below intrinsic
        chain.quotes[1].bid = Some(0.0); // 70 put with no bid
        let smile = build_smile(&chain, market(), Selection::All).unwrap();
        assert_eq!(smile.rejected.len(), 2);
        assert!(
            smile
                .rejected
                .iter()
                .any(|r| r.reason.contains("no-arbitrage"))
        );
        assert!(smile.rejected.iter().any(|r| r.reason == "zero bid"));
    }

    #[test]
    fn parity_flags_a_stale_quote() {
        let mut chain = synthetic_chain(0.2, 0.0, 0.0);
        let idx = chain
            .quotes
            .iter()
            .position(|q| q.strike == 100.0 && q.kind == OptionType::Call)
            .unwrap();
        chain.quotes[idx].mid += 0.5;
        let smile = build_smile(&chain, market(), Selection::OutOfTheMoney).unwrap();
        let bad: Vec<_> = smile.parity.iter().filter(|p| !p.check.holds()).collect();
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0].strike, 100.0);
    }
}
